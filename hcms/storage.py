import psycopg2
from psycopg2.extras import RealDictCursor, Json
import time

class PostgresStorageProvider:
    def __init__(self, dsn: str):
        self.dsn = dsn
        self._init_db()

    def _get_connection(self):
        return psycopg2.connect(self.dsn, cursor_factory=RealDictCursor)

    def _init_db(self):
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                # 1. Extensões base
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                
                # 2. Garante a existência da tabela (estrutura básica)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS memories (
                        id TEXT PRIMARY KEY,
                        content TEXT,
                        embedding vector(384),
                        fts_tokens tsvector,
                        metadata JSONB,
                        importance FLOAT DEFAULT 0.5,
                        creation_time DOUBLE PRECISION
                    );
                """)

                # 3. MIGRAÇÃO: Adiciona colunas que podem estar faltando de versões antigas
                cur.execute("ALTER TABLE memories ADD COLUMN IF NOT EXISTS is_permanent BOOLEAN DEFAULT FALSE;")
                cur.execute("ALTER TABLE memories ADD COLUMN IF NOT EXISTS last_accessed DOUBLE PRECISION;")
                cur.execute("ALTER TABLE memories ADD COLUMN IF NOT EXISTS access_count INTEGER DEFAULT 0;")

                # 4. Tabela de Co-ativações
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS coactivations (
                        id_a TEXT,
                        id_b TEXT,
                        strength FLOAT DEFAULT 1.0,
                        PRIMARY KEY (id_a, id_b)
                    );
                """)
                cur.execute("CREATE INDEX IF NOT EXISTS idx_coact_lookup ON coactivations(id_a, id_b);")

                # 5. Índices de Performance
                cur.execute("CREATE INDEX IF NOT EXISTS idx_mem_vec ON memories USING hnsw (embedding vector_cosine_ops);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_mem_fts ON memories USING GIN (fts_tokens);")
                
                # 6. ÍNDICE PARCIAL: (Agora a coluna is_permanent já existe com certeza)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_memories_decay_target 
                    ON memories(last_accessed) 
                    WHERE is_permanent = FALSE;
                """)
                
                # 7. Trigger de FTS Dual
                cur.execute("""
                    CREATE OR REPLACE FUNCTION memories_fts_trigger() RETURNS trigger AS $$
                    BEGIN
                        NEW.fts_tokens :=
                            to_tsvector('portuguese', COALESCE(NEW.content, '')) ||
                            to_tsvector('simple', COALESCE(NEW.content, ''));
                        RETURN NEW;
                    END
                    $$ LANGUAGE plpgsql;
                """)
                cur.execute("DROP TRIGGER IF EXISTS trg_memories_fts_update ON memories;")
                cur.execute("CREATE TRIGGER trg_memories_fts_update BEFORE INSERT OR UPDATE ON memories FOR EACH ROW EXECUTE FUNCTION memories_fts_trigger();")
                
                conn.commit()


    def upsert_memory(self, mem_id, content, embedding, metadata, importance=0.5, is_permanent=False):
        now = time.time()
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO memories (id, content, embedding, metadata, importance, is_permanent, last_accessed, creation_time)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        content = EXCLUDED.content,
                        importance = GREATEST(memories.importance, EXCLUDED.importance),
                        is_permanent = memories.is_permanent OR EXCLUDED.is_permanent,
                        last_accessed = EXCLUDED.last_accessed;
                """, (mem_id, content, embedding, Json(metadata or {}), importance, is_permanent, now, now))
                conn.commit()

    def update_access(self, mem_ids: list, timestamp: float):
        if not mem_ids: return
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE memories m SET
                        last_accessed = %s,
                        access_count = access_count + 1
                    FROM unnest(%s::text[]) AS u(id)
                    WHERE m.id = u.id
                """, (timestamp, mem_ids))
                conn.commit()

    def record_coactivation(self, pairs: list):
        if not pairs: return
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                for a, b in pairs:
                    id1, id2 = sorted([a, b])
                    cur.execute("""
                        INSERT INTO coactivations (id_a, id_b, strength) VALUES (%s, %s, 1.0)
                        ON CONFLICT (id_a, id_b) DO UPDATE SET strength = coactivations.strength + 0.1;
                    """, (id1, id2))
                conn.commit()

    def get_coactivation_scores(self, target_ids: list, context_ids: list):
        if not target_ids or not context_ids: return []
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id_a, id_b, strength::FLOAT FROM coactivations
                    WHERE (id_a = ANY(%s) AND id_b = ANY(%s))
                       OR (id_b = ANY(%s) AND id_a = ANY(%s))
                """, (target_ids, context_ids, target_ids, context_ids))
                return cur.fetchall()

    def fetch_all(self, query, params=None):
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                return cur.fetchall()

    def execute(self, query, params=None):
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                conn.commit()