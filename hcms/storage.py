import psycopg2
from psycopg2.extras import RealDictCursor, Json
import time


class PostgresStorageProvider:
    def __init__(self, dsn: str):
        self.dsn = dsn
        self._init_db()

    def _get_connection(self):
        return psycopg2.connect(self.dsn, cursor_factory=RealDictCursor)

    # =========================================================================
    # INIT DB (Schema + índices + FTS)
    # =========================================================================
    def _init_db(self):
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                # Extensões
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

                # -------------------------------
                # Tabela principal de memórias
                # -------------------------------
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS memories (
                        id TEXT PRIMARY KEY,
                        content TEXT,

                        -- Embedding principal
                        embedding vector(384),

                        -- Full-Text Search
                        fts_tokens tsvector,

                        -- Metadados
                        metadata JSONB,

                        -- Importância semântica
                        importance FLOAT DEFAULT 1.0,

                        -- Gestão de tiers
                        tier INTEGER DEFAULT 0,
                        access_count INTEGER DEFAULT 0,
                        last_access DOUBLE PRECISION,
                        creation_time DOUBLE PRECISION,

                        -- Text compression
                        compression_type TEXT DEFAULT 'none',
                        compressed_data BYTEA,

                        -- Embedding compression
                        embedding_compression TEXT DEFAULT 'none',
                        compressed_embedding BYTEA
                    );
                """)

                # -------------------------------
                # Função + Trigger FTS
                # -------------------------------
                cur.execute("""
                    CREATE OR REPLACE FUNCTION memories_fts_trigger()
                    RETURNS trigger AS $$
                    BEGIN
                        NEW.fts_tokens :=
                            to_tsvector('simple', coalesce(NEW.content, ''));
                        RETURN NEW;
                    END
                    $$ LANGUAGE plpgsql;
                """)

                cur.execute("""
                    DROP TRIGGER IF EXISTS trg_memories_fts_update ON memories;
                """)

                cur.execute("""
                    CREATE TRIGGER trg_memories_fts_update
                    BEFORE INSERT OR UPDATE ON memories
                    FOR EACH ROW
                    EXECUTE FUNCTION memories_fts_trigger();
                """)

                # -------------------------------
                # Grafo de relações
                # -------------------------------
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS edges (
                        src TEXT REFERENCES memories(id) ON DELETE CASCADE,
                        dst TEXT REFERENCES memories(id) ON DELETE CASCADE,
                        type TEXT,
                        weight FLOAT DEFAULT 1.0,
                        last_update DOUBLE PRECISION
                            DEFAULT EXTRACT(EPOCH FROM NOW()),
                        PRIMARY KEY (src, dst, type)
                    );
                """)

                # -------------------------------
                # Índices
                # -------------------------------

                # HNSW (busca vetorial)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_mem_vec
                    ON memories
                    USING hnsw (embedding vector_cosine_ops)
                    WITH (m = 16, ef_construction = 64);
                """)

                # Full-Text Search
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_memories_fts
                    ON memories USING GIN (fts_tokens);
                """)

                # Tier management
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_mem_tier_access
                    ON memories (tier, last_access);
                """)

                # Graph traversal
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_edges_src_weight
                    ON edges (src, weight DESC);
                """)

                # Logs de acesso
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS access_logs (
                        mem_id TEXT,
                        access_time DOUBLE PRECISION
                    );
                """)

                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_access_logs_time
                    ON access_logs (access_time, mem_id);
                """)

                conn.commit()

    # =========================================================================
    # INSERT / UPSERT de memória
    # =========================================================================
    def upsert_memory(
        self,
        mem_id: str,
        content: str,
        embedding,
        metadata: dict | None = None,
        importance: float = 1.0,
        tier: int = 0
    ):
        now = time.time()

        query = """
            INSERT INTO memories (
                id,
                content,
                embedding,
                metadata,
                importance,
                tier,
                access_count,
                last_access,
                creation_time
            )
            VALUES (
                %(id)s,
                %(content)s,
                %(embedding)s,
                %(metadata)s,
                %(importance)s,
                %(tier)s,
                0,
                %(now)s,
                %(now)s
            )
            ON CONFLICT (id) DO UPDATE SET
                content = EXCLUDED.content,
                embedding = EXCLUDED.embedding,
                metadata = EXCLUDED.metadata,
                importance = EXCLUDED.importance,
                tier = EXCLUDED.tier,
                last_access = EXCLUDED.last_access;
        """

        params = {
            "id": mem_id,
            "content": content,
            "embedding": embedding,
            "metadata": Json(metadata) if metadata else None,
            "importance": importance,
            "tier": tier,
            "now": now
        }

        self.execute(query, params)

    # =========================================================================
    # Exec helpers
    # =========================================================================
    def execute(self, query, params=None):
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                conn.commit()

    def fetch_all(self, query, params=None):
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                return cur.fetchall()

    # =========================================================================
    # Sincroniza estatísticas de acesso (logs → tabela principal)
    # =========================================================================
    def sync_access_stats(self):
        """
        Agrega access_logs e atualiza:
        - access_count (incremental)
        - last_access (máximo entre atual e logs)
        """

        query = """
            WITH agg AS (
                SELECT
                    mem_id,
                    COUNT(*) AS cnt,
                    MAX(access_time) AS latest
                FROM access_logs
                GROUP BY mem_id
            )
            UPDATE memories m
            SET
                access_count = m.access_count + agg.cnt,
                last_access  = GREATEST(
                    COALESCE(m.last_access, 0),
                    agg.latest
                )
            FROM agg
            WHERE m.id = agg.mem_id;
        """

        self.execute(query)
