export interface Memory {
  id: string;
  content: string;
  importance: number;
  access_count: number; // Alterado de 'tier' para 'access_count'
  last_accessed?: number; // Opcional: útil para debug
  is_permanent: boolean;
}

export interface Message {
  role: 'user' | 'assistant';
  content: string;
}