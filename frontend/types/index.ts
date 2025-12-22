export interface Memory {
  id: string;
  content: string;
  importance: number;
  tier: number;
}

export interface Message {
  role: 'user' | 'assistant';
  content: string;
}