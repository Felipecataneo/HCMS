"use client";
import { useState, useEffect } from "react";
import { Message, Memory } from "@/types";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { MemoryDashboard } from "@/components/memory-dashboard";
import { Send, User, Bot } from "lucide-react";

const API_URL = "http://localhost:8000";

export default function HCMSApp() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [memories, setMemories] = useState<Memory[]>([]);
  const [loading, setLoading] = useState(false);

  // Busca memórias do backend
  const fetchMemories = async () => {
    const res = await fetch(`${API_URL}/memories`);
    const data = await res.json();
    setMemories(data);
  };

  useEffect(() => { fetchMemories(); }, []);

  const sendMessage = async () => {
    if (!input.trim() || loading) return;

    const userMsg: Message = { role: "user", content: input };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setLoading(true);

    try {
      const res = await fetch(`${API_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: input }),
      });
      const data = await res.json();
      setMessages((prev) => [...prev, { role: "assistant", content: data.reply }]);
      fetchMemories(); // Atualiza memórias após a resposta do agente
    } finally {
      setLoading(false);
    }
  };

    const deleteMemory = async (id: string) => {
    try {
        const res = await fetch(`${API_URL}/memories/${id}`, { 
        method: "DELETE" 
        });
        
        if (res.ok) {
        // Atualiza o estado local imediatamente removendo a memória deletada
        // Isso faz o refresh ser instantâneo na UI
        setMemories((prev) => prev.filter(m => m.id !== id));
        
        // Opcional: toast de confirmação
        console.log("Memory purged.");
        }
    } catch (error) {
        console.error("Failed to delete memory:", error);
    }
    };
    return (
    // h-screen + overflow-hidden trava a página na altura da janela
    <div className="flex h-screen bg-white overflow-hidden"> 
        
        {/* Chat Area - flex-1 garante que o chat ocupe o espaço restante */}
        <div className="flex-1 flex flex-col min-w-0 border-x overflow-hidden">
        <header className="p-4 border-b bg-white/80 backdrop-blur-md z-10">
            <h1 className="text-xl font-bold tracking-tight">HCMS <span className="text-indigo-600">v2</span></h1>
        </header>

        {/* ScrollArea do Chat */}
        <ScrollArea className="flex-1 p-4">
          <div className="space-y-6 max-w-3xl mx-auto">
            {messages.map((m, i) => (
              <div 
                key={i} 
                className={`flex gap-3 ${m.role === "user" ? "flex-row-reverse" : ""}`}
              >
                {/* Avatar */}
                <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
                  m.role === "user" ? "bg-slate-900" : "bg-indigo-600"
                }`}>
                  {m.role === "user" ? (
                    <User className="w-4 h-4 text-white" />
                  ) : (
                    <Bot className="w-4 h-4 text-white" />
                  )}
                </div>

                {/* Balão de Mensagem */}
                <div className={`max-w-[80%] p-3 rounded-2xl shadow-sm ${
                  m.role === "user" 
                    ? "bg-slate-100 text-slate-900 rounded-tr-none" 
                    : "bg-white border border-slate-200 text-slate-900 rounded-tl-none"
                }`}>
                  <p className="text-sm leading-relaxed whitespace-pre-wrap">
                    {m.content}
                  </p>
                </div>
              </div>
            ))}
            
            {loading && (
              <div className="flex gap-3">
                <div className="w-8 h-8 rounded-full bg-indigo-600 flex items-center justify-center animate-pulse">
                  <Bot className="w-4 h-4 text-white" />
                </div>
                <div className="bg-slate-50 border border-slate-200 p-3 rounded-2xl rounded-tl-none">
                  <div className="flex gap-1">
                    <span className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce [animation-delay:-0.3s]"></span>
                    <span className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce [animation-delay:-0.15s]"></span>
                    <span className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce"></span>
                  </div>
                </div>
              </div>
            )}
          </div>
        </ScrollArea>

        {/* Input fixo no fundo */}
        <div className="p-4 border-t bg-white">
            <form className="flex gap-2 max-w-3xl mx-auto" onSubmit={(e) => { e.preventDefault(); sendMessage(); }}>
            <Input 
                placeholder="Fale algo para eu lembrar..." 
                value={input}
                onChange={(e) => setInput(e.target.value)}
                className="flex-1"
            />
            <Button type="submit" size="icon" disabled={loading}>
                <Send className="w-4 h-4" />
            </Button>
            </form>
        </div>
        </div>

        {/* Memory Sidebar - h-full garante que ela ocupe a altura toda */}
        <MemoryDashboard 
        memories={memories} 
        onDelete={deleteMemory} 
        onRefresh={fetchMemories} 
        />
    </div>
    );
}