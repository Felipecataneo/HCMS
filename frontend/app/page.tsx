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
    try {
      const res = await fetch(`${API_URL}/memories`);
      if (!res.ok) throw new Error(`Erro na busca: ${res.status}`);
      const data = await res.json();
      console.log("Memórias recebidas:", data); // Verifique se o array chega aqui
      setMemories(data);
    } catch (err) {
      console.error("Falha ao carregar memórias. O servidor está rodando?", err);
    }
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
      
      if (!res.ok) {
         const errorData = await res.json();
         throw new Error(errorData.detail || "Erro no servidor");
      }

      const data = await res.json();
      setMessages((prev) => [...prev, { role: "assistant", content: data.reply }]);
      
      // Delay de 500ms para dar tempo ao background task do backend salvar a memória
      setTimeout(() => {
        fetchMemories();
      }, 500);

    } catch (err) {
      console.error("Erro no chat:", err);
      setMessages((prev) => [...prev, { role: "assistant", content: "Erro: Não consegui conectar ao cérebro (Ollama/Backend)." }]);
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

    const togglePermanent = async (id: string) => {
      try {
        const res = await fetch(`${API_URL}/memories/${id}/toggle-permanent`, {
          method: "POST",
        });
        if (res.ok) {
          // Atualiza a lista localmente para refletir a mudança imediatamente
          setMemories((prev) =>
            prev.map((m) =>
              m.id === id ? { ...m, is_permanent: !m.is_permanent } : m
            )
          );
        }
      } catch (error) {
        console.error("Failed to toggle permanence:", error);
      }
    };


    return (
      <div className="flex h-screen bg-white overflow-hidden"> 
        
        {/* Chat Area */}
        <div className="flex-1 flex flex-col min-w-0 border-x h-full"> {/* Adicionado h-full aqui */}
          <header className="p-4 border-b bg-white/80 backdrop-blur-md z-10 flex-shrink-0">
            <h1 className="text-xl font-bold tracking-tight">HCMS <span className="text-indigo-600">v3</span></h1>
          </header>

          {/* Container do ScrollArea - O pulo do gato é o flex-1 + h-full + min-h-0 */}
          <div className="flex-1 overflow-hidden h-full min-h-0 relative">
            <ScrollArea className="h-full w-full p-4">
              <div className="space-y-6 max-w-3xl mx-auto pb-10"> {/* pb-10 evita que a última msg cole no input */}
                {messages.map((m, i) => (
                  <div 
                    key={i} 
                    className={`flex gap-3 ${m.role === "user" ? "flex-row-reverse" : ""}`}
                  >
                    {/* Avatar */}
                    <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
                      m.role === "user" ? "bg-slate-900" : "bg-indigo-600"
                    }`}>
                      {m.role === "user" ? <User className="w-4 h-4 text-white" /> : <Bot className="w-4 h-4 text-white" />}
                    </div>

                    {/* Balão */}
                    <div className={`max-w-[80%] p-3 rounded-2xl shadow-sm ${
                      m.role === "user" 
                        ? "bg-slate-100 text-slate-900 rounded-tr-none" 
                        : "bg-white border border-slate-200 text-slate-900 rounded-tl-none"
                    }`}>
                      <p className="text-sm leading-relaxed whitespace-pre-wrap">{m.content}</p>
                    </div>
                  </div>
                ))}
                
                {loading && (
                  <div className="flex gap-3">
                    <div className="w-8 h-8 rounded-full bg-indigo-600 flex items-center justify-center animate-pulse">
                      <Bot className="w-4 h-4 text-white" />
                    </div>
                    <div className="bg-slate-50 border border-slate-200 p-3 rounded-2xl rounded-tl-none">
                      <div className="flex gap-1 italic text-xs text-slate-400">Agente processando...</div>
                    </div>
                  </div>
                )}
              </div>
            </ScrollArea>
          </div>

          {/* Input fixo no fundo - flex-shrink-0 impede que o input seja esmagado */}
          <div className="p-4 border-t bg-white flex-shrink-0">
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

        {/* Memory Sidebar */}
        <MemoryDashboard 
          memories={memories} 
          onDelete={deleteMemory} 
          onRefresh={fetchMemories} 
          onTogglePermanent={togglePermanent}
        />
      </div>
    );
}