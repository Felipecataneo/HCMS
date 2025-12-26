"use client";

import { useState } from "react";
import { Memory } from "@/types";

import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";

import {
  Brain,
  Trash2,
  RefreshCw,
  SlidersHorizontal,
  Lock,
  Unlock,
} from "lucide-react";

interface Props {
  memories: Memory[];
  onDelete: (id: string) => void;
  onRefresh: () => void;
  onTogglePermanent: (id: string) => void;
}

export function MemoryDashboard({
  memories,
  onDelete,
  onRefresh,
  onTogglePermanent,
}: Props) {
  const [minImportance, setMinImportance] = useState([0.0]);

  /**
   * REGRA v4.2:
   * - Memórias permanentes (crystallized) ignoram o filtro de importância
   */
  const filteredMemories = memories.filter(
    (m) => m.is_permanent || m.importance >= minImportance[0]
  );

  return (
    <div className="flex flex-col h-full border-l bg-slate-50/50 p-4 w-80 overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between mb-6 flex-shrink-0">
        <div className="flex items-center gap-2">
          <Brain className="w-5 h-5 text-indigo-600" />
          <h2 className="font-semibold text-slate-900 text-sm">
            Memory Engine
          </h2>
        </div>
        <Button
          variant="ghost"
          size="icon"
          className="h-8 w-8 hover:bg-slate-200"
          onClick={onRefresh}
        >
          <RefreshCw className="w-4 h-4 text-slate-500" />
        </Button>
      </div>

      {/* Filtro de Importância */}
      <div className="space-y-4 mb-6 px-1 flex-shrink-0">
        <div className="flex justify-between items-center text-[10px] font-bold text-slate-500 uppercase tracking-widest">
          <div className="flex items-center gap-1">
            <SlidersHorizontal className="w-3 h-3" />
            Min Importance
          </div>
          <span className="bg-slate-200 px-1.5 py-0.5 rounded">
            {minImportance[0].toFixed(1)}
          </span>
        </div>
        <Slider
          value={minImportance}
          max={1}
          step={0.1}
          onValueChange={setMinImportance}
          className="py-2"
        />
      </div>

      {/* Lista de Memórias */}
      <ScrollArea className="flex-1 -mx-4 px-4 h-full">
        <div className="space-y-3 pb-6">
          {filteredMemories.length === 0 ? (
            <div className="text-center py-10 px-4">
              <p className="text-xs text-slate-400 italic">
                No memories meet the current importance threshold.
              </p>
            </div>
          ) : (
            filteredMemories.map((m) => (
              <Card
                key={m.id}
                className={`group relative overflow-hidden transition-all border ${
                  m.is_permanent
                    ? "bg-indigo-50/40 border-indigo-200"
                    : "bg-white border-slate-200"
                } hover:shadow-sm`}
              >
                <CardContent className="p-3">
                  <div className="flex justify-between items-start mb-2">
                    {/* Badges */}
                    <div className="flex gap-2">
                      <Badge
                        variant="outline"
                        className={`text-[10px] font-bold ${
                          m.importance > 0.7
                            ? "border-red-100 bg-red-50 text-red-600"
                            : "border-slate-100 bg-slate-50 text-slate-500"
                        }`}
                      >
                        IMP: {m.importance.toFixed(1)}
                      </Badge>

                      <Badge
                        variant="outline"
                        className="text-[10px] border-indigo-100 bg-indigo-50 text-indigo-600"
                      >
                        ACC: {m.access_count}
                      </Badge>
                    </div>

                    {/* Ações */}
                    <div className="flex gap-1 opacity-0 group-hover:opacity-100 transition-all">
                      <Button
                        variant="ghost"
                        size="icon"
                        className={`h-6 w-6 ${
                          m.is_permanent
                            ? "text-indigo-600 hover:bg-indigo-100"
                            : "text-slate-400 hover:bg-slate-100"
                        }`}
                        onClick={() => onTogglePermanent(m.id)}
                      >
                        {m.is_permanent ? (
                          <Lock className="w-3 h-3" />
                        ) : (
                          <Unlock className="w-3 h-3" />
                        )}
                      </Button>

                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-6 w-6 hover:bg-red-50 hover:text-red-600"
                        onClick={() => onDelete(m.id)}
                      >
                        <Trash2 className="w-3 h-3" />
                      </Button>
                    </div>
                  </div>

                  <p className="text-[13px] text-slate-600 leading-snug font-medium">
                    {m.content}
                  </p>
                </CardContent>
              </Card>
            ))
          )}
        </div>
      </ScrollArea>

      {/* Footer */}
      <div className="mt-auto pt-4 border-t border-slate-200 flex-shrink-0">
        <p className="text-[10px] text-center text-slate-400 font-medium uppercase tracking-tighter">
          HCMS v4.2 — Crystallized Memory Engine
        </p>
      </div>
    </div>
  );
}
