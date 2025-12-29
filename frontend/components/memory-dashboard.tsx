"use client";

import { useState, useMemo, useEffect } from "react";
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
  Flame,
  Clock,
  TrendingUp,
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
  /* =========================
     STATE
  ========================== */
  const [minImportance, setMinImportance] = useState<number[]>([0]);
  const [sortBy, setSortBy] = useState<
    "importance" | "recent" | "access"
  >("importance");

  /* =========================
     FILTER + SORT
  ========================== */
  const processedMemories = useMemo(() => {
    const min = minImportance[0] ?? 0;

    const filtered = memories.filter(
      (m) => m.is_permanent || m.importance >= min
    );

    filtered.sort((a, b) => {
      if (a.is_permanent && !b.is_permanent) return -1;
      if (!a.is_permanent && b.is_permanent) return 1;

      switch (sortBy) {
        case "importance":
          return b.importance - a.importance;
        case "access":
          return b.access_count - a.access_count;
        case "recent":
          return (b.last_accessed || 0) - (a.last_accessed || 0);
        default:
          return 0;
      }
    });

    return filtered;
  }, [memories, minImportance, sortBy]);

  /* =========================
     STATS
  ========================== */
  const stats = useMemo(() => {
    const total = memories.length;
    const permanent = memories.filter((m) => m.is_permanent).length;
    const avgImportance =
      memories.reduce((sum, m) => sum + m.importance, 0) / total || 0;
    const totalAccess = memories.reduce(
      (sum, m) => sum + m.access_count,
      0
    );

    return { total, permanent, avgImportance, totalAccess };
  }, [memories]);

  /* =========================
     TIME HANDLING
  ========================== */
  const [now, setNow] = useState<number | null>(null);

  useEffect(() => {
    const updateNow = () => setNow(Math.floor(Date.now() / 1000));
    updateNow();
    const id = setInterval(updateNow, 60_000);
    return () => clearInterval(id);
  }, []);

  const formatTimeAgo = (timestamp?: number) => {
    if (!timestamp) return "nunca";
    if (now === null) return "agora";
    const seconds = now - timestamp;
    if (seconds < 60) return "agora";
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m atrás`;
    if (seconds < 86400) return `${Math.floor(seconds / 3600)}h atrás`;
    return `${Math.floor(seconds / 86400)}d atrás`;
  };

  /* =========================
     RENDER
  ========================== */
  return (
    <aside className="flex flex-col h-full min-h-0 w-80 border-l bg-slate-50/50 p-4">
      {/* ================= HEADER ================= */}
      <div className="space-y-4 mb-6 flex-shrink-0">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Brain className="w-5 h-5 text-indigo-600" />
            <h2 className="font-semibold text-sm text-slate-900">
              Memory Engine
            </h2>
          </div>
          <Button
            variant="ghost"
            size="icon"
            className="h-8 w-8"
            onClick={onRefresh}
          >
            <RefreshCw className="w-4 h-4 text-slate-500" />
          </Button>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-2 gap-2">
          <div className="bg-white border rounded-lg p-2">
            <div className="text-[10px] uppercase font-bold text-slate-500">
              Total
            </div>
            <div className="text-lg font-bold">{stats.total}</div>
          </div>
          <div className="bg-indigo-50 border border-indigo-200 rounded-lg p-2">
            <div className="text-[10px] uppercase font-bold text-indigo-600">
              Locked
            </div>
            <div className="text-lg font-bold text-indigo-700">
              {stats.permanent}
            </div>
          </div>
        </div>

        <div className="bg-white border rounded-lg p-2 space-y-1">
          <div className="flex justify-between text-[10px]">
            <span className="text-slate-500">Avg Importance</span>
            <span className="font-bold">
              {stats.avgImportance.toFixed(2)}
            </span>
          </div>
          <div className="flex justify-between text-[10px]">
            <span className="text-slate-500">Total Access</span>
            <span className="font-bold">{stats.totalAccess}</span>
          </div>
        </div>
      </div>

      {/* ================= FILTERS ================= */}
      <div className="space-y-4 mb-4 flex-shrink-0">
        {/* Slider */}
        <div className="space-y-2 select-none">
          <div className="flex justify-between text-[10px] font-bold uppercase text-slate-500">
            <div className="flex items-center gap-1">
              <SlidersHorizontal className="w-3 h-3" />
              Min Importance
            </div>
            <span className="bg-slate-200 px-1.5 py-0.5 rounded">
              {minImportance[0].toFixed(2)}
            </span>
          </div>

          <Slider
            value={minImportance}
            min={0}
            max={1}
            step={0.05}
            onValueChange={(v) => setMinImportance(v)}
            className="py-2 touch-pan-x"
          />
        </div>

        {/* Sort */}
        <div className="flex gap-1">
          <Button
            size="sm"
            className="flex-1 h-7 text-[10px]"
            variant={sortBy === "importance" ? "default" : "outline"}
            onClick={() => setSortBy("importance")}
          >
            <Flame className="w-3 h-3 mr-1" /> IMP
          </Button>
          <Button
            size="sm"
            className="flex-1 h-7 text-[10px]"
            variant={sortBy === "recent" ? "default" : "outline"}
            onClick={() => setSortBy("recent")}
          >
            <Clock className="w-3 h-3 mr-1" /> REC
          </Button>
          <Button
            size="sm"
            className="flex-1 h-7 text-[10px]"
            variant={sortBy === "access" ? "default" : "outline"}
            onClick={() => setSortBy("access")}
          >
            <TrendingUp className="w-3 h-3 mr-1" /> ACC
          </Button>
        </div>
      </div>

      {/* ================= LIST ================= */}
      <ScrollArea className="flex-1 min-h-0 -mx-4 px-4">
        <div className="space-y-3 pb-10">
          {processedMemories.length === 0 ? (
            <div className="text-center py-10">
              <Brain className="w-10 h-10 mx-auto text-slate-300 mb-2" />
              <p className="text-xs text-slate-400 italic">
                No memories match this filter
              </p>
              <Button
                size="sm"
                variant="ghost"
                onClick={() => setMinImportance([0])}
              >
                Reset filter
              </Button>
            </div>
          ) : (
            processedMemories.map((m) => (
              <Card
                key={m.id}
                className={`border transition ${
                  m.is_permanent
                    ? "bg-indigo-50 border-indigo-200"
                    : "bg-white"
                }`}
              >
                <CardContent className="p-3 space-y-2">
                  {/* Header */}
                  <div className="flex justify-between items-start">
                    <div className="flex gap-1 flex-wrap">
                      <Badge className="text-[10px]">
                        <Flame className="w-2.5 h-2.5 mr-0.5" />
                        {m.importance.toFixed(2)}
                      </Badge>
                      <Badge className="text-[10px]">
                        <TrendingUp className="w-2.5 h-2.5 mr-0.5" />
                        {m.access_count}
                      </Badge>
                      {m.last_accessed && (
                        <Badge className="text-[10px]">
                          <Clock className="w-2.5 h-2.5 mr-0.5" />
                          {formatTimeAgo(m.last_accessed)}
                        </Badge>
                      )}
                    </div>

                    <div className="flex gap-1">
                      <Button
                        size="icon"
                        variant="ghost"
                        className="h-6 w-6"
                        onClick={() => onTogglePermanent(m.id)}
                      >
                        {m.is_permanent ? (
                          <Lock className="w-3 h-3" />
                        ) : (
                          <Unlock className="w-3 h-3" />
                        )}
                      </Button>
                      <Button
                        size="icon"
                        variant="ghost"
                        className="h-6 w-6 text-red-500"
                        onClick={() => onDelete(m.id)}
                      >
                        <Trash2 className="w-3 h-3" />
                      </Button>
                    </div>
                  </div>

                  <p className="text-sm text-slate-700 leading-snug">
                    {m.content}
                  </p>

                  <div className="h-1 bg-slate-100 rounded">
                    <div
                      className="h-full bg-indigo-500"
                      style={{ width: `${m.importance * 100}%` }}
                    />
                  </div>
                </CardContent>
              </Card>
            ))
          )}
        </div>
      </ScrollArea>

      {/* ================= FOOTER ================= */}
      <footer className="pt-4 border-t text-center text-[10px] text-slate-400 flex-shrink-0">
        HCMS v5 — RAG Memory Engine
      </footer>
    </aside>
  );
}
