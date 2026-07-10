import { useMemo } from "react";

export function ActivityVolumeGrid() {
  const days = useMemo(() => {
    const arr = [];
    const startDate = new Date();
    startDate.setDate(startDate.getDate() - 83);

    for (let i = 0; i < 84; i++) {
      const d = new Date(startDate);
      d.setDate(startDate.getDate() + i);
      const dayOfWeek = d.getDay();
      const isWeekend = dayOfWeek === 0 || dayOfWeek === 6;
      const seed = (i * 7 + dayOfWeek * 13) % 100;
      let distance = 0;
      if (seed < 38) {
        distance = isWeekend ? (seed % 14) + 8 : (seed % 7) + 3;
      }
      arr.push({ date: d, distance });
    }
    return arr;
  }, []);

  const columns = useMemo(() => {
    const cols = [];
    for (let i = 0; i < 12; i++) {
      cols.push(days.slice(i * 7, (i + 1) * 7));
    }
    return cols;
  }, [days]);

  return (
    <div className="flex flex-col gap-2 mt-4 pt-3 border-t border-border/40">
      <div className="flex justify-between items-center">
        <span className="text-[10px] font-bold text-gray-500 dark:text-gray-400 tracking-wider">
          Weekly Consistency
        </span>
        <span className="text-[10px] text-muted font-medium">Last 12 weeks</span>
      </div>

      <div className="flex gap-1.5 justify-center py-1 overflow-x-auto">
        {columns.map((week, wIdx) => (
          <div key={wIdx} className="flex flex-col gap-1.5">
            {week.map((day, dIdx) => {
              const formattedDate = day.date.toLocaleDateString("en-US", {
                month: "short",
                day: "numeric",
              });

              let dotClass =
                "bg-zinc-200 dark:bg-zinc-800/80 hover:bg-slate-200 dark:hover:bg-slate-700";
              if (day.distance > 0) {
                if (day.distance < 5) {
                  dotClass = "bg-emerald-300 dark:bg-emerald-500/40 hover:brightness-110";
                } else if (day.distance < 10) {
                  dotClass = "bg-emerald-400 dark:bg-emerald-500/70 hover:brightness-110";
                } else {
                  dotClass =
                    "bg-emerald-500 hover:brightness-110 shadow-[0_0_6px_rgba(16,185,129,0.4)]";
                }
              }

              return (
                <div key={dIdx} className="group relative">
                  <div
                    className={`w-3.5 h-3.5 rounded-none transition-all duration-200 cursor-pointer ${dotClass}`}
                    aria-label={`${formattedDate}: ${
                      day.distance > 0 ? `${day.distance.toFixed(1)} km` : "No run"
                    }`}
                  />
                  <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 hidden group-hover:flex bg-slate-900/95 dark:bg-slate-950/95 text-white text-[10px] font-medium rounded-none px-2.5 py-1 whitespace-nowrap shadow-none border border-white/10 backdrop-blur-xs pointer-events-none z-50">
                    {formattedDate}:{" "}
                    <strong className="font-extrabold text-emerald-400 ml-1">
                      {day.distance > 0 ? `${day.distance.toFixed(1)} km` : "No run"}
                    </strong>
                  </div>
                </div>
              );
            })}
          </div>
        ))}
      </div>
    </div>
  );
}
