import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime
import csv
import time
from typing import List, Tuple, Dict, Optional

# -------------------------
# Small tooltip helper
# -------------------------
class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip = None
        widget.bind("<Enter>", self.show)
        widget.bind("<Leave>", self.hide)

    def show(self, _=None):
        if self.tip or not self.text: return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + 20
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        label = tk.Label(self.tip, text=self.text, justify='left',
                         background="#ffffe0", foreground="#000", borderwidth=1, relief="solid",
                         font=("Segoe UI", 8))
        label.pack(ipadx=4, ipady=2)
        self.tip.geometry(f"+{x}+{y}")

    def hide(self, _=None):
        if self.tip:
            self.tip.destroy()
            self.tip = None

# -------------------------
# Algorithms encapsulation
# -------------------------
class PageReplacementAlgorithms:
    @staticmethod
    def fifo(pages: List[int], capacity: int) -> Tuple[int, int, List[List[Optional[int]]]]:
        memory: List[Optional[int]] = []
        page_faults = 0
        hits = 0
        frames_over_time = []

        for page in pages:
            if page not in memory:
                if len(memory) >= capacity:
                    memory.pop(0)
                memory.append(page)
                page_faults += 1
            else:
                hits += 1
            # record snapshot of frames (pad with None to fixed length)
            frames_over_time.append(list(memory) + [None] * max(0, capacity - len(memory)))

        return page_faults, hits, frames_over_time

    @staticmethod
    def lru(pages: List[int], capacity: int) -> Tuple[int, int, List[List[Optional[int]]]]:
        memory: List[Optional[int]] = []
        page_faults = 0
        hits = 0
        frames_over_time = []

        for page in pages:
            if page not in memory:
                if len(memory) >= capacity:
                    memory.pop(0)
                memory.append(page)
                page_faults += 1
            else:
                # move to end (most recently used)
                memory.remove(page)
                memory.append(page)
                hits += 1
            frames_over_time.append(list(memory) + [None] * max(0, capacity - len(memory)))

        return page_faults, hits, frames_over_time

    @staticmethod
    def optimal(pages: List[int], capacity: int) -> Tuple[int, int, List[List[Optional[int]]]]:
        memory: List[Optional[int]] = []
        page_faults = 0
        hits = 0
        frames_over_time = []

        n = len(pages)
        for i, page in enumerate(pages):
            if page in memory:
                hits += 1
            else:
                page_faults += 1
                if len(memory) < capacity:
                    memory.append(page)
                else:
                    farthest = -1
                    victim_idx = 0
                    for idx, mem_page in enumerate(memory):
                        try:
                            next_use = pages[i+1:].index(mem_page)
                        except ValueError:
                            next_use = float('inf')
                        if next_use > farthest:
                            farthest = next_use
                            victim_idx = idx
                    memory[victim_idx] = page
            frames_over_time.append(list(memory) + [None] * max(0, capacity - len(memory)))

        return page_faults, hits, frames_over_time

# -------------------------
# Main GUI
# -------------------------
class AdvancedPageSimulator:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Page Replacement Simulator")
        self.root.geometry("1200x820")
        self.root.minsize(980, 720)

        # state
        self.pages: List[int] = []
        self.capacity: int = 3
        self.results: Dict[str, Dict] = {}
        self.frames_history: Dict[str, List[List[Optional[int]]]] = {}
        self.current_step = 0
        self.max_steps = 0
        self.playing = False
        self.play_interval_ms = 700

        # UI pieces
        self._build_ui()
        self._bind_shortcuts()

    def _build_ui(self):
        style = ttk.Style()
        style.theme_use('clam')

        # TOP: configuration
        conf = ttk.LabelFrame(self.root, text="Configuration", padding=(10, 8))
        conf.pack(fill="x", padx=8, pady=8)

        ttk.Label(conf, text="Reference String (space separated):").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        self.ref_entry = ttk.Entry(conf, width=68)
        self.ref_entry.insert(0, "7 0 1 2 0 3 0 4 2 3 0 3 2 1 2 0 1 7 0 1")
        self.ref_entry.grid(row=0, column=1, columnspan=4, sticky="w", padx=6)

        ttk.Label(conf, text="Frames:").grid(row=1, column=0, sticky="w", padx=6, pady=6)
        self.frames_entry = ttk.Entry(conf, width=6)
        self.frames_entry.insert(0, "3")
        self.frames_entry.grid(row=1, column=1, sticky="w", padx=6)

        # algorithm checkboxes
        ttk.Label(conf, text="Algorithms:").grid(row=1, column=2, sticky="e")
        self.var_fifo = tk.BooleanVar(value=True)
        self.var_lru = tk.BooleanVar(value=True)
        self.var_opt = tk.BooleanVar(value=True)
        ttk.Checkbutton(conf, text="FIFO", variable=self.var_fifo).grid(row=1, column=3, sticky="w")
        ttk.Checkbutton(conf, text="LRU", variable=self.var_lru).grid(row=1, column=4, sticky="w")
        ttk.Checkbutton(conf, text="Optimal", variable=self.var_opt).grid(row=1, column=5, sticky="w")

        ToolTip(self.ref_entry, "Enter integers separated by spaces, e.g. '1 2 3 2 1'")

        # buttons: run / step controls / import/export
        btn_frame = ttk.Frame(conf)
        btn_frame.grid(row=2, column=0, columnspan=6, sticky="w", pady=(8, 2), padx=6)

        ttk.Button(btn_frame, text="Run →", command=self.run_all).pack(side="left", padx=4)
        ttk.Button(btn_frame, text="Run Step-by-Step", command=self.prepare_step_by_step).pack(side="left", padx=4)
        ttk.Button(btn_frame, text="Load CSV Testcases", command=self.load_testcases_csv).pack(side="left", padx=4)
        ttk.Button(btn_frame, text="Export Results (CSV)", command=self.export_results).pack(side="left", padx=4)
        ttk.Button(btn_frame, text="Save Chart as PNG", command=self.save_chart).pack(side="left", padx=4)

        # speed slider
        ttk.Label(btn_frame, text="Playback speed:").pack(side="left", padx=(20, 4))
        self.speed_scale = ttk.Scale(btn_frame, from_=100, to=2000, value=700, command=self._speed_changed)
        self.speed_scale.pack(side="left", padx=4, ipadx=30)
        ToolTip(self.speed_scale, "Lower = faster playback (ms between steps)")

        # middle: left = log & stats, right = timeline + frames
        mid = ttk.Frame(self.root)
        mid.pack(fill="both", expand=True, padx=8, pady=6)

        # Left: logs + metrics + chart
        left = ttk.Frame(mid)
        left.pack(side="left", fill="both", expand=True)

        # Logs
        log_card = ttk.LabelFrame(left, text="Console / Log", padding=(6, 6))
        log_card.pack(fill="x", padx=6, pady=(0, 6))
        self.log_area = tk.Text(log_card, height=10, state='disabled', bg="#f7f7f7")
        self.log_area.pack(fill="x")
        ToolTip(self.log_area, "Shows run-time messages and results")

        # Stats cards
        stats_card = ttk.LabelFrame(left, text="Summary Metrics", padding=(6, 6))
        stats_card.pack(fill="x", padx=6, pady=(0, 6))
        self.total_label = ttk.Label(stats_card, text="Total runs: 0")
        self.total_label.grid(row=0, column=0, sticky="w", padx=6)
        self.best_label = ttk.Label(stats_card, text="Best algorithm: N/A")
        self.best_label.grid(row=0, column=1, sticky="w", padx=6)
        self.avg_label = ttk.Label(stats_card, text="Avg Efficiency: N/A")
        self.avg_label.grid(row=0, column=2, sticky="w", padx=6)

        # Matplotlib chart
        chart_card = ttk.LabelFrame(left, text="Chart", padding=(6,6))
        chart_card.pack(fill="both", expand=True, padx=6, pady=(0,6))
        self.fig, self.ax = plt.subplots(figsize=(6,3))
        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_card)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Right: timeline table + frame visual + step controls
        right = ttk.Frame(mid, width=420)
        right.pack(side="right", fill="y", expand=False)

        # Timeline (Treeview)
        tv_card = ttk.LabelFrame(right, text="Timeline / Steps", padding=(6,6))
        tv_card.pack(fill="both", expand=True, padx=6, pady=(0,6))
        columns = ("step", "page", "fifo", "lru", "opt")
        self.timeline = ttk.Treeview(tv_card, columns=columns, show="headings", height=10)
        for c in columns:
            self.timeline.heading(c, text=c.capitalize())
            self.timeline.column(c, width=80, anchor="center")
        self.timeline.pack(fill="both", expand=True)

        # Frames visual (grid)
        frames_card = ttk.LabelFrame(right, text="Frames Visual (current step)", padding=(6,6))
        frames_card.pack(fill="x", padx=6, pady=(0,6))
        self.frame_canvas = tk.Canvas(frames_card, height=160)
        self.frame_canvas.pack(fill="x")

        # Step controls
        controls = ttk.Frame(right)
        controls.pack(fill="x", padx=6, pady=(6,6))
        ttk.Button(controls, text="⏮ Prev", command=self.step_prev).pack(side="left", padx=4)
        ttk.Button(controls, text="⏯ Play/Pause", command=self.toggle_play).pack(side="left", padx=4)
        ttk.Button(controls, text="⏭ Next", command=self.step_next).pack(side="left", padx=4)
        ttk.Button(controls, text="⏹ Stop", command=self.stop_playback).pack(side="left", padx=4)
        ttk.Button(controls, text="Reset", command=self.reset_simulation).pack(side="right", padx=4)

        self.status_bar = ttk.Label(self.root, text=f"Ready — {datetime.now().strftime('%H:%M:%S')}", anchor="w")
        self.status_bar.pack(side="bottom", fill="x")

    def _bind_shortcuts(self):
        self.root.bind("<space>", lambda e: self.toggle_play())
        self.root.bind("<Right>", lambda e: self.step_next())
        self.root.bind("<Left>", lambda e: self.step_prev())

    def _speed_changed(self, val):
        try:
            self.play_interval_ms = int(float(val))
        except Exception:
            pass

    # ---------------------------
    # Logging helpers
    # ---------------------------
    def log(self, message: str):
        self.log_area.config(state='normal')
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_area.insert(tk.END, f"[{ts}] {message}\n")
        self.log_area.see(tk.END)
        self.log_area.config(state='disabled')
        self.status_bar.config(text=message)

    # ---------------------------
    # Input parsing & validation
    # ---------------------------
    def _parse_inputs(self) -> bool:
        ref = self.ref_entry.get().strip()
        if not ref:
            messagebox.showerror("Input Error", "Reference string cannot be empty.")
            return False
        try:
            pages = list(map(int, ref.split()))
        except ValueError:
            messagebox.showerror("Input Error", "Reference string must contain integers separated by spaces.")
            return False

        try:
            cap = int(self.frames_entry.get())
            if cap <= 0:
                raise ValueError
        except Exception:
            messagebox.showerror("Input Error", "Frames must be a positive integer.")
            return False

        self.pages = pages
        self.capacity = cap
        return True

    # ---------------------------
    # Run & compute results
    # ---------------------------
    def run_all(self):
        if not self._parse_inputs():
            return
        selected = []
        if self.var_fifo.get(): selected.append(("FIFO", PageReplacementAlgorithms.fifo))
        if self.var_lru.get(): selected.append(("LRU", PageReplacementAlgorithms.lru))
        if self.var_opt.get(): selected.append(("Optimal", PageReplacementAlgorithms.optimal))
        if not selected:
            messagebox.showwarning("No Algorithm", "Select at least one algorithm to run.")
            return

        self.results.clear()
        self.frames_history.clear()
        start = time.time()
        for name, func in selected:
            t0 = time.time()
            faults, hits, frames = func(self.pages, self.capacity)
            t_exec = (time.time() - t0) * 1000.0
            eff = (hits / max(1, len(self.pages))) * 100.0
            self.results[name] = {"faults": faults, "hits": hits, "eff": eff, "time_ms": t_exec}
            self.frames_history[name] = frames
            self.log(f"{name}: Faults={faults} Hits={hits} Eff={eff:.2f}% (exec {t_exec:.2f} ms)")

        total_time = (time.time() - start) * 1000.0
        self.log(f"Run completed in {total_time:.2f} ms")
        self._update_summary()
        self._plot_chart()
        # build timeline table (merged)
        self._populate_timeline_table()
        # set step bounds
        self.max_steps = len(self.pages)
        self.current_step = 0
        self._draw_frames(step=0)

    def _update_summary(self):
        total = len(self.results)
        self.total_label.config(text=f"Total runs: {total}")
        if self.results:
            best = min(self.results.items(), key=lambda x: x[1]['faults'])[0]
            avg_eff = sum(r['eff'] for r in self.results.values()) / len(self.results)
            self.best_label.config(text=f"Best algorithm: {best}")
            self.avg_label.config(text=f"Avg Efficiency: {avg_eff:.2f}%")
        else:
            self.best_label.config(text="Best algorithm: N/A")
            self.avg_label.config(text="Avg Efficiency: N/A")

    def _plot_chart(self, per_step: Optional[int] = None):
        # if per_step provided, compute faults/hits up to that step
        algos = list(self.results.keys())
        if not algos:
            self.ax.clear()
            self.ax.text(0.5, 0.5, "No data", ha='center', va='center')
            self.canvas.draw()
            return

        if per_step is None:
            faults = [self.results[a]['faults'] for a in algos]
            hits = [self.results[a]['hits'] for a in algos]
        else:
            # compute for step
            faults, hits = [], []
            max_s = min(per_step + 1, len(self.pages))
            for a in algos:
                # count faults within frames_history by comparing snapshots: simpler approach:
                frames = self.frames_history[a][:max_s]
                # faults are number of changes in memory count where new page introduced: rough compute:
                # We'll compute hits as sum of pages that were present in previous snapshot
                # (approximation for per-step). For compactness use hits = total occurrences where page in snapshot earlier.
                # But to keep it consistent, compute hits by simulating early again:
                func = {"FIFO": PageReplacementAlgorithms.fifo,
                        "LRU": PageReplacementAlgorithms.lru,
                        "Optimal": PageReplacementAlgorithms.optimal}[a]
                f_up, h_up, _ = func(self.pages[:max_s], self.capacity)
                faults.append(f_up)
                hits.append(h_up)

        self.ax.clear()
        x = range(len(algos))
        width = 0.35
        self.ax.bar([i - width/2 for i in x], faults, width, label='Faults')
        self.ax.bar([i + width/2 for i in x], hits, width, label='Hits')
        self.ax.set_xticks(x)
        self.ax.set_xticklabels(algos)
        self.ax.set_ylabel("Count")
        self.ax.set_title("Hits vs Faults")
        self.ax.legend()

        # secondary axis: efficiency
        ratios = []
        for h, f in zip(hits, faults):
            denom = h + f
            ratios.append((h / denom) * 100 if denom > 0 else 0.0)
        ax2 = self.ax.twinx()
        ax2.plot(list(range(len(algos))), ratios, marker='o', linestyle='--')
        ax2.set_ylabel("Hit Ratio (%)")
        ax2.set_ylim(0, 100)

        # annotate ratios
        for i, v in enumerate(ratios):
            self.ax.text(i, max(faults[i], hits[i]) + 0.5, f"{v:.1f}%", ha='center')

        self.canvas.draw()

    def _populate_timeline_table(self):
        # clear
        for r in self.timeline.get_children():
            self.timeline.delete(r)
        # Populate up to the length of pages
        n = len(self.pages)
        for i in range(n):
            page = self.pages[i]
            row = [i+1, page]
            for algo in ("FIFO", "LRU", "Optimal"):
                if algo in self.frames_history:
                    frames = self.frames_history[algo]
                    if i < len(frames):
                        # show snapshot as comma-separated
                        snapshot = frames[i]
                        row.append(",".join(str(x) for x in snapshot if x is not None) or "-")
                    else:
                        row.append("-")
                else:
                    row.append("-")
            self.timeline.insert("", "end", values=row)

    # ---------------------------
    # Step-by-step / Animation
    # ---------------------------
    def prepare_step_by_step(self):
        if not self._parse_inputs():
            return
        # run algorithms but keep their frames_history
        self.var_fifo.set(True); self.var_lru.set(True); self.var_opt.set(True)
        self.run_all()
        if self.max_steps == 0:
            messagebox.showinfo("Nothing to animate", "Reference string is empty.")
            return
        self.current_step = 0
        self._draw_frames(0)
        self.log("Prepared step-by-step execution. Use Next/Prev/Play to animate.")

    def _draw_frames(self, step: int):
        self.frame_canvas.delete("all")
        # draw each algo as separate rows
        algos = [a for a in ("FIFO", "LRU", "Optimal") if a in self.frames_history]
        rows = len(algos)
        height = 120
        width = max(300, self.frame_canvas.winfo_width() - 10)
        self.frame_canvas.config(height=rows * 48 + 20)
        cell_w = max(60, (width - 20) // max(1, self.capacity))
        y = 10
        for r_idx, algo in enumerate(algos):
            frames = self.frames_history[algo]
            snapshot = frames[step] if step < len(frames) else [None]*self.capacity
            # label
            self.frame_canvas.create_text(10, y + 12, anchor="w", text=algo, font=("Segoe UI", 10, "bold"))
            # boxes
            x = 80
            for f_idx in range(self.capacity):
                val = snapshot[f_idx] if f_idx < len(snapshot) else None
                rect = self.frame_canvas.create_rectangle(x, y, x + cell_w - 6, y + 30, fill="#ffffff", outline="#000000")
                text = str(val) if val is not None else ""
                self.frame_canvas.create_text(x + (cell_w - 6)/2, y + 15, text=text)
                x += cell_w
            # highlight requested page for this step
            requested = self.pages[step] if step < len(self.pages) else None
            self.frame_canvas.create_text(width - 50, y + 12, anchor="w", text=f"Req: {requested}")
            y += 48
        # update chart to show aggregated results up to this step
        self._plot_chart(per_step=step)
        # update timeline selection
        items = self.timeline.get_children()
        if items:
            idx = min(step, len(items)-1)
            self.timeline.selection_set(items[idx])
            self.timeline.see(items[idx])

    def step_next(self):
        if self.max_steps == 0:
            return
        self.current_step = min(self.max_steps - 1, self.current_step + 1)
        self._draw_frames(self.current_step)

    def step_prev(self):
        if self.max_steps == 0:
            return
        self.current_step = max(0, self.current_step - 1)
        self._draw_frames(self.current_step)

    def toggle_play(self):
        if not self.frames_history:
            messagebox.showinfo("Nothing to play", "Run a simulation first (or prepare step-by-step).")
            return
        self.playing = not self.playing
        if self.playing:
            self._play_loop()
            self.log("Play started")
        else:
            self.log("Play paused")

    def _play_loop(self):
        if not self.playing:
            return
        self.step_next()
        if self.current_step >= self.max_steps - 1:
            self.playing = False
            self.log("Playback finished")
            return
        self.root.after(self.play_interval_ms, self._play_loop)

    def stop_playback(self):
        self.playing = False
        self.log("Playback stopped")

    def reset_simulation(self):
        self.pages = []
        self.frames_history.clear()
        self.results.clear()
        self.current_step = 0
        self.max_steps = 0
        for r in self.timeline.get_children():
            self.timeline.delete(r)
        self.frame_canvas.delete("all")
        self.ax.clear()
        self.canvas.draw()
        self.log("Simulation reset")

    # ---------------------------
    # Import / Export helpers
    # ---------------------------
    def load_testcases_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files","*.csv"), ("All files","*.*")])
        if not path:
            return
        try:
            with open(path, newline='') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            if not rows:
                messagebox.showinfo("No rows", "CSV contained no rows.")
                return
            # present first row but allow user to pick which row
            # For simplicity load first row
            r = rows[0]
            ref = r.get("reference_string") or r.get("reference") or r.get("ref") or ""
            cap = r.get("capacity") or r.get("frames") or r.get("cap") or "3"
            self.ref_entry.delete(0, tk.END); self.ref_entry.insert(0, ref)
            self.frames_entry.delete(0, tk.END); self.frames_entry.insert(0, cap)
            self.log(f"Loaded testcase from {path} (first row).")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV: {e}")

    def export_results(self):
        if not self.results:
            messagebox.showinfo("No results", "Run a simulation before exporting.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files","*.csv")])
        if not path:
            return
        try:
            with open(path, "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Algorithm","Faults","Hits","Efficiency (%)","Exec Time (ms)"])
                for algo, data in self.results.items():
                    writer.writerow([algo, data['faults'], data['hits'], f"{data['eff']:.2f}", f"{data['time_ms']:.3f}"])
            self.log(f"Exported results to {path}")
            messagebox.showinfo("Exported", "Results exported successfully.")
        except Exception as e:
            messagebox.showerror("Export failed", str(e))

    def save_chart(self):
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG image","*.png")])
        if not path:
            return
        try:
            self.fig.savefig(path, bbox_inches='tight', dpi=200)
            self.log(f"Saved chart to {path}")
            messagebox.showinfo("Saved", "Chart saved as PNG.")
        except Exception as e:
            messagebox.showerror("Save failed", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = AdvancedPageSimulator(root)
    root.mainloop()
