import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime
import csv
import time
from typing import List, Tuple, Dict, Optional


# ==========================================
# HELPER CLASSES
# ==========================================

class ToolTip:
    """
    Displays a small pop-up box with text when hovering over a widget.
    Great for UX/Accessibility.
    """

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


# ==========================================
# ALGORITHM LOGIC (THE "MODEL")
# ==========================================

class PageReplacementAlgorithms:
    """
    Pure logic class.
    Returns: (Faults, Hits, Frame_Snapshots_Over_Time)
    """

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
            # Record snapshot (pad with None to maintain fixed size for visualization)
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
                # Move to end (mark as most recently used)
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

        for i, page in enumerate(pages):
            if page in memory:
                hits += 1
            else:
                page_faults += 1
                if len(memory) < capacity:
                    memory.append(page)
                else:
                    # Look ahead to find the page used farthest in the future
                    farthest = -1
                    victim_idx = 0
                    for idx, mem_page in enumerate(memory):
                        try:
                            next_use = pages[i + 1:].index(mem_page)
                        except ValueError:
                            next_use = float('inf')
                        if next_use > farthest:
                            farthest = next_use
                            victim_idx = idx
                    memory[victim_idx] = page
            frames_over_time.append(list(memory) + [None] * max(0, capacity - len(memory)))

        return page_faults, hits, frames_over_time


# ==========================================
# MAIN GUI APPLICATION (THE "VIEW" & "CONTROLLER")
# ==========================================

class AdvancedPageSimulator:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Page Replacement Simulator")
        self.root.geometry("1200x850")
        self.root.minsize(980, 720)

        # Application State
        self.pages: List[int] = []
        self.capacity: int = 3
        self.results: Dict[str, Dict] = {}
        self.frames_history: Dict[str, List[List[Optional[int]]]] = {}

        # Animation State
        self.current_step = 0
        self.max_steps = 0
        self.playing = False
        self.play_interval_ms = 700

        # Build Interface
        self._build_ui()
        self._bind_shortcuts()
        self.log("System Ready. Load a test case or enter data manually.")

    def _build_ui(self):
        style = ttk.Style()
        style.theme_use('clam')

        # --- 1. CONFIGURATION PANEL (Top) ---
        conf = ttk.LabelFrame(self.root, text="Configuration & Inputs", padding=(10, 8))
        conf.pack(fill="x", padx=10, pady=5)

        # Input Rows
        ttk.Label(conf, text="Reference String:").grid(row=0, column=0, sticky="w", padx=5)
        self.ref_entry = ttk.Entry(conf, width=70)
        self.ref_entry.insert(0, "7 0 1 2 0 3 0 4 2 3 0 3 2 1 2 0 1 7 0 1")
        self.ref_entry.grid(row=0, column=1, columnspan=4, sticky="w", padx=5)
        ToolTip(self.ref_entry, "Enter integers separated by spaces (e.g., '1 2 3 4')")

        ttk.Label(conf, text="Frames Capacity:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.frames_entry = ttk.Entry(conf, width=10)
        self.frames_entry.insert(0, "3")
        self.frames_entry.grid(row=1, column=1, sticky="w", padx=5)

        # Algorithm Selection
        ttk.Label(conf, text="Algorithms:").grid(row=1, column=2, sticky="e")
        self.var_fifo = tk.BooleanVar(value=True)
        self.var_lru = tk.BooleanVar(value=True)
        self.var_opt = tk.BooleanVar(value=True)
        ttk.Checkbutton(conf, text="FIFO", variable=self.var_fifo).grid(row=1, column=3, sticky="w")
        ttk.Checkbutton(conf, text="LRU", variable=self.var_lru).grid(row=1, column=4, sticky="w")
        ttk.Checkbutton(conf, text="Optimal", variable=self.var_opt).grid(row=1, column=5, sticky="w")

        # Control Buttons
        btn_frame = ttk.Frame(conf)
        btn_frame.grid(row=2, column=0, columnspan=6, sticky="w", pady=(10, 0))

        ttk.Button(btn_frame, text="▶ Run All", command=self.run_all).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="🎬 Initialize Animation", command=self.prepare_step_by_step).pack(side="left",
                                                                                                     padx=5)
        ttk.Separator(btn_frame, orient="vertical").pack(side="left", fill="y", padx=10)
        ttk.Button(btn_frame, text="📂 Load CSV", command=self.load_testcases_csv).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="💾 Export Results", command=self.export_results).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="📊 Save Chart", command=self.save_chart).pack(side="left", padx=5)

        # Animation Speed Slider
        ttk.Label(btn_frame, text="Speed:").pack(side="left", padx=(20, 5))
        self.speed_scale = ttk.Scale(btn_frame, from_=100, to=2000, value=700, command=self._speed_changed)
        self.speed_scale.pack(side="left", padx=5, ipadx=20)
        ToolTip(self.speed_scale, "Left = Fast, Right = Slow")

        # --- 2. MAIN CONTENT AREA (Middle) ---
        mid = ttk.Frame(self.root)
        mid.pack(fill="both", expand=True, padx=10, pady=5)

        # LEFT COLUMN (Logs, Metrics, Chart)
        left = ttk.Frame(mid)
        left.pack(side="left", fill="both", expand=True, padx=(0, 5))

        # Console Log
        log_card = ttk.LabelFrame(left, text="System Console", padding=(5, 5))
        log_card.pack(fill="x", pady=(0, 5))
        self.log_area = tk.Text(log_card, height=8, state='disabled', bg="#f0f0f0", font=("Consolas", 9))
        self.log_area.pack(fill="x")

        # Summary Metrics
        stats_card = ttk.LabelFrame(left, text="Performance Summary", padding=(5, 5))
        stats_card.pack(fill="x", pady=(0, 5))
        self.total_label = ttk.Label(stats_card, text="Total Runs: 0", font=("Segoe UI", 9, "bold"))
        self.total_label.pack(side="left", padx=10)
        self.best_label = ttk.Label(stats_card, text="Best Algo: N/A", font=("Segoe UI", 9, "bold"), foreground="green")
        self.best_label.pack(side="left", padx=10)
        self.avg_label = ttk.Label(stats_card, text="Avg Efficiency: N/A", font=("Segoe UI", 9, "bold"))
        self.avg_label.pack(side="left", padx=10)

        # Chart Area
        chart_card = ttk.LabelFrame(left, text="Analytics Visualization", padding=(5, 5))
        chart_card.pack(fill="both", expand=True)
        self.fig, self.ax = plt.subplots(figsize=(5, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_card)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # RIGHT COLUMN (Animation & Timeline)
        right = ttk.Frame(mid, width=450)
        right.pack(side="right", fill="y", expand=False, padx=(5, 0))

        # Timeline Table
        tv_card = ttk.LabelFrame(right, text="Execution Trace / Timeline", padding=(5, 5))
        tv_card.pack(fill="both", expand=True, pady=(0, 5))
        columns = ("step", "page", "fifo", "lru", "opt")
        self.timeline = ttk.Treeview(tv_card, columns=columns, show="headings", height=15)

        self.timeline.heading("step", text="#")
        self.timeline.column("step", width=40, anchor="center")
        self.timeline.heading("page", text="Page")
        self.timeline.column("page", width=50, anchor="center")
        for col in ["fifo", "lru", "opt"]:
            self.timeline.heading(col, text=col.upper())
            self.timeline.column(col, width=80, anchor="center")

        scrollbar = ttk.Scrollbar(tv_card, orient="vertical", command=self.timeline.yview)
        self.timeline.configure(yscroll=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.timeline.pack(fill="both", expand=True)

        # Memory Visualizer
        frames_card = ttk.LabelFrame(right, text="Memory Frames Visualizer", padding=(5, 5))
        frames_card.pack(fill="x", pady=(0, 5))
        self.frame_canvas = tk.Canvas(frames_card, height=180, bg="#ffffff")
        self.frame_canvas.pack(fill="x")

        # Playback Controls
        controls = ttk.LabelFrame(right, text="Playback Controls", padding=(5, 5))
        controls.pack(fill="x")

        c_inner = ttk.Frame(controls)
        c_inner.pack(anchor="center")
        ttk.Button(c_inner, text="⏮ Prev", command=self.step_prev).pack(side="left", padx=2)
        ttk.Button(c_inner, text="⏯ Play/Pause", command=self.toggle_play).pack(side="left", padx=2)
        ttk.Button(c_inner, text="⏭ Next", command=self.step_next).pack(side="left", padx=2)
        ttk.Button(c_inner, text="⏹ Stop", command=self.stop_playback).pack(side="left", padx=2)
        ttk.Button(c_inner, text="Reset System", command=self.reset_simulation).pack(side="left", padx=10)

        # Status Bar
        self.status_bar = ttk.Label(self.root, text=f"Ready", anchor="w", relief="sunken")
        self.status_bar.pack(side="bottom", fill="x")

    def _bind_shortcuts(self):
        self.root.bind("<space>", lambda e: self.toggle_play())
        self.root.bind("<Right>", lambda e: self.step_next())
        self.root.bind("<Left>", lambda e: self.step_prev())

    def _speed_changed(self, val):
        try:
            self.play_interval_ms = int(float(val))
        except:
            pass

    # --- LOGGING & INPUTS ---
    def log(self, message: str):
        self.log_area.config(state='normal')
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_area.insert(tk.END, f"[{ts}] {message}\n")
        self.log_area.see(tk.END)
        self.log_area.config(state='disabled')
        self.status_bar.config(text=f"Status: {message}")

    def _parse_inputs(self) -> bool:
        ref = self.ref_entry.get().strip()
        if not ref:
            messagebox.showerror("Error", "Reference string empty.")
            return False
        try:
            self.pages = list(map(int, ref.split()))
            self.capacity = int(self.frames_entry.get())
            if self.capacity <= 0: raise ValueError
        except ValueError:
            messagebox.showerror("Error", "Invalid inputs. Use integers.")
            return False
        return True

    # --- CORE LOGIC ---
    def run_all(self):
        if not self._parse_inputs(): return

        # Determine which algorithms to run
        selected = []
        if self.var_fifo.get(): selected.append(("FIFO", PageReplacementAlgorithms.fifo))
        if self.var_lru.get(): selected.append(("LRU", PageReplacementAlgorithms.lru))
        if self.var_opt.get(): selected.append(("Optimal", PageReplacementAlgorithms.optimal))

        if not selected:
            messagebox.showwarning("Warning", "No algorithm selected.")
            return

        self.results.clear()
        self.frames_history.clear()

        start_time = time.time()

        for name, func in selected:
            t0 = time.time()
            faults, hits, frames = func(self.pages, self.capacity)
            t_exec = (time.time() - t0) * 1000.0
            eff = (hits / max(1, len(self.pages))) * 100.0

            self.results[name] = {"faults": faults, "hits": hits, "eff": eff, "time": t_exec}
            self.frames_history[name] = frames

            self.log(f"{name} Result: {faults} Faults | {hits} Hits | {eff:.1f}% Efficiency")

        self.log(f"All simulations completed in {(time.time() - start_time) * 1000:.2f} ms")

        self._update_summary()
        self._plot_chart()
        self._populate_timeline_table()

        # Prepare for animation but don't start yet
        self.max_steps = len(self.pages)
        self.current_step = self.max_steps  # Jump to end for static view
        self._draw_frames(self.max_steps - 1)

    def _update_summary(self):
        total = len(self.results)
        self.total_label.config(text=f"Total Runs: {total}")
        if self.results:
            best = min(self.results.items(), key=lambda x: x[1]['faults'])[0]
            avg = sum(r['eff'] for r in self.results.values()) / total
            self.best_label.config(text=f"Best Algo: {best}")
            self.avg_label.config(text=f"Avg Efficiency: {avg:.1f}%")

    def _plot_chart(self, per_step=None):
        self.ax.clear()
        if not self.results:
            self.canvas.draw()
            return

        algos = list(self.results.keys())

        # If showing animation step, calculate transient results
        if per_step is not None:
            curr_faults, curr_hits = [], []
            limit = min(per_step + 1, len(self.pages))
            temp_pages = self.pages[:limit]

            for a in algos:
                func = {"FIFO": PageReplacementAlgorithms.fifo, "LRU": PageReplacementAlgorithms.lru,
                        "Optimal": PageReplacementAlgorithms.optimal}[a]
                f, h, _ = func(temp_pages, self.capacity)
                curr_faults.append(f)
                curr_hits.append(h)
            faults, hits = curr_faults, curr_hits
        else:
            faults = [self.results[a]['faults'] for a in algos]
            hits = [self.results[a]['hits'] for a in algos]

        # Draw Bar Chart
        x = range(len(algos))
        width = 0.35
        self.ax.bar([i - width / 2 for i in x], faults, width, label='Faults', color='#ff6b6b')
        self.ax.bar([i + width / 2 for i in x], hits, width, label='Hits', color='#4ecdc4')

        self.ax.set_xticks(x)
        self.ax.set_xticklabels(algos)
        self.ax.set_title("Hits vs Faults Comparison")
        self.ax.legend()
        self.ax.grid(True, axis='y', alpha=0.3)

        # Efficiency Line
        ax2 = self.ax.twinx()
        ratios = [(h / (h + f) * 100) if (h + f) > 0 else 0 for h, f in zip(hits, faults)]
        ax2.plot(x, ratios, marker='o', color='#2d3436', linestyle='--', linewidth=1)
        ax2.set_ylabel("Efficiency %")
        ax2.set_ylim(0, 110)

        self.canvas.draw()

    def _populate_timeline_table(self):
        for item in self.timeline.get_children():
            self.timeline.delete(item)

        for i, page in enumerate(self.pages):
            row = [i + 1, page]
            for algo in ["FIFO", "LRU", "Optimal"]:
                if algo in self.frames_history:
                    # Format snapshot as "[1, 2, 3]"
                    frames = self.frames_history[algo]
                    snap = frames[i] if i < len(frames) else []
                    snap_str = str([x for x in snap if x is not None])
                    row.append(snap_str)
                else:
                    row.append("-")
            self.timeline.insert("", "end", values=row)

    # --- ANIMATION ENGINE ---
    def prepare_step_by_step(self):
        if not self._parse_inputs(): return
        # Ensure we have data
        self.var_fifo.set(True);
        self.var_lru.set(True);
        self.var_opt.set(True)
        self.run_all()

        if self.max_steps == 0: return
        self.current_step = 0
        self._draw_frames(0)
        self.log("Animation Initialized. Press Play or use Step buttons.")

    def _draw_frames(self, step):
        self.frame_canvas.delete("all")

        algos = [a for a in ["FIFO", "LRU", "Optimal"] if a in self.frames_history]
        if not algos: return

        # Dimensions
        c_width = self.frame_canvas.winfo_width()
        box_size = 40
        spacing = 10
        start_y = 20

        # For each algorithm
        for i, algo in enumerate(algos):
            frames = self.frames_history[algo]
            if step < len(frames):
                snapshot = frames[step]

                # Draw Algo Name
                y = start_y + (i * 60)
                self.frame_canvas.create_text(10, y + 20, text=algo, anchor="w", font=("Segoe UI", 10, "bold"))

                # Draw Frame Boxes
                start_x = 80
                for f_idx, val in enumerate(snapshot):
                    x = start_x + (f_idx * (box_size + spacing))
                    # Check if this specific box holds the newly arrived page
                    is_new = (val == self.pages[step]) if step < len(self.pages) else False
                    color = "#dafce0" if is_new and val is not None else "white"

                    self.frame_canvas.create_rectangle(x, y, x + box_size, y + box_size, fill=color, outline="black")
                    if val is not None:
                        self.frame_canvas.create_text(x + box_size / 2, y + box_size / 2, text=str(val),
                                                      font=("Consolas", 10, "bold"))

        # Highlight Timeline Row
        children = self.timeline.get_children()
        if children and step < len(children):
            self.timeline.selection_set(children[step])
            self.timeline.see(children[step])

        # Update Chart dynamically
        self._plot_chart(per_step=step)

    def step_next(self):
        if self.current_step < self.max_steps - 1:
            self.current_step += 1
            self._draw_frames(self.current_step)

    def step_prev(self):
        if self.current_step > 0:
            self.current_step -= 1
            self._draw_frames(self.current_step)

    def toggle_play(self):
        self.playing = not self.playing
        if self.playing:
            self.log("Playback Started")
            self._play_loop()
        else:
            self.log("Playback Paused")

    def _play_loop(self):
        if not self.playing: return
        if self.current_step < self.max_steps - 1:
            self.step_next()
            self.root.after(self.play_interval_ms, self._play_loop)
        else:
            self.playing = False
            self.log("Playback Finished")

    def stop_playback(self):
        self.playing = False
        self.current_step = 0
        self._draw_frames(0)
        self.log("Playback Stopped")

    def reset_simulation(self):
        self.playing = False
        self.results.clear()
        self.frames_history.clear()
        self.ax.clear()
        self.canvas.draw()
        self.frame_canvas.delete("all")
        for item in self.timeline.get_children():
            self.timeline.delete(item)
        self.log("System Reset")

    # --- IO HELPERS ---
    def load_testcases_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not path: return
        try:
            with open(path, newline='') as f:
                reader = csv.reader(f)
                # Skip header if it exists, roughly checking first char
                rows = list(reader)
                if not rows: return

                # Heuristic: Try to find a row that looks like data
                # Assuming CSV format: ID, Description, Frames, RefString
                # We'll pick the last row usually or the first valid one
                data_row = rows[-1]  # Load the last entry

                # Attempt to parse frames and ref string from columns
                # We look for a long string (ref) and a short number (frames)
                frames_val = "3"
                ref_val = ""

                for col in data_row:
                    if len(col) < 5 and col.isdigit():
                        frames_val = col
                    if " " in col and len(col) > 5:
                        ref_val = col

                if ref_val:
                    self.ref_entry.delete(0, tk.END);
                    self.ref_entry.insert(0, ref_val)
                    self.frames_entry.delete(0, tk.END);
                    self.frames_entry.insert(0, frames_val)
                    self.log(f"Loaded test case from {path}")
                else:
                    messagebox.showerror("Error", "Could not identify reference string in CSV")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV: {e}")

    def export_results(self):
        if not self.results: return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
        if path:
            try:
                with open(path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["Algorithm", "Faults", "Hits", "Efficiency", "Time(ms)"])
                    for algo, data in self.results.items():
                        writer.writerow(
                            [algo, data['faults'], data['hits'], f"{data['eff']:.2f}%", f"{data['time']:.2f}"])
                self.log(f"Exported to {path}")
                messagebox.showinfo("Success", "Data exported successfully.")
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def save_chart(self):
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Image", "*.png")])
        if path:
            self.fig.savefig(path, dpi=300, bbox_inches='tight')
            self.log(f"Chart saved to {path}")
            messagebox.showinfo("Success", "Chart saved successfully.")


if __name__ == "__main__":
    root = tk.Tk()
    app = AdvancedPageSimulator(root)
    root.mainloop()