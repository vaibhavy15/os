import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime
import csv
import time
from typing import List, Tuple, Dict, Optional

# ==========================================
# PROFESSIONAL THEME (Light Mode)
# ==========================================
THEME = {
    "bg_main": "#f3f4f6",  # Light Grey (Dashboard Background)
    "bg_panel": "#ffffff",  # White (Cards/Panels)
    "fg_text": "#1f2937",  # Dark Slate (Text)
    "accent": "#2563eb",  # Royal Blue (Primary Buttons)
    "accent_hover": "#1d4ed8",  # Darker Blue (Hover)
    "success": "#059669",  # Green (Success)
    "warning": "#d97706",  # Orange (Warning)
    "chart_grid": "#e5e7eb",  # Light Grid Lines
    "input_bg": "#f9fafb",  # Very Light Grey (Inputs)
    "box_new": "#dcfce7",  # Light Green (Animation Box - New)
    "box_old": "#ffffff"  # White (Animation Box - Old)
}


# ==========================================
# HELPER CLASSES
# ==========================================

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


# ==========================================
# ALGORITHM LOGIC
# ==========================================

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
# MAIN GUI APPLICATION
# ==========================================

class AdvancedPageSimulator:
    def __init__(self, root):
        self.root = root
        self.root.title("Page Replacement Simulator (v2.0)")
        self.root.geometry("1250x850")
        self.root.minsize(1000, 720)
        self.root.configure(bg=THEME["bg_main"])

        # State
        self.pages: List[int] = []
        self.capacity: int = 3
        self.results: Dict[str, Dict] = {}
        self.frames_history: Dict[str, List[List[Optional[int]]]] = {}
        self.current_step = 0
        self.max_steps = 0
        self.playing = False
        self.play_interval_ms = 700

        self._apply_styles()
        self._build_ui()
        self._bind_shortcuts()
        self.log("System Ready.")

    def _apply_styles(self):
        style = ttk.Style()
        style.theme_use('clam')

        # General Configuration
        style.configure(".", background=THEME["bg_main"], foreground=THEME["fg_text"], font=("Segoe UI", 10))
        style.configure("TLabel", background=THEME["bg_main"], foreground=THEME["fg_text"])

        # Panels (Labelframes)
        style.configure("TLabelframe", background=THEME["bg_panel"], bordercolor="#e5e7eb")
        style.configure("TLabelframe.Label", background=THEME["bg_panel"], foreground=THEME["accent"],
                        font=("Segoe UI", 11, "bold"))

        # Custom Panel Frame
        style.configure("Card.TFrame", background=THEME["bg_panel"])

        # Entries
        style.configure("TEntry", fieldbackground=THEME["input_bg"], bordercolor="#d1d5db")

        # Buttons (Modern Flat Blue)
        style.configure("TButton", background=THEME["accent"], foreground="white", borderwidth=0,
                        font=("Segoe UI", 10, "bold"))
        style.map("TButton", background=[('active', THEME["accent_hover"]), ('pressed', '#1e40af')])

        # Treeview (Table)
        style.configure("Treeview",
                        background="white",
                        foreground="black",
                        fieldbackground="white",
                        font=("Segoe UI", 9),
                        rowheight=25)
        style.configure("Treeview.Heading", background="#f3f4f6", foreground="#374151", font=("Segoe UI", 9, "bold"))
        style.map("Treeview", background=[('selected', THEME["accent"])], foreground=[('selected', 'white')])

    def _build_ui(self):
        # --- 1. HEADER & CONFIGURATION (Top) ---
        conf_container = ttk.Frame(self.root, style="Card.TFrame", padding=15)
        conf_container.pack(fill="x", padx=15, pady=15)

        # Header Label
        ttk.Label(conf_container, text="Configuration", font=("Segoe UI", 14, "bold"),
                  background=THEME["bg_panel"]).pack(anchor="w", pady=(0, 10))

        # Inputs Grid
        grid_frame = ttk.Frame(conf_container, style="Card.TFrame")
        grid_frame.pack(fill="x")

        ttk.Label(grid_frame, text="Reference String:", background=THEME["bg_panel"]).grid(row=0, column=0, sticky="w",
                                                                                           padx=(0, 10))
        self.ref_entry = ttk.Entry(grid_frame, width=80)
        self.ref_entry.insert(0, "7 0 1 2 0 3 0 4 2 3 0 3 2 1 2 0 1 7 0 1")
        self.ref_entry.grid(row=0, column=1, sticky="w")

        ttk.Label(grid_frame, text="Frames Capacity:", background=THEME["bg_panel"]).grid(row=0, column=2, sticky="w",
                                                                                          padx=(20, 10))
        self.frames_entry = ttk.Entry(grid_frame, width=10)
        self.frames_entry.insert(0, "3")
        self.frames_entry.grid(row=0, column=3, sticky="w")

        # Algorithm Selection
        algo_frame = ttk.Frame(conf_container, style="Card.TFrame")
        algo_frame.pack(fill="x", pady=(15, 0))

        ttk.Label(algo_frame, text="Algorithms:", background=THEME["bg_panel"], font=("Segoe UI", 10, "bold")).pack(
            side="left", padx=(0, 10))

        # Custom Checkbutton Style for Light Mode
        style = ttk.Style()
        style.configure("TCheckbutton", background=THEME["bg_panel"])

        self.var_fifo = tk.BooleanVar(value=True)
        self.var_lru = tk.BooleanVar(value=True)
        self.var_opt = tk.BooleanVar(value=True)

        ttk.Checkbutton(algo_frame, text="FIFO", variable=self.var_fifo).pack(side="left", padx=10)
        ttk.Checkbutton(algo_frame, text="LRU", variable=self.var_lru).pack(side="left", padx=10)
        ttk.Checkbutton(algo_frame, text="Optimal", variable=self.var_opt).pack(side="left", padx=10)

        # Action Buttons
        btn_frame = ttk.Frame(conf_container, style="Card.TFrame")
        btn_frame.pack(fill="x", pady=(15, 0))

        ttk.Button(btn_frame, text="RUN SIMULATION", command=self.run_all).pack(side="left", padx=(0, 5))
        ttk.Button(btn_frame, text="Animation Mode", command=self.prepare_step_by_step).pack(side="left", padx=5)
        ttk.Separator(btn_frame, orient="vertical").pack(side="left", fill="y", padx=15)
        ttk.Button(btn_frame, text="Load CSV", command=self.load_testcases_csv).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Export CSV", command=self.export_results).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Save Chart", command=self.save_chart).pack(side="left", padx=5)

        ttk.Label(btn_frame, text="Anim Speed:", background=THEME["bg_panel"]).pack(side="left", padx=(30, 5))
        self.speed_scale = ttk.Scale(btn_frame, from_=100, to=2000, value=700, command=self._speed_changed)
        self.speed_scale.pack(side="left", ipadx=20)

        # --- 2. MAIN CONTENT (Middle) ---
        mid = ttk.Frame(self.root)
        mid.pack(fill="both", expand=True, padx=15, pady=5)

        # LEFT SIDE (Metrics & Charts)
        left = ttk.Frame(mid)
        left.pack(side="left", fill="both", expand=True, padx=(0, 15))

        # Metrics Panel
        metrics_panel = ttk.Frame(left, style="Card.TFrame", padding=15)
        metrics_panel.pack(fill="x", pady=(0, 15))

        self.total_label = ttk.Label(metrics_panel, text="Total Runs: 0", background=THEME["bg_panel"],
                                     font=("Segoe UI", 12))
        self.total_label.pack(side="left", expand=True)

        self.best_label = ttk.Label(metrics_panel, text="Best Algo: -", background=THEME["bg_panel"],
                                    foreground=THEME["success"], font=("Segoe UI", 12, "bold"))
        self.best_label.pack(side="left", expand=True)

        self.avg_label = ttk.Label(metrics_panel, text="Avg Efficiency: -", background=THEME["bg_panel"],
                                   font=("Segoe UI", 12))
        self.avg_label.pack(side="left", expand=True)

        # Chart Panel
        chart_panel = ttk.Frame(left, style="Card.TFrame", padding=10)
        chart_panel.pack(fill="both", expand=True)

        ttk.Label(chart_panel, text="Performance Visualization", background=THEME["bg_panel"],
                  font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 5))

        # Matplotlib (Clean White Style)
        plt.style.use('default')  # Reset to clean white
        self.fig, self.ax = plt.subplots(figsize=(5, 4))
        self.fig.patch.set_facecolor('white')
        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_panel)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # RIGHT SIDE (Trace & Animation)
        right = ttk.Frame(mid, width=450)
        right.pack(side="right", fill="y", expand=False)

        # Console
        console_panel = ttk.Frame(right, style="Card.TFrame", padding=10)
        console_panel.pack(fill="x", pady=(0, 15))
        ttk.Label(console_panel, text="System Log", background=THEME["bg_panel"], font=("Segoe UI", 10, "bold")).pack(
            anchor="w")
        self.log_area = tk.Text(console_panel, height=5, state='disabled', bg="#f8f9fa", font=("Consolas", 9),
                                relief="flat")
        self.log_area.pack(fill="x", pady=(5, 0))

        # Timeline Table
        table_panel = ttk.Frame(right, style="Card.TFrame", padding=10)
        table_panel.pack(fill="both", expand=True, pady=(0, 15))
        ttk.Label(table_panel, text="Execution Timeline", background=THEME["bg_panel"],
                  font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 5))

        columns = ("step", "page", "fifo", "lru", "opt")
        self.timeline = ttk.Treeview(table_panel, columns=columns, show="headings", height=10)
        self.timeline.heading("step", text="#")
        self.timeline.column("step", width=30, anchor="center")
        self.timeline.heading("page", text="Page")
        self.timeline.column("page", width=40, anchor="center")
        for col in ["fifo", "lru", "opt"]:
            self.timeline.heading(col, text=col)
            self.timeline.column(col, width=80, anchor="center")
        self.timeline.pack(fill="both", expand=True)

        # Visualizer
        vis_panel = ttk.Frame(right, style="Card.TFrame", padding=10)
        vis_panel.pack(fill="x")
        ttk.Label(vis_panel, text="Memory State Visualizer", background=THEME["bg_panel"],
                  font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 5))

        self.frame_canvas = tk.Canvas(vis_panel, height=180, bg="white", highlightthickness=0)
        self.frame_canvas.pack(fill="x")

        # Playback Controls
        ctrl_frame = ttk.Frame(vis_panel, style="Card.TFrame")
        ctrl_frame.pack(fill="x", pady=(10, 0))

        c_inner = ttk.Frame(ctrl_frame, style="Card.TFrame")
        c_inner.pack(anchor="center")
        ttk.Button(c_inner, text="⏮", command=self.step_prev, width=3).pack(side="left", padx=2)
        ttk.Button(c_inner, text="⏯ Play/Pause", command=self.toggle_play).pack(side="left", padx=2)
        ttk.Button(c_inner, text="⏭", command=self.step_next, width=3).pack(side="left", padx=2)
        ttk.Button(c_inner, text="Reset", command=self.reset_simulation).pack(side="left", padx=10)

        # Status Bar
        self.status_bar = tk.Label(self.root, text=" Ready", bg="#e5e7eb", fg="#374151", anchor="w",
                                   font=("Segoe UI", 9))
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

    # --- LOGIC ---

    def log(self, message: str):
        self.log_area.config(state='normal')
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_area.insert(tk.END, f"[{ts}] {message}\n")
        self.log_area.see(tk.END)
        self.log_area.config(state='disabled')
        self.status_bar.config(text=f" Status: {message}")

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
            messagebox.showerror("Error", "Invalid inputs.")
            return False
        return True

    def run_all(self):
        if not self._parse_inputs(): return

        selected = []
        if self.var_fifo.get(): selected.append(("FIFO", PageReplacementAlgorithms.fifo))
        if self.var_lru.get(): selected.append(("LRU", PageReplacementAlgorithms.lru))
        if self.var_opt.get(): selected.append(("Optimal", PageReplacementAlgorithms.optimal))

        if not selected:
            messagebox.showwarning("Warning", "Select at least one algorithm.")
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
            self.log(f"{name}: {faults} Faults, {hits} Hits")

        self._update_summary()
        self._plot_chart()
        self._populate_timeline_table()

        self.max_steps = len(self.pages)
        self.current_step = self.max_steps
        self._draw_frames(self.max_steps - 1)

    def _update_summary(self):
        total = len(self.results)
        self.total_label.config(text=f"Total Runs: {total}")
        if self.results:
            best = min(self.results.items(), key=lambda x: x[1]['faults'])[0]
            avg = sum(r['eff'] for r in self.results.values()) / total
            self.best_label.config(text=f"Best: {best}")
            self.avg_label.config(text=f"Avg Eff: {avg:.1f}%")

    def _plot_chart(self, per_step=None):
        self.ax.clear()
        if not self.results:
            self.canvas.draw()
            return

        algos = list(self.results.keys())

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

        x = range(len(algos))
        width = 0.35
        # Professional Light Theme Colors (Red/Green muted)
        self.ax.bar([i - width / 2 for i in x], faults, width, label='Faults', color='#ef4444')
        self.ax.bar([i + width / 2 for i in x], hits, width, label='Hits', color='#10b981')

        self.ax.set_xticks(x)
        self.ax.set_xticklabels(algos)
        self.ax.set_title("Hits vs Faults Analysis", fontsize=10, fontweight='bold', color=THEME['fg_text'])
        self.ax.legend(facecolor='white', framealpha=1)
        self.ax.grid(True, axis='y', color=THEME["chart_grid"], alpha=0.7)

        # Remove top/right spines
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['bottom'].set_color(THEME["chart_grid"])
        self.ax.spines['left'].set_color(THEME["chart_grid"])

        # Efficiency Line
        ax2 = self.ax.twinx()
        ratios = [(h / (h + f) * 100) if (h + f) > 0 else 0 for h, f in zip(hits, faults)]
        ax2.plot(x, ratios, marker='o', color='#374151', linestyle='--', linewidth=1.5)
        ax2.set_ylabel("Efficiency %")
        ax2.set_ylim(0, 110)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

        self.canvas.draw()

    def _populate_timeline_table(self):
        for item in self.timeline.get_children():
            self.timeline.delete(item)
        for i, page in enumerate(self.pages):
            row = [i + 1, page]
            for algo in ["FIFO", "LRU", "Optimal"]:
                if algo in self.frames_history:
                    frames = self.frames_history[algo]
                    snap = frames[i] if i < len(frames) else []
                    snap_str = str([x for x in snap if x is not None])
                    row.append(snap_str)
                else:
                    row.append("-")
            self.timeline.insert("", "end", values=row)

    # --- ANIMATION ---
    def prepare_step_by_step(self):
        if not self._parse_inputs(): return
        self.var_fifo.set(True);
        self.var_lru.set(True);
        self.var_opt.set(True)
        self.run_all()
        if self.max_steps == 0: return
        self.current_step = 0
        self._draw_frames(0)
        self.log("Animation Ready.")

    def _draw_frames(self, step):
        self.frame_canvas.delete("all")
        algos = [a for a in ["FIFO", "LRU", "Optimal"] if a in self.frames_history]
        if not algos: return

        box_size = 35
        spacing = 8
        start_y = 15

        for i, algo in enumerate(algos):
            frames = self.frames_history[algo]
            if step < len(frames):
                snapshot = frames[step]
                y = start_y + (i * 55)

                # Algo Label
                self.frame_canvas.create_text(10, y + box_size / 2, text=algo, anchor="w", fill=THEME["fg_text"],
                                              font=("Segoe UI", 9, "bold"))

                # Draw Frame Boxes
                start_x = 70
                for f_idx, val in enumerate(snapshot):
                    x = start_x + (f_idx * (box_size + spacing))
                    is_new = (val == self.pages[step]) if step < len(self.pages) else False

                    # Clean Light Colors
                    fill_col = THEME["box_new"] if is_new and val is not None else "white"
                    outline_col = THEME["accent"] if is_new else "#9ca3af"

                    self.frame_canvas.create_rectangle(x, y, x + box_size, y + box_size, fill=fill_col,
                                                       outline=outline_col)
                    if val is not None:
                        self.frame_canvas.create_text(x + box_size / 2, y + box_size / 2, text=str(val), fill="black",
                                                      font=("Consolas", 10, "bold"))

        children = self.timeline.get_children()
        if children and step < len(children):
            self.timeline.selection_set(children[step])
            self.timeline.see(children[step])
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
        if self.playing: self._play_loop()

    def _play_loop(self):
        if not self.playing: return
        if self.current_step < self.max_steps - 1:
            self.step_next()
            self.root.after(self.play_interval_ms, self._play_loop)
        else:
            self.playing = False

    def reset_simulation(self):
        self.playing = False
        self.results.clear()
        self.frames_history.clear()
        self.ax.clear()
        self.canvas.draw()
        self.frame_canvas.delete("all")
        for item in self.timeline.get_children():
            self.timeline.delete(item)
        self.log("Reset Complete.")

    def load_testcases_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not path: return
        try:
            with open(path, newline='') as f:
                reader = csv.reader(f)
                rows = list(reader)
                if rows:
                    data = rows[-1]
                    ref_val = next((c for c in data if " " in c and len(c) > 5), "")
                    if ref_val:
                        self.ref_entry.delete(0, tk.END);
                        self.ref_entry.insert(0, ref_val)
                        self.log("CSV Loaded.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def export_results(self):
        if not self.results: return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if path:
            with open(path, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(["Algo", "Faults", "Hits"])
                for k, v in self.results.items(): w.writerow([k, v['faults'], v['hits']])
            self.log("Exported.")

    def save_chart(self):
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
        if path: self.fig.savefig(path, dpi=300, facecolor='white')


if __name__ == "__main__":
    root = tk.Tk()
    app = AdvancedPageSimulator(root)
    root.mainloop()