import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.style as mpl_style

# --- COLOR PALETTE (Modern Dark Theme) ---
COLORS = {
    "bg": "#222831",  # Darkest Grey (Background)
    "panel": "#393E46",  # Lighter Grey (Panels)
    "accent": "#00ADB5",  # Teal (Buttons/Highlights)
    "text": "#EEEEEE",  # White/Light Grey (Text)
    "danger": "#FF2E63",  # Red (Reset/Errors)
    "success": "#00ADB5"  # Greenish/Teal (Success)
}


class ModernPageSimulator:
    def __init__(self, root):
        self.root = root
        self.root.title("PageSwap Pro | Algorithm Simulator")
        self.root.geometry("1100x750")
        self.root.configure(bg=COLORS["bg"])

        # Initialize Logic Variables
        self.fifo_res = (0, 0)
        self.lru_res = (0, 0)
        self.opt_res = (0, 0)

        self.setup_styles()
        self.build_layout()

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')

        # Frame Styles
        style.configure("TFrame", background=COLORS["bg"])
        style.configure("Panel.TFrame", background=COLORS["panel"], relief="flat")

        # Label Styles
        style.configure("TLabel", background=COLORS["bg"], foreground=COLORS["text"], font=("Segoe UI", 10))
        style.configure("Header.TLabel", background=COLORS["panel"], foreground=COLORS["accent"],
                        font=("Segoe UI", 14, "bold"))
        style.configure("SubHeader.TLabel", background=COLORS["panel"], foreground=COLORS["text"],
                        font=("Segoe UI", 11))

        # Button Styles
        style.configure("Accent.TButton",
                        background=COLORS["accent"],
                        foreground="white",
                        font=("Segoe UI", 10, "bold"),
                        borderwidth=0, focuscolor="none")
        style.map("Accent.TButton", background=[('active', '#007A80')])  # Darker teal on hover

        style.configure("Danger.TButton",
                        background=COLORS["panel"],
                        foreground=COLORS["danger"],
                        font=("Segoe UI", 10),
                        borderwidth=1, focuscolor="none")
        style.map("Danger.TButton", background=[('active', '#4a4a4a')])

    def build_layout(self):
        # --- 1. Sidebar (Control Plane) ---
        sidebar = ttk.Frame(self.root, style="Panel.TFrame", width=300, padding=20)
        sidebar.pack(side="left", fill="y")

        # Title
        ttk.Label(sidebar, text="CONFIGURATION", style="Header.TLabel").pack(anchor="w", pady=(0, 20))

        # Inputs
        ttk.Label(sidebar, text="Reference String:", style="SubHeader.TLabel").pack(anchor="w", pady=(10, 5))
        self.ref_entry = tk.Entry(sidebar, bg="#222831", fg="white", insertbackground="white", relief="flat",
                                  font=("Consolas", 10))
        self.ref_entry.pack(fill="x", ipady=5)
        self.ref_entry.insert(0, "7 0 1 2 0 3 0 4 2 3 0 3 2 1 2 0 1 7 0 1")

        ttk.Label(sidebar, text="Frame Capacity:", style="SubHeader.TLabel").pack(anchor="w", pady=(20, 5))
        self.cap_entry = tk.Entry(sidebar, bg="#222831", fg="white", insertbackground="white", relief="flat",
                                  font=("Consolas", 10))
        self.cap_entry.pack(fill="x", ipady=5)
        self.cap_entry.insert(0, "3")

        # Actions
        ttk.Button(sidebar, text="RUN SIMULATION", style="Accent.TButton", command=self.run_simulation).pack(fill="x",
                                                                                                             pady=(30,
                                                                                                                   10),
                                                                                                             ipady=5)
        ttk.Button(sidebar, text="RESET DATA", style="Danger.TButton", command=self.reset_data).pack(fill="x", pady=5)

        # Log Area (Mini console)
        ttk.Label(sidebar, text="System Logs:", style="SubHeader.TLabel").pack(anchor="w", pady=(30, 5))
        self.log_area = tk.Text(sidebar, height=15, bg="#222831", fg="#00ADB5", font=("Consolas", 9), relief="flat",
                                state="disabled")
        self.log_area.pack(fill="both", expand=True)

        # --- 2. Main Dashboard (Data Plane) ---
        main_area = ttk.Frame(self.root, style="TFrame")
        main_area.pack(side="right", fill="both", expand=True, padx=20, pady=20)

        # Metric Cards Area
        self.cards_frame = ttk.Frame(main_area, style="TFrame")
        self.cards_frame.pack(fill="x", pady=(0, 20))

        # Placeholders for cards (Will be filled on run)
        self.card_fifo = self.create_metric_card(self.cards_frame, "FIFO", "Waiting...")
        self.card_lru = self.create_metric_card(self.cards_frame, "LRU", "Waiting...")
        self.card_opt = self.create_metric_card(self.cards_frame, "OPTIMAL", "Waiting...")

        # Visualization Area
        self.viz_frame = ttk.Frame(main_area, style="Panel.TFrame")
        self.viz_frame.pack(fill="both", expand=True)

        # Initial empty plot
        self.setup_empty_plot()

    def create_metric_card(self, parent, title, value):
        frame = tk.Frame(parent, bg=COLORS["panel"], padx=15, pady=15)
        frame.pack(side="left", fill="x", expand=True, padx=5)

        tk.Label(frame, text=title, bg=COLORS["panel"], fg="#aaaaaa", font=("Segoe UI", 9, "bold")).pack(anchor="w")
        val_label = tk.Label(frame, text=value, bg=COLORS["panel"], fg=COLORS["accent"], font=("Segoe UI", 18, "bold"))
        val_label.pack(anchor="w")
        return val_label

    def log(self, msg):
        self.log_area.config(state="normal")
        self.log_area.insert(tk.END, f"> {msg}\n")
        self.log_area.see(tk.END)
        self.log_area.config(state="disabled")

    def setup_empty_plot(self):
        # Dark mode for Matplotlib
        plt.style.use('dark_background')
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.fig.patch.set_facecolor(COLORS["panel"])
        self.ax.set_facecolor(COLORS["panel"])
        self.ax.text(0.5, 0.5, "Ready to Simulate", ha='center', color='gray')
        self.ax.axis('off')

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.viz_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def reset_data(self):
        self.ref_entry.delete(0, tk.END)
        self.cap_entry.delete(0, tk.END)
        self.log("Inputs cleared.")

    # --- ALGORITHMS (Model) ---
    def run_algo(self, name, func, pages, capacity):
        faults, hits = func(pages, capacity)
        ratio = (hits / len(pages)) * 100
        self.log(f"{name}: {faults} Faults | {hits} Hits | {ratio:.1f}% Eff.")
        return faults, hits, ratio

    def fifo_logic(self, pages, capacity):
        memory = []
        faults = 0
        hits = 0
        for page in pages:
            if page not in memory:
                if len(memory) >= capacity: memory.pop(0)
                memory.append(page)
                faults += 1
            else:
                hits += 1
        return faults, hits

    def lru_logic(self, pages, capacity):
        memory = []
        faults = 0
        hits = 0
        for page in pages:
            if page not in memory:
                if len(memory) >= capacity: memory.pop(0)
                memory.append(page)
                faults += 1
            else:
                memory.remove(page)
                memory.append(page)
                hits += 1
        return faults, hits

    def opt_logic(self, pages, capacity):
        memory = []
        faults = 0
        hits = 0
        for i, page in enumerate(pages):
            if page not in memory:
                if len(memory) < capacity:
                    memory.append(page)
                else:
                    farthest = -1
                    victim = -1
                    for m_idx, m_page in enumerate(memory):
                        try:
                            next_use = pages[i + 1:].index(m_page)
                        except:
                            next_use = float('inf')
                        if next_use > farthest:
                            farthest = next_use
                            victim = m_idx
                    memory[victim] = page
                faults += 1
            else:
                hits += 1
        return faults, hits

    # --- CONTROLLER ---
    def run_simulation(self):
        try:
            ref_str = list(map(int, self.ref_entry.get().split()))
            cap = int(self.cap_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid Input. Use integers.")
            return

        self.log("-" * 20)
        self.log(f"Simulating {len(ref_str)} pages in {cap} frames...")

        # Run Logic
        f_res = self.run_algo("FIFO", self.fifo_logic, ref_str, cap)
        l_res = self.run_algo("LRU", self.lru_logic, ref_str, cap)
        o_res = self.run_algo("OPT", self.opt_logic, ref_str, cap)

        # Update Cards
        self.card_fifo.config(text=f"{f_res[0]} Faults ({f_res[2]:.0f}%)")
        self.card_lru.config(text=f"{l_res[0]} Faults ({l_res[2]:.0f}%)")
        self.card_opt.config(text=f"{o_res[0]} Faults ({o_res[2]:.0f}%)")

        # Update Graph
        self.update_graph(f_res, l_res, o_res)

    def update_graph(self, f, l, o):
        self.ax.clear()
        self.ax.axis('on')

        # Data
        algos = ['FIFO', 'LRU', 'Optimal']
        faults = [f[0], l[0], o[0]]
        hits = [f[1], l[1], o[1]]

        # Colors
        bar_width = 0.6
        indices = range(len(algos))

        # Stacked Bar Chart
        p1 = self.ax.bar(indices, hits, bar_width, color='#00ADB5', label='Hits')  # Teal
        p2 = self.ax.bar(indices, faults, bar_width, bottom=hits, color='#393E46', edgecolor='white',
                         label='Faults')  # Grey

        # Styling
        self.ax.set_title('Performance Analysis (Lower Total Height is Better if looking at Faults)', color='white',
                          pad=20)
        self.ax.set_xticks(indices)
        self.ax.set_xticklabels(algos, color='white')
        self.ax.set_ylabel('Page Access Count', color='white')
        self.ax.tick_params(axis='y', colors='white')
        self.ax.legend(facecolor=COLORS["panel"], labelcolor='white')

        # Grid
        self.ax.grid(axis='y', linestyle='--', alpha=0.1)

        # Redraw
        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = ModernPageSimulator(root)
    root.mainloop()