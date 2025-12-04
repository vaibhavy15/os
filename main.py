import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class PageReplacementSimulator:
    def __init__(self, root):
        self.root = root
        self.root.title("Page Replacement Algorithm Simulator")
        self.root.geometry("900x700")

        # Styles
        style = ttk.Style()
        style.theme_use('clam')

        # --- UI Layout ---
        self.setup_ui()

    def setup_ui(self):
        # Input Frame
        input_frame = ttk.LabelFrame(self.root, text="Configuration", padding=(10, 10))
        input_frame.pack(fill="x", padx=10, pady=5)

        # Reference String Input
        ttk.Label(input_frame, text="Reference String (space separated):").grid(row=0, column=0, padx=5, sticky="w")
        self.ref_string_entry = ttk.Entry(input_frame, width=50)
        self.ref_string_entry.insert(0, "7 0 1 2 0 3 0 4 2 3 0 3 2 1 2 0 1 7 0 1")  # Default case
        self.ref_string_entry.grid(row=0, column=1, padx=5, pady=5)

        # Number of Frames Input
        ttk.Label(input_frame, text="Number of Frames:").grid(row=1, column=0, padx=5, sticky="w")
        self.frames_entry = ttk.Entry(input_frame, width=10)
        self.frames_entry.insert(0, "3")
        self.frames_entry.grid(row=1, column=1, padx=5, sticky="w")

        # Buttons
        btn_frame = ttk.Frame(input_frame)
        btn_frame.grid(row=2, column=1, pady=10, sticky="w")
        ttk.Button(btn_frame, text="Run Simulation", command=self.run_simulation).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Clear", command=self.clear_inputs).pack(side="left", padx=5)

        # Output Area (Text Logs)
        self.log_area = tk.Text(self.root, height=15, width=100, state='disabled', bg="#f0f0f0")
        self.log_area.pack(padx=10, pady=5)

        # Graph Area
        self.graph_frame = ttk.Frame(self.root)
        self.graph_frame.pack(fill="both", expand=True, padx=10, pady=5)

    def log(self, message):
        self.log_area.config(state='normal')
        self.log_area.insert(tk.END, message + "\n")
        self.log_area.see(tk.END)
        self.log_area.config(state='disabled')

    def clear_inputs(self):
        self.ref_string_entry.delete(0, tk.END)
        self.frames_entry.delete(0, tk.END)

    # --- ALGORITHMS ---

    def fifo(self, pages, capacity):
        memory = []
        page_faults = 0
        hits = 0

        for page in pages:
            if page not in memory:
                if len(memory) >= capacity:
                    memory.pop(0)
                memory.append(page)
                page_faults += 1
            else:
                hits += 1
        return page_faults, hits

    def lru(self, pages, capacity):
        memory = []
        page_faults = 0
        hits = 0

        for page in pages:
            if page not in memory:
                if len(memory) >= capacity:
                    memory.pop(0)  # Remove least recently used (front of list)
                memory.append(page)
                page_faults += 1
            else:
                memory.remove(page)
                memory.append(page)  # Move to end (most recently used)
                hits += 1
        return page_faults, hits

    def optimal(self, pages, capacity):
        memory = []
        page_faults = 0
        hits = 0

        for i, page in enumerate(pages):
            if page not in memory:
                if len(memory) < capacity:
                    memory.append(page)
                else:
                    # Look ahead to find the page that will not be used for longest time
                    farthest_idx = -1
                    victim_idx = -1

                    for m_idx, mem_page in enumerate(memory):
                        try:
                            # Slice list from current index forward
                            next_use = pages[i + 1:].index(mem_page)
                        except ValueError:
                            next_use = float('inf')

                        if next_use > farthest_idx:
                            farthest_idx = next_use
                            victim_idx = m_idx

                    memory[victim_idx] = page
                page_faults += 1
            else:
                hits += 1
        return page_faults, hits

    # --- CONTROLLER ---

    def run_simulation(self):
        try:
            # Parse inputs
            ref_str_raw = self.ref_string_entry.get()
            pages = list(map(int, ref_str_raw.split()))
            capacity = int(self.frames_entry.get())

            # Clear logs and graphs
            self.log_area.config(state='normal')
            self.log_area.delete(1.0, tk.END)
            self.log_area.config(state='disabled')
            for widget in self.graph_frame.winfo_children():
                widget.destroy()

            # Run Algos
            fifo_f, fifo_h = self.fifo(pages, capacity)
            lru_f, lru_h = self.lru(pages, capacity)
            opt_f, opt_h = self.optimal(pages, capacity)

            # Logging Results
            self.log(f"--- SIMULATION RESULTS (Frames: {capacity}) ---")
            self.log(f"Total Pages Requested: {len(pages)}")
            self.log("-" * 40)
            self.log(f"FIFO    -> Faults: {fifo_f} | Hits: {fifo_h} | Hit Ratio: {fifo_h / len(pages):.2f}")
            self.log(f"LRU     -> Faults: {lru_f} | Hits: {lru_h} | Hit Ratio: {lru_h / len(pages):.2f}")
            self.log(f"OPTIMAL -> Faults: {opt_f} | Hits: {opt_h} | Hit Ratio: {opt_h / len(pages):.2f}")

            # Visualization
            self.plot_results(fifo_f, lru_f, opt_f, fifo_h, lru_h, opt_h)

        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid integers for frames and reference string.")

    def plot_results(self, f_f, l_f, o_f, f_h, l_h, o_h):
        algorithms = ['FIFO', 'LRU', 'Optimal']
        faults = [f_f, l_f, o_f]
        hits = [f_h, l_h, o_h]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        # Bar Chart - Faults vs Hits
        x = range(len(algorithms))
        width = 0.35

        ax1.bar([i - width / 2 for i in x], faults, width, label='Faults', color='#ff6b6b')
        ax1.bar([i + width / 2 for i in x], hits, width, label='Hits', color='#4ecdc4')
        ax1.set_ylabel('Count')
        ax1.set_title('Faults vs Hits Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(algorithms)
        ax1.legend()

        # Efficiency Line (Hit Ratio)
        ratios = [h / (h + f) * 100 for h, f in zip(hits, faults)]
        ax2.plot(algorithms, ratios, marker='o', linestyle='-', color='#1a535c')
        ax2.set_ylabel('Hit Ratio (%)')
        ax2.set_title('Algorithm Efficiency')
        ax2.set_ylim(0, 100)
        for i, v in enumerate(ratios):
            ax2.text(i, v + 2, f"{v:.1f}%", ha='center')

        # Embed into Tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)


if __name__ == "__main__":
    root = tk.Tk()
    app = PageReplacementSimulator(root)
    root.mainloop()