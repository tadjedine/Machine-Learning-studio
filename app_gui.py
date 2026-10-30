import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Import  custom modules
from src.data_manager import DataManager
from src.clustering_engine import ClusteringEngine
from src.classification_engine import ClassificationEngine

class ClusteringApp:
    def __init__(self, master):
        self.master = master
        master.title("Data Analysis & Modeling Tool")
        master.geometry("1400x900")
        
        self.dm = DataManager()
        self.engine = ClusteringEngine()
        self.clf_engine = ClassificationEngine()
        
        # UI Styling
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure(".", font=("Segoe UI", 10), background="#f0f2f5")
        self.style.configure("TNotebook.Tab", font=("Segoe UI", 10, "bold"), padding=[10, 5])
        self.style.configure("Header.TLabel", font=("Segoe UI", 11, "bold"), foreground="#1a73e8")
        
        self._setup_tabs()

    def _setup_tabs(self):
        self.notebook = ttk.Notebook(self.master)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)

        self.tab_data = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_data, text="  DATASET & PREPROCESSING ")
        self._init_tab_data()

        self.tab_cluster = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_cluster, text="  CLUSTERING ")
        self._init_tab_cluster()

        self.tab_classif = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_classif, text="  CLASSIFICATION ")
        self._init_tab_classification()

    # --- TAB 1: DATASET ---
    def _init_tab_data(self):
        self.left_sidebar = ttk.Frame(self.tab_data, width=320)
        self.left_sidebar.pack(side='left', fill='y', padx=10, pady=10)
        self.left_sidebar.pack_propagate(False) 
        
        ctrl = ttk.LabelFrame(self.left_sidebar, text=" Configuration ")
        ctrl.pack(fill='both', expand=True)

        ttk.Button(ctrl, text="📂 Open Dataset", command=self._load_data).pack(fill='x', padx=5, pady=5)
        ttk.Button(ctrl, text="📊 Descriptive Statistics", command=self._show_stats_popup).pack(fill='x', padx=5, pady=2)
        
        self.lbl_file = ttk.Label(ctrl, text="File: None", foreground="#d93025", wraplength=280)
        self.lbl_file.pack(pady=5)
        
        ttk.Label(ctrl, text="Target/Label (Optional):", style="Header.TLabel").pack(anchor="w", padx=5)
        self.var_target = tk.StringVar()
        self.combo_target = ttk.Combobox(ctrl, textvariable=self.var_target, state="readonly")
        self.combo_target.pack(fill='x', padx=5, pady=5)

        ttk.Separator(ctrl).pack(fill='x', pady=10)
        
        self.var_clean_abe = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl, text="Fix Errors (Text -> NaN)", variable=self.var_clean_abe).pack(anchor="w", padx=5)
        self.var_label_enc = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl, text="Encode Categorical Text", variable=self.var_label_enc).pack(anchor="w", padx=5)

        ttk.Label(ctrl, text="Missing Values Strategy:").pack(anchor="w", padx=5, pady=(5,0))
        self.var_missing = tk.StringVar(value="mean")
        ttk.Combobox(ctrl, textvariable=self.var_missing, values=["drop", "mean", "median", "none"], state="readonly").pack(fill='x', padx=5)

        ttk.Label(ctrl, text="Normalization:").pack(anchor="w", padx=5, pady=(5,0))
        self.var_norm = tk.StringVar(value="zscore")
        ttk.Combobox(ctrl, textvariable=self.var_norm, values=["zscore", "minmax", "none"], state="readonly").pack(fill='x', padx=5)

        self.var_use_pca = tk.BooleanVar(value=False)
        ttk.Checkbutton(ctrl, text="Apply PCA ", variable=self.var_use_pca).pack(anchor="w", padx=5, pady=5)

        ttk.Button(ctrl, text="⚙️ APPLY PREPROCESSING", command=self._apply_preprocessing).pack(fill='x', padx=5, pady=(15, 2))

        right_frame = ttk.Frame(self.tab_data)
        right_frame.pack(side='right', fill='both', expand=True, padx=10, pady=10)
        
        ttk.Button(right_frame, text="📈 View Scatter Plot ", command=self._show_dataset_scatter).pack(fill='x', pady=5)
        
        self.display_container = ttk.LabelFrame(right_frame, text=" Data Preview ")
        self.display_container.pack(fill='both', expand=True)

    def _apply_preprocessing(self):
        target = self.var_target.get() if self.var_target.get() != "" else None
        self.dm.reset_data() 
        try:
            if self.var_clean_abe.get(): self.dm.clean_aberrant_values(target)
            if self.var_label_enc.get(): self.dm.apply_label_encoding(target)
            self.dm.handle_missing_values(self.var_missing.get())
            self.dm.normalize_data(self.var_norm.get(), target)
            if self.var_use_pca.get(): self.dm.apply_pca(n_components=2, target_to_keep=target)

            self._show_data_table()
            messagebox.showinfo("Success", "Preprocessing completed successfully.")
        except Exception as e: messagebox.showerror("Error", str(e))

    def _show_data_table(self):
        # Clear previous widgets
        for w in self.display_container.winfo_children(): w.destroy()
        
        data = self.dm.get_data()
        if data is None: return

        # Create a Frame to hold the Treeview and Scrollbars
        table_frame = ttk.Frame(self.display_container)
        table_frame.pack(fill='both', expand=True)

        df = data.head(100).round(3) # Show first 100 rows for performance
        cols = list(df.columns)
        
        # Define Scrollbars
        tree_scroll_y = ttk.Scrollbar(table_frame, orient="vertical")
        tree_scroll_x = ttk.Scrollbar(table_frame, orient="horizontal")

        # Define Treeview
        tree = ttk.Treeview(table_frame, columns=cols, show='headings', 
                            yscrollcommand=tree_scroll_y.set, 
                            xscrollcommand=tree_scroll_x.set)
        
        # Configure Scrollbars
        tree_scroll_y.config(command=tree.yview)
        tree_scroll_x.config(command=tree.xview)

        # Pack everything
        tree_scroll_y.pack(side='right', fill='y')
        tree_scroll_x.pack(side='bottom', fill='x')
        tree.pack(side='left', fill='both', expand=True)

        for c in cols:
            tree.heading(c, text=c)
            tree.column(c, width=120, anchor="center")
            
        for _, row in df.iterrows(): 
            tree.insert("", "end", values=list(row))

    def _show_dataset_scatter(self):
        data = self.dm.get_data()
        if data is None: return
        df_num = data.select_dtypes(include=[np.number])
        target = self.var_target.get()
        color_data = data[target] if target and target in data.columns else None

        if 'PC1' in df_num.columns and 'PC2' in df_num.columns:
            coords = df_num[['PC1', 'PC2']].values
            title = "PCA Projection "
        else:
            temp_df = df_num.drop(columns=[target]) if target and target in df_num.columns else df_num
            if temp_df.shape[1] < 2:
                messagebox.showerror("Error", "Need at least 2 numeric columns for 2D Scatter plot.")
                return
            # Handle potential NaNs for visualization only
            temp_df = temp_df.fillna(0)
            pca = PCA(n_components=2)
            coords = pca.fit_transform(temp_df)
            title = "Feature Projection (PCA)"

        win = tk.Toplevel(self.master)
        win.title(title)
        fig, ax = plt.subplots(figsize=(6, 5))
        
        if color_data is not None:
            # Handle string targets for coloring
            if color_data.dtype == 'object':
                codes, uniques = pd.factorize(color_data)
                sc = ax.scatter(coords[:, 0], coords[:, 1], c=codes, cmap='viridis', alpha=0.7, edgecolors='k')
                # Create a legend handles
                from matplotlib.lines import Line2D
                legend_elements = [Line2D([0], [0], marker='o', color='w', label=u, markerfacecolor=plt.cm.viridis(i/len(uniques)), markersize=10) for i, u in enumerate(uniques)]
                ax.legend(handles=legend_elements, title=target)
            else:
                sc = ax.scatter(coords[:, 0], coords[:, 1], c=color_data, cmap='viridis', alpha=0.7, edgecolors='k')
                plt.colorbar(sc, label=target)
        else:
             ax.scatter(coords[:, 0], coords[:, 1], alpha=0.7, edgecolors='k')

        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.get_tk_widget().pack(fill='both', expand=True); canvas.draw()

    # --- TAB 2: CLUSTERING ---
    def _init_tab_cluster(self):
        left_side = ttk.Frame(self.tab_cluster, width=320)
        left_side.pack(side='left', fill='y', padx=10, pady=10)
        left_side.pack_propagate(False)

        p_frame = ttk.LabelFrame(left_side, text=" Parameters ")
        p_frame.pack(fill='both', expand=True)

        ttk.Label(p_frame, text="Algorithm:").pack(anchor="w", padx=5)
        self.var_algo = tk.StringVar(value="KMeans")
        self.algo_menu = ttk.OptionMenu(p_frame, self.var_algo, "KMeans", "KMeans", "KMedoids", "AGNES", "DIANA", "DBSCAN", command=self._toggle_cluster_params)
        self.algo_menu.pack(fill='x', pady=5)

        # K Parameter Frame
        self.frame_k = ttk.Frame(p_frame)
        self.frame_k.pack(fill='x')
        ttk.Label(self.frame_k, text="Number of Clusters (K):").pack(anchor="w", padx=5)
        
        k_container = ttk.Frame(self.frame_k)
        k_container.pack(fill='x', padx=5)
        self.var_k = tk.StringVar(value="3")
        ttk.Entry(k_container, textvariable=self.var_k, width=10).pack(side='left', fill='x', expand=True)
        # Auto Detect Button
        ttk.Button(k_container, text=" Auto", width=6, command=self._auto_detect_k).pack(side='left', padx=(5,0))

        # DBSCAN Parameter Frame
        self.frame_dbscan = ttk.Frame(p_frame)
        ttk.Label(self.frame_dbscan, text="Epsilon (eps):").pack(anchor="w", padx=5)
        self.var_eps = tk.StringVar(value="0.5")
        ttk.Entry(self.frame_dbscan, textvariable=self.var_eps).pack(fill='x', padx=5)
        ttk.Label(self.frame_dbscan, text="Min Samples:").pack(anchor="w", padx=5)
        self.var_minpts = tk.StringVar(value="5")
        ttk.Entry(self.frame_dbscan, textvariable=self.var_minpts).pack(fill='x', padx=5)

        ttk.Button(p_frame, text=" Run Clustering", command=self._run_clustering).pack(fill='x', pady=10, padx=5)
        ttk.Button(p_frame, text="📉 Elbow Method", command=self._show_elbow).pack(fill='x', pady=2, padx=5)
        ttk.Button(p_frame, text="🌳 Dendrogram", command=self._show_dendro).pack(fill='x', pady=2, padx=5)
        ttk.Button(p_frame, text="📊 Algorithm Comparison", command=self._show_comparison).pack(fill='x', pady=2, padx=5)

        self.lbl_cluster_res = ttk.Label(p_frame, text="Metrics: -", foreground="#28a745", font=("Segoe UI", 10, "bold"))
        self.lbl_cluster_res.pack(pady=20)

        right_side = ttk.LabelFrame(self.tab_cluster, text=" Cluster Projection ")
        right_side.pack(side='right', fill='both', expand=True, padx=10, pady=10)
        self.fig_cl = plt.Figure(figsize=(5, 4), dpi=100)
        self.ax_cl = self.fig_cl.add_subplot(111)
        self.canvas_cl = FigureCanvasTkAgg(self.fig_cl, master=right_side)
        self.canvas_cl.get_tk_widget().pack(fill='both', expand=True)

    def _toggle_cluster_params(self, choice):
        if choice == "DBSCAN":
            self.frame_k.pack_forget()
            self.frame_dbscan.pack(fill='x', after=self.algo_menu)
        else:
            self.frame_dbscan.pack_forget()
            self.frame_k.pack(fill='x', after=self.algo_menu)

    def _validate_data_for_clustering(self):
        """Checks if data is ready for clustering."""
        data = self.dm.get_data()
        if data is None: 
            return None, "No dataset loaded."
        
        # Remove target if present
        df_num = data.select_dtypes(include=[np.number])
        t = self.var_target.get()
        if t and t in data.columns:
            if t in df_num.columns: 
                df_num = df_num.drop(columns=[t])
        
        # Check 1: Is there numeric data?
        if df_num.shape[1] == 0:
            return None, "No numeric columns found. Please use Preprocessing to Encode Categorical text."

        # Check 2: Are there NaNs?
        if df_num.isnull().values.any():
            return None, "Dataset contains Missing Values (NaN). Please use Preprocessing to fill or drop them."
            
        return df_num, None

    def _auto_detect_k(self):
        """Calculates optimal K using geometric Elbow method and updates UI."""
        df_num, error = self._validate_data_for_clustering()
        if error:
            messagebox.showerror("Preprocessing Error", error)
            return

        try:
            inertias = []
            K_range = range(1, 11)
            # Use small sample for speed if dataset is huge
            sample_data = df_num.sample(n=min(len(df_num), 1000), random_state=42)
            
            for k in K_range:
                model = KMeans(n_clusters=k, n_init='auto', random_state=42)
                model.fit(sample_data)
                inertias.append(model.inertia_)
            
            # Geometric Elbow Method (Max distance from line connecting first and last point)
            x0, y0 = 1, inertias[0]
            x1, y1 = 10, inertias[-1]
            distances = []
            for i, k in enumerate(K_range):
                # Distance formula of point (x,y) to line defined by p1, p2
                # abs((y2-y1)x - (x2-x1)y + x2y1 - y2x1) / sqrt((y2-y1)^2 + (x2-x1)^2)
                x, y = k, inertias[i]
                num = abs((y1-y0)*x - (x1-x0)*y + x1*y0 - y1*x0)
                den = np.sqrt((y1-y0)**2 + (x1-x0)**2)
                distances.append(num/den)
            
            optimal_k = K_range[np.argmax(distances)]
            self.var_k.set(str(optimal_k))
            messagebox.showinfo("Auto Detect", f"Optimal K detected: {optimal_k}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not detect K: {str(e)}")

    def _run_clustering(self):
        df_num, error = self._validate_data_for_clustering()
        if error:
            messagebox.showerror("Preprocessing Error", error)
            return
        
        try:
            k_val = int(self.var_k.get())
            params = {
                'k': k_val, 
                'eps': float(self.var_eps.get()), 
                'min_pts': int(self.var_minpts.get())
            }
            
            res = self.engine.run_algorithm(df_num, self.var_algo.get(), params)
            
            if "success" in res:
                metric_text = f"Silhouette: {res['metrics']['silhouette']}"
                if self.var_algo.get() != "DBSCAN": 
                    metric_text += f"\nInertia: {res['metrics']['inertia']}"
                self.lbl_cluster_res.config(text=metric_text)
                self._plot_cluster_graph(df_num, res['centroids'])
            else:
                messagebox.showerror("Clustering Error", res.get("error", "Unknown error"))
        except ValueError:
             messagebox.showerror("Input Error", "Check your parameters (K, Epsilon,..). They must be numbers.")

    def _plot_cluster_graph(self, df_num, centroids):
        self.fig_cl.clear()
        self.ax_cl = self.fig_cl.add_subplot(111)
        
        # PCA for visualization
        if df_num.shape[1] > 1:
            pca = PCA(n_components=2)
            coords = pca.fit_transform(df_num)
            
            # Scatter points
            sc = self.ax_cl.scatter(coords[:, 0], coords[:, 1], c=self.engine.labels, cmap='plasma', edgecolors='k', alpha=0.6)
            
            # Scatter centroids
            if centroids is not None:
                c_coords = pca.transform(centroids)
                self.ax_cl.scatter(c_coords[:, 0], c_coords[:, 1], marker='X', s=200, c='red', label='Centroids', edgecolors='white')
                self.ax_cl.legend()
        else:
             self.ax_cl.text(0.5, 0.5, "Not enough dimensions", ha='center')

        self.ax_cl.set_title(f"Clustering Result ({self.var_algo.get()})")
        self.canvas_cl.draw()

    # --- TAB 3: CLASSIFICATION ---
    def _init_tab_classification(self):
        left_side = ttk.Frame(self.tab_classif, width=320)
        left_side.pack(side='left', fill='y', padx=10, pady=10)
        p_frame = ttk.LabelFrame(left_side, text=" Model Settings ")
        p_frame.pack(fill='both', expand=True)

        self.var_algo_clf = tk.StringVar(value="k-NN")
        ttk.OptionMenu(p_frame, self.var_algo_clf, "k-NN", "k-NN", "Naive Bayes", "C4.5", "SVM").pack(fill='x', pady=5)
        ttk.Button(p_frame, text="🚀 Train & Test Model", command=self._run_classification).pack(fill='x', pady=10, padx=5)
        
        ttk.Button(p_frame, text="📈 K-Precision Curve ", command=self._plot_knn_curve).pack(fill='x', pady=2, padx=5)
        
        self.lbl_acc = ttk.Label(p_frame, text="Accuracy: -", font=("Segoe UI", 10, "bold"))
        self.lbl_acc.pack(anchor="w", padx=10, pady=(20, 2))
        self.lbl_prec = ttk.Label(p_frame, text="Precision: -")
        self.lbl_prec.pack(anchor="w", padx=10, pady=2)
        self.lbl_rec = ttk.Label(p_frame, text="Recall: -")
        self.lbl_rec.pack(anchor="w", padx=10)
        self.lbl_f1 = ttk.Label(p_frame, text="F1-Score: -")
        self.lbl_f1.pack(anchor="w", padx=10)

        right_side = ttk.LabelFrame(self.tab_classif, text=" Results Dashboard ")
        right_side.pack(side='right', fill='both', expand=True, padx=10, pady=10)
        self.fig_clf = plt.Figure(figsize=(5, 4), dpi=100)
        self.canvas_clf = FigureCanvasTkAgg(self.fig_clf, master=right_side)
        self.canvas_clf.get_tk_widget().pack(fill='both', expand=True)

    def _run_classification(self):
        target = self.var_target.get()
        if not target:
            messagebox.showerror("Selection Error", "Please select a Target column in the first tab.")
            return
        
        # Check if target is in dataset
        if target not in self.dm.get_data().columns:
             messagebox.showerror("Error", f"Target '{target}' not found in current dataset.")
             return

        try:
            splits = self.dm.split_data(target)
            if splits is None: return
            
            res = self.clf_engine.run_classification(*splits, self.var_algo_clf.get())
            
            self.lbl_acc.config(text=f"Accuracy: {res['accuracy']:.4f}")
            self.lbl_prec.config(text=f"Precision: {res['precision']:.4f}")
            self.lbl_rec.config(text=f"Recall: {res['recall']:.4f}")
            self.lbl_f1.config(text=f"F1-Score: {res['f1']:.4f}")
            
            self.fig_clf.clear()
            ax = self.fig_clf.add_subplot(111)
            sns.heatmap(res['cm'], annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title(f"Confusion Matrix: {self.var_algo_clf.get()}")
            self.canvas_clf.draw()
        except ValueError as ve:
             messagebox.showerror("Classification Error", f"Data issue: {ve}\nDid you encode categorical variables?")

    def _plot_knn_curve(self):
        target = self.var_target.get()
        if not target:
            messagebox.showerror("Selection Error", "Select a Target column in the first tab first.")
            return
        try:
            splits = self.dm.split_data(target)
            if splits:
                ks, precs = self.clf_engine.get_knn_curve_data(*splits)
                self.fig_clf.clear()
                ax = self.fig_clf.add_subplot(111)
                ax.plot(ks, precs, 'b-o', markersize=8)
                ax.set_title("K-Precision Curve (Neighbors 1-10)")
                ax.set_xlabel("Value of K (Neighbors)")
                ax.set_ylabel("Macro Precision")
                ax.grid(True, linestyle='--')
                self.canvas_clf.draw()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    # --- SHARED TOOLS ---
    def _load_data(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx")])
        if path:
            msg = self.dm.load_data(path)
            if "Error" in msg:
                 messagebox.showerror("Load Error", msg)
            else:
                self.lbl_file.config(text=f"File: {os.path.basename(path)}", foreground="#1a73e8")
                self.combo_target['values'] = [""] + list(self.dm.raw_data.columns)
                self._show_data_table()

    def _show_stats_popup(self):
        stats = self.dm.get_statistics()
        if stats is not None:
            win = tk.Toplevel(self.master)
            win.title("Stats")
            win.geometry("600x400")
            
            # Scrollbars for Stats
            f = ttk.Frame(win)
            f.pack(fill='both', expand=True)
            sc_y = ttk.Scrollbar(f, orient="vertical")
            sc_x = ttk.Scrollbar(f, orient="horizontal")
            
            tree = ttk.Treeview(f, columns=["Metric"] + list(stats.columns), show='headings',
                                yscrollcommand=sc_y.set, xscrollcommand=sc_x.set)
            
            sc_y.config(command=tree.yview)
            sc_x.config(command=tree.xview)
            sc_y.pack(side='right', fill='y')
            sc_x.pack(side='bottom', fill='x')
            tree.pack(side='left', fill='both', expand=True)

            for c in ["Metric"] + list(stats.columns): 
                tree.heading(c, text=c)
                tree.column(c, width=100)
            
            for i, r in stats.iterrows(): 
                tree.insert("", "end", values=[i] + list(r))

    def _show_elbow(self):
        df_num, error = self._validate_data_for_clustering()
        if error:
            messagebox.showerror("Preprocessing Error", error)
            return

        try:
            inertias = []
            ks = range(1, 11)
            for k in ks:
                inertias.append(KMeans(n_clusters=k, n_init='auto').fit(df_num).inertia_)
            
            self.fig_cl.clear()
            self.ax_cl = self.fig_cl.add_subplot(111)
            self.ax_cl.plot(ks, inertias, 'ro-')
            
            # Visualize the current K choice
            curr_k = int(self.var_k.get())
            if curr_k in ks:
                self.ax_cl.axvline(x=curr_k, color='blue', linestyle='--', label=f'Selected K={curr_k}')
                self.ax_cl.legend()
                
            self.ax_cl.set_title("Elbow Method")
            self.ax_cl.set_xlabel("Number of Clusters K")
            self.ax_cl.set_ylabel("Inertia")
            self.canvas_cl.draw()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _show_dendro(self):
        df_num, error = self._validate_data_for_clustering()
        if error:
            messagebox.showerror("Preprocessing Error", error)
            return
            
        from scipy.cluster.hierarchy import dendrogram, linkage
        try:
            # Use sample if data is too big
            data_to_plot = df_num if len(df_num) < 2000 else df_num.sample(2000)
            
            self.fig_cl.clear()
            self.ax_cl = self.fig_cl.add_subplot(111)
            dendrogram(linkage(data_to_plot, 'ward'), ax=self.ax_cl, truncate_mode='lastp', p=15)
            self.ax_cl.set_title("Dendrogram (Ward Linkage)")
            self.canvas_cl.draw()
        except Exception as e:
             messagebox.showerror("Error", str(e))

    def _show_comparison(self):
        df_num, error = self._validate_data_for_clustering()
        if error:
            messagebox.showerror("Preprocessing Error", error)
            return
            
        algos = ["KMeans", "KMedoids", "AGNES", "DIANA", "DBSCAN"]
        inertias = []
        p = {'k': int(self.var_k.get()), 'eps': float(self.var_eps.get()), 'min_pts': int(self.var_minpts.get())}
        
        self.fig_cl.clear()
        self.ax_cl = self.fig_cl.add_subplot(111)
        
        for a in algos:
            res = self.engine.run_algorithm(df_num, a, p)
            if "metrics" in res:
                inertias.append(res['metrics'].get('inertia', 0))
            else:
                inertias.append(0)
                
        self.ax_cl.bar(algos, inertias, color='#1a73e8')
        self.ax_cl.set_ylabel("Inertia / SSE")
        self.ax_cl.set_title("Algorithm Comparison")
        self.canvas_cl.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = ClusteringApp(root)
    root.mainloop()