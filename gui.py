# =============================================================================
# PyInstaller Compatibility and Error Handling
# =============================================================================
import sys
import os

# PyInstaller compatibility fixes
if getattr(sys, 'frozen', False):
    # Running as compiled executable
    os.environ['TCL_LIBRARY'] = os.path.join(sys._MEIPASS, 'tcl')
    os.environ['TK_LIBRARY'] = os.path.join(sys._MEIPASS, 'tk')
    os.environ['GDAL_DATA'] = os.path.join(sys._MEIPASS, 'gdal_data')
    os.environ['PROJ_LIB'] = os.path.join(sys._MEIPASS, 'proj_data')
    
    # Change to the temp directory where PyInstaller extracts files
    os.chdir(sys._MEIPASS)

# Wrap imports in try-except to catch startup errors
try:
    import customtkinter as ctk
    import tkinter as tk
    from tkinter import filedialog, messagebox
    import threading
    import pandas as pd
    import numpy as np
    from main_engine import run_complete_analysis
    from config import Config
except Exception as e:
    import traceback
    error_msg = f"Failed to import modules:\n{traceback.format_exc()}"
    
    # Write error to log file
    try:
        with open('startup_error.log', 'w') as f:
            f.write(error_msg)
    except:
        pass
    
    print(error_msg)
    input("Press Enter to exit...")
    sys.exit(1)






import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import threading
import pandas as pd
import numpy as np
from main_engine import run_complete_analysis
from config import Config

class LandslideGUI:
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("Landslide Susceptibility Modeler v2.0")
        self.root.geometry("900x700")
        
        # Set theme
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # Add parent directory to path for imports
        import sys
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.append(os.path.dirname(self.script_dir))

        self.models_df = None

        # Initialize variables
        self.spatial_cv_var = tk.BooleanVar(value=True)
        self.stacking_var = tk.BooleanVar(value=False)
        self.all_models_var = tk.BooleanVar(value=False)
        self.coordinates_var = tk.BooleanVar(value=False)

        self.setup_ui()
        
    def setup_ui(self):
        # Main container with padding
        main_frame = ctk.CTkFrame(self.root)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Title
        title = ctk.CTkLabel(main_frame, text="Landslide Susceptibility Modeler", 
                            font=ctk.CTkFont(size=24, weight="bold"))
        title.pack(pady=(0, 20))
        
        # File Selection Section
        file_frame = ctk.CTkFrame(main_frame)
        file_frame.pack(fill="x", pady=(0, 20))
        
        ctk.CTkLabel(file_frame, text="File Selection", 
                    font=ctk.CTkFont(size=16, weight="bold")).pack(anchor="w", padx=20, pady=(10, 10))
        
        # CSV Input
        csv_frame = ctk.CTkFrame(file_frame, fg_color="transparent")
        csv_frame.pack(fill="x", padx=20, pady=5)
        ctk.CTkLabel(csv_frame, text="Input CSV:", width=120, anchor="w").pack(side="left")
        self.csv_entry = ctk.CTkEntry(csv_frame, width=400)
        self.csv_entry.insert(0, os.path.join(self.script_dir, "input_data", "LS_Data.csv"))
        self.csv_entry.pack(side="left", padx=5)
        ctk.CTkButton(csv_frame, text="Browse", width=80, 
                     command=self.browse_csv).pack(side="left")
        
        # Raster Folder
        raster_frame = ctk.CTkFrame(file_frame, fg_color="transparent")
        raster_frame.pack(fill="x", padx=20, pady=5)
        ctk.CTkLabel(raster_frame, text="Raster Folder:", width=120, anchor="w").pack(side="left")
        self.raster_entry = ctk.CTkEntry(raster_frame, width=400)
        self.raster_entry.insert(0, os.path.join(self.script_dir, "input_rasters"))
        self.raster_entry.pack(side="left", padx=5)
        ctk.CTkButton(raster_frame, text="Browse", width=80, 
                     command=self.browse_raster).pack(side="left")
        
        # Output Folder
        output_frame = ctk.CTkFrame(file_frame, fg_color="transparent")
        output_frame.pack(fill="x", padx=20, pady=(5, 15))
        ctk.CTkLabel(output_frame, text="Output Folder:", width=120, anchor="w").pack(side="left")
        self.output_entry = ctk.CTkEntry(output_frame, width=400)
        self.output_entry.insert(0, os.path.join(self.script_dir, "output"))
        self.output_entry.pack(side="left", padx=5)
        ctk.CTkButton(output_frame, text="Browse", width=80, 
                     command=self.browse_output).pack(side="left")
        
        # Configuration Section
        config_frame = ctk.CTkFrame(main_frame)
        config_frame.pack(fill="x", pady=(0, 20))
        
        ctk.CTkLabel(config_frame, text="Configuration", 
                    font=ctk.CTkFont(size=16, weight="bold")).pack(anchor="w", padx=20, pady=(10, 10))
        
        # Model Selection
        model_frame = ctk.CTkFrame(config_frame, fg_color="transparent")
        model_frame.pack(fill="x", padx=20, pady=5)
        ctk.CTkLabel(model_frame, text="Model:", width=120, anchor="w").pack(side="left")
        self.model_var = tk.StringVar(value="")
        self.model_combo = ctk.CTkComboBox(model_frame, width=250, 
                                          values=["", "Random Forest", "XGBoost", "Support Vector Machine"],
                                          variable=self.model_var)
        self.model_combo.pack(side="left", padx=5)
        
        # Balance Strategy
        balance_frame = ctk.CTkFrame(config_frame, fg_color="transparent")
        balance_frame.pack(fill="x", padx=20, pady=5)
        ctk.CTkLabel(balance_frame, text="Balance Strategy:", width=120, anchor="w").pack(side="left")
        self.balance_var = tk.StringVar(value="smote")
        self.balance_combo = ctk.CTkComboBox(balance_frame, width=250, 
                                            values=["smote", "undersample", "none"],
                                            variable=self.balance_var)
        self.balance_combo.pack(side="left", padx=5)
        
        # Workers
        workers_frame = ctk.CTkFrame(config_frame, fg_color="transparent")
        workers_frame.pack(fill="x", padx=20, pady=5)
        ctk.CTkLabel(workers_frame, text="Parallel Workers:", width=120, anchor="w").pack(side="left")
        self.workers_var = tk.IntVar(value=4)
        self.workers_slider = ctk.CTkSlider(workers_frame, from_=1, to=16, number_of_steps=15,
                                           variable=self.workers_var, width=200)
        self.workers_slider.pack(side="left", padx=5)
        self.workers_label = ctk.CTkLabel(workers_frame, text="4", width=30)
        self.workers_label.pack(side="left", padx=5)
        self.workers_slider.configure(command=self.update_workers_label)
        
        # Checkboxes
        checkbox_frame = ctk.CTkFrame(config_frame, fg_color="transparent")
        checkbox_frame.pack(fill="x", padx=20, pady=(5, 15))
        
        ctk.CTkCheckBox(checkbox_frame, text="Use Spatial CV", 
                       variable=self.spatial_cv_var).pack(side="left", padx=10)
        ctk.CTkCheckBox(checkbox_frame, text="Analyze All Models", 
                       variable=self.all_models_var).pack(side="left", padx=10)
        ctk.CTkCheckBox(checkbox_frame, text="Require Coordinates", 
                       variable=self.coordinates_var).pack(side="left", padx=10)
        
        # Log Section
        log_frame = ctk.CTkFrame(main_frame)
        log_frame.pack(fill="both", expand=True, pady=(0, 20))
        
        ctk.CTkLabel(log_frame, text="Log Output", 
                    font=ctk.CTkFont(size=16, weight="bold")).pack(anchor="w", padx=20, pady=(10, 10))
        
        self.log_text = ctk.CTkTextbox(log_frame, height=200, width=800)
        self.log_text.pack(fill="both", expand=True, padx=20, pady=(0, 15))
        
        # Progress Bar
        self.progress_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        self.progress_frame.pack(fill="x", padx=20, pady=(0, 10))
        ctk.CTkLabel(self.progress_frame, text="Progress:", width=120, anchor="w").pack(side="left")
        self.progress_bar = ctk.CTkProgressBar(self.progress_frame)
        self.progress_bar.pack(side="left", fill="x", expand=True, padx=5)
        self.progress_bar.set(0.0)
        self.progress_label = ctk.CTkLabel(self.progress_frame, text="0%", width=50)
        self.progress_label.pack(side="left", padx=5)

        # Buttons
        button_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        button_frame.pack(fill="x")

        self.compare_button = ctk.CTkButton(button_frame, text="Compare Models",
                                           command=self.compare_models,
                                           height=40, font=ctk.CTkFont(size=14, weight="bold"),
                                           fg_color="blue", hover_color="darkblue")
        self.compare_button.pack(side="left", padx=5)

        self.run_button = ctk.CTkButton(button_frame, text="Run Analysis",
                                       command=self.run_analysis,
                                       height=40, font=ctk.CTkFont(size=14, weight="bold"),
                                       fg_color="green", hover_color="darkgreen")
        self.run_button.pack(side="left", padx=5)
        
        ctk.CTkButton(button_frame, text="Clear Log", 
                     command=self.clear_log, height=40).pack(side="left", padx=5)
        
        ctk.CTkButton(button_frame, text="Exit", 
                     command=self.root.destroy, height=40,
                     fg_color="red", hover_color="darkred").pack(side="right", padx=5)
    
    def update_workers_label(self, value):
        self.workers_label.configure(text=str(int(value)))

    def update_progress(self, value):
        """Update progress bar - can be called from any thread due to root.after() wrapper"""
        self.progress_bar.set(value / 100.0)
        self.progress_label.configure(text=f"{int(value)}%")
        self.root.update_idletasks()

    def browse_csv(self):
        filename = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if filename:
            self.csv_entry.delete(0, tk.END)
            self.csv_entry.insert(0, filename)
    
    def browse_raster(self):
        folder = filedialog.askdirectory()
        if folder:
            self.raster_entry.delete(0, tk.END)
            self.raster_entry.insert(0, folder)
    
    def browse_output(self):
        folder = filedialog.askdirectory()
        if folder:
            self.output_entry.delete(0, tk.END)
            self.output_entry.insert(0, folder)
    
    def log(self, message):
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")
    
    def clear_log(self):
        self.log_text.delete("1.0", "end")
    
    def compare_models(self):
        self.compare_button.configure(state="disabled", text="Comparing...")
        self.log("Starting model comparison...\n")

        thread = threading.Thread(target=self.compare_models_thread, daemon=True)
        thread.start()

    def run_analysis(self):
        if self.models_df is None:
            messagebox.showerror("Error", "Please run model comparison first to select a model.")
            return
        self.run_button.configure(state="disabled", text="Running...")
        self.log("Starting full analysis...\n")

        thread = threading.Thread(target=self.run_analysis_thread, daemon=True)
        thread.start()

    def compare_models_thread(self):
        try:
            # Update config
            Config.BALANCE_STRATEGY = self.balance_var.get()
            Config.USE_SPATIAL_CV = self.spatial_cv_var.get()
            Config.USE_STACKING = self.stacking_var.get()
            Config.PARALLEL_WORKERS = self.workers_var.get()
            Config.REQUIRE_COORDINATES = self.coordinates_var.get()

            csv_path = self.csv_entry.get()
            raster_path = self.raster_entry.get()
            output_path = self.output_entry.get()

            # Validate
            if not os.path.exists(csv_path):
                self.root.after(0, lambda: self.show_error("Input CSV not found"))
                return
            if not os.path.exists(raster_path):
                self.root.after(0, lambda: self.show_error("Raster folder not found"))
                return

            os.makedirs(output_path, exist_ok=True)

            result = run_complete_analysis(
                csv_path,
                raster_path,
                output_path,
                None,
                False,
                gui_log_callback=lambda msg: self.root.after(0, lambda m=msg: self.log(m)),
                progress_callback=lambda val: self.root.after(0, lambda v=val: self.update_progress(v)),
                mode='compare'
            )

            if isinstance(result, pd.DataFrame):
                self.models_df = result
                # Sort by 'Accuracy' or first numeric column
                if 'Accuracy' in result.columns:
                    sorted_df = result.sort_values('Accuracy', ascending=False)
                else:
                    numeric_cols = result.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        sorted_df = result.sort_values(numeric_cols[0], ascending=False)
                    else:
                        sorted_df = result
                models = sorted_df['Model'].tolist()
                self.root.after(0, lambda: self.model_combo.configure(values=models))
                self.root.after(0, lambda: self.log(f"Model comparison results:\n{sorted_df.to_string()}\n\nSelect a model and run full analysis.\n"))
                self.root.after(0, lambda: self.show_success("Model comparison completed!"))
            else:
                self.root.after(0, lambda: self.show_error("Model comparison failed."))

        except Exception as e:
            self.root.after(0, lambda: self.show_error(f"Comparison failed: {str(e)}"))
            import traceback
            print(traceback.format_exc())
        finally:
            self.root.after(0, self.reset_compare_button)

    def run_analysis_thread(self):
        try:
            # Update config
            Config.BALANCE_STRATEGY = self.balance_var.get()
            Config.USE_SPATIAL_CV = self.spatial_cv_var.get()
            Config.PARALLEL_WORKERS = self.workers_var.get()
            Config.REQUIRE_COORDINATES = self.coordinates_var.get()
            
            csv_path = self.csv_entry.get()
            raster_path = self.raster_entry.get()
            output_path = self.output_entry.get()
            model = self.model_var.get() if self.model_var.get() else None
            
            # Validate
            if not os.path.exists(csv_path):
                self.root.after(0, lambda: self.show_error("Input CSV not found"))
                return
            if not os.path.exists(raster_path):
                self.root.after(0, lambda: self.show_error("Raster folder not found"))
                return
            
            os.makedirs(output_path, exist_ok=True)
            
            # Determine mode
            mode = 'map_only' if self.models_df is not None else 'full'
            analyze_all = self.all_models_var.get() if mode == 'full' else False

            success = run_complete_analysis(
                csv_path,
                raster_path,
                output_path,
                model,
                analyze_all,
                gui_log_callback=lambda msg: self.root.after(0, lambda m=msg: self.log(m)),
                progress_callback=lambda val: self.root.after(0, lambda v=val: self.update_progress(v)),
                mode=mode
            )
            
            if success:
                self.root.after(0, lambda: self.show_success("Analysis completed successfully!"))
            else:
                self.root.after(0, lambda: self.show_error("Analysis failed. Check logs."))
                
        except Exception as e:
            self.root.after(0, lambda: self.show_error(f"Analysis failed: {str(e)}"))
            import traceback
            print(traceback.format_exc())
        finally:
            self.root.after(0, self.reset_run_button)
    
    def reset_compare_button(self):
        self.compare_button.configure(state="normal", text="Compare Models")

    def reset_run_button(self):
        self.run_button.configure(state="normal", text="Run Analysis")
    
    def show_error(self, message):
        self.log(f"ERROR: {message}\n")
        messagebox.showerror("Error", message)

    def show_success(self, message):
        self.log(f"SUCCESS: {message}\n")
        messagebox.showinfo("Success", message)
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    try:
        app = LandslideGUI()
        app.run()
    except Exception as e:
        import traceback
        error_msg = f"Application Error:\n{traceback.format_exc()}"
        print(error_msg)
        
        # Write error to file
        try:
            with open('crash_error.log', 'w') as f:
                f.write(error_msg)
        except:
            pass
        
        # Show error dialog if possible
        try:
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("Startup Error", error_msg)
        except:
            pass
        
        input("Press Enter to exit...")
        sys.exit(1)
    
