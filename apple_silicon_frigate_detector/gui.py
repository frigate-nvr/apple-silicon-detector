#!/usr/bin/env python3
"""
GUI Application for Apple Silicon Frigate Detector
"""

import json
import logging
import os
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

from .zmq_onnx_client import ZmqOnnxClient


class FrigateDetectorGUI:
    """Simple GUI for the Frigate Detector."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Apple Silicon Frigate Detector")
        self.root.geometry("600x500")
        self.root.resizable(True, True)
        
        # State
        self.client = None
        self.server_thread = None
        self.is_running = False
        
        # Load settings
        self.settings_file = Path.home() / ".frigate_detector_settings.json"
        self.settings = self.load_settings()
        
        # Setup GUI
        self.setup_gui()
        self.setup_logging()
        
        # Load saved settings
        self.load_gui_settings()
        
    def setup_gui(self):
        """Setup the GUI elements."""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Model selection
        ttk.Label(main_frame, text="ONNX Model:").grid(row=0, column=0, sticky=tk.W, pady=5)
        
        model_frame = ttk.Frame(main_frame)
        model_frame.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=5)
        model_frame.columnconfigure(0, weight=1)
        
        self.model_var = tk.StringVar()
        self.model_entry = ttk.Entry(model_frame, textvariable=self.model_var)
        self.model_entry.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        
        ttk.Button(model_frame, text="Browse", command=self.browse_model).grid(row=0, column=1)
        
        # Endpoint
        ttk.Label(main_frame, text="Endpoint:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.endpoint_var = tk.StringVar(value="tcp://*:5555")
        ttk.Entry(main_frame, textvariable=self.endpoint_var).grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5)
        
        # Execution providers
        ttk.Label(main_frame, text="Providers:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.providers_var = tk.StringVar(value="CoreMLExecutionProvider CPUExecutionProvider")
        ttk.Entry(main_frame, textvariable=self.providers_var).grid(row=2, column=1, sticky=(tk.W, tk.E), pady=5)
        
        # Control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, columnspan=2, pady=20)
        
        self.start_button = ttk.Button(button_frame, text="Start Server", command=self.start_server)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(button_frame, text="Stop Server", command=self.stop_server, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # Status
        ttk.Label(main_frame, text="Status:").grid(row=4, column=0, sticky=tk.W, pady=(10, 5))
        self.status_var = tk.StringVar(value="Stopped")
        self.status_label = ttk.Label(main_frame, textvariable=self.status_var, foreground="red")
        self.status_label.grid(row=4, column=1, sticky=tk.W, pady=(10, 5))
        
        # Log output
        ttk.Label(main_frame, text="Log Output:").grid(row=5, column=0, sticky=(tk.W, tk.N), pady=(10, 5))
        
        log_frame = ttk.Frame(main_frame)
        log_frame.grid(row=5, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(5, weight=1)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=15, width=50)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Clear log button
        ttk.Button(log_frame, text="Clear Log", command=self.clear_log).grid(row=1, column=0, pady=5)
        
    def setup_logging(self):
        """Setup logging to display in the GUI."""
        class GUILogHandler(logging.Handler):
            def __init__(self, text_widget):
                super().__init__()
                self.text_widget = text_widget
                
            def emit(self, record):
                msg = self.format(record)
                self.text_widget.insert(tk.END, msg + '\n')
                self.text_widget.see(tk.END)
                
        # Setup logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        
        # Add GUI handler
        gui_handler = GUILogHandler(self.log_text)
        gui_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        
        logger = logging.getLogger()
        logger.addHandler(gui_handler)
        
    def browse_model(self):
        """Browse for ONNX model file."""
        filename = filedialog.askopenfilename(
            title="Select ONNX Model",
            filetypes=[("ONNX files", "*.onnx"), ("All files", "*.*")],
            initialdir=str(Path.home())
        )
        if filename:
            self.model_var.set(filename)
            
    def start_server(self):
        """Start the detector server."""
        model_path = self.model_var.get().strip()
        if not model_path:
            messagebox.showerror("Error", "Please select an ONNX model file")
            return
            
        if not Path(model_path).exists():
            messagebox.showerror("Error", f"Model file does not exist: {model_path}")
            return
            
        endpoint = self.endpoint_var.get().strip()
        providers_str = self.providers_var.get().strip()
        providers = [p.strip() for p in providers_str.split() if p.strip()] if providers_str else None
        
        try:
            # Save settings
            self.save_settings()
            
            # Create client
            self.client = ZmqOnnxClient(
                endpoint=endpoint,
                model_path=model_path,
                providers=providers
            )
            
            # Start server in separate thread
            self.server_thread = threading.Thread(target=self._run_server, daemon=True)
            self.server_thread.start()
            
            # Update UI
            self.is_running = True
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.status_var.set("Running")
            self.status_label.config(foreground="green")
            
            logging.info(f"Started detector server with model: {model_path}")
            logging.info(f"Endpoint: {endpoint}")
            if providers:
                logging.info(f"Providers: {providers}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start server: {e}")
            logging.error(f"Failed to start server: {e}")
            
    def _run_server(self):
        """Run the server (called in separate thread)."""
        try:
            self.client.start_server()
        except Exception as e:
            logging.error(f"Server error: {e}")
            # Reset UI on main thread
            self.root.after(0, self._server_stopped)
            
    def stop_server(self):
        """Stop the detector server."""
        if self.client:
            try:
                # This will cause the server loop to exit
                self.is_running = False
                if hasattr(self.client, 'socket'):
                    self.client.socket.close()
                if hasattr(self.client, 'context'):
                    self.client.context.term()
                logging.info("Server stopped")
            except Exception as e:
                logging.error(f"Error stopping server: {e}")
                
        self._server_stopped()
        
    def _server_stopped(self):
        """Update UI when server stops."""
        self.is_running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_var.set("Stopped")
        self.status_label.config(foreground="red")
        
    def clear_log(self):
        """Clear the log output."""
        self.log_text.delete(1.0, tk.END)
        
    def load_settings(self):
        """Load settings from file."""
        if self.settings_file.exists():
            try:
                with open(self.settings_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {}
        
    def save_settings(self):
        """Save current settings to file."""
        settings = {
            'model_path': self.model_var.get(),
            'endpoint': self.endpoint_var.get(),
            'providers': self.providers_var.get(),
        }
        try:
            with open(self.settings_file, 'w') as f:
                json.dump(settings, f, indent=2)
        except Exception as e:
            logging.warning(f"Failed to save settings: {e}")
            
    def load_gui_settings(self):
        """Load saved settings into GUI."""
        if 'model_path' in self.settings:
            self.model_var.set(self.settings['model_path'])
        if 'endpoint' in self.settings:
            self.endpoint_var.set(self.settings['endpoint'])
        if 'providers' in self.settings:
            self.providers_var.set(self.settings['providers'])
            
    def on_closing(self):
        """Handle window closing."""
        if self.is_running:
            if messagebox.askokcancel("Quit", "Server is running. Stop and quit?"):
                self.stop_server()
                self.root.destroy()
        else:
            self.root.destroy()


def main():
    """Main GUI entry point."""
    root = tk.Tk()
    app = FrigateDetectorGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
