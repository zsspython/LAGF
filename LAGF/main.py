import tkinter as tk
from tkinter import filedialog,messagebox
import os
import numpy as np 
import pandas as pd
import pymzml
from scipy.interpolate import interp1d
from joblib import Parallel, delayed
from scipy.ndimage import uniform_filter1d
from scipy.signal import savgol_filter
import sys
sys.path.append("..")
from LAGF.extract_EIC_LAGF import extract_EIC_LAGF
from LAGF.auto_min_peakheight import estimate_min_peak_height
from LAGF.peaks import Peaks
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from io import BytesIO
from tkinter import ttk
import matplotlib
matplotlib.use("TkAgg")


def close_app():
    root.destroy()
    
    
class StdoutRedirector(object):
    # Redirects the output class
    def __init__(self,text_widget):
        self.t = text_widget
        self.stdoutbak = sys.stdout
        self.stderrbak = sys.stderr

    def write(self, info):
        # The info information is the output information received by the standard output sys.stdout and sys.stderr
        self.t.insert('end', '\n')
        self.t.insert('insert', info)
        self.t.update()	
        self.t.see(tk.END)
    
    def restoreStd(self):
        # Recovery standard output
        sys.stdout = self.stdoutbak
        sys.stderr = self.stderrbak
    
    def flush(self):
        pass

class MyGUI:
    def __init__(self, master):
        self.master = master
        master.title("Mass Spectrometry Analysis")
        master.geometry("600x600") 

        window_width = master.winfo_reqwidth()
        window_height = master.winfo_reqheight()
        position_right = int(master.winfo_screenwidth() / 2 - window_width )
        position_down = int(master.winfo_screenheight() / 2 - window_height)
        master.geometry("+{}+{}".format(position_right, position_down))
        
        menubar = tk.Menu(self.master)
        self.master.config(menu=menubar)

        fileMenu = tk.Menu(menubar)
        fileMenu.add_command(label="Exit", command=close_app)
        fileMenu.add_command(label="Show feature table", command=self.show_peak_table)
        menubar.add_cascade(label="File", menu=fileMenu)

        # Part 1 - File reading
        self.file_frame = tk.Frame(master, padx=10, pady=10, bd=2, relief=tk.GROOVE)
        self.file_frame.pack(padx=10, pady=10, fill=tk.X)

        self.import_button = tk.Button(self.file_frame, text=" Select mzML File ", command=self.open_file)
        self.import_button.pack(side=tk.LEFT)

        self.file_label = tk.Label(self.file_frame, text="Selected File: None")
        self.file_label.pack(side=tk.LEFT, padx=10)

        # Part 2 - Extract EIC
        self.extract_frame = tk.Frame(master, padx=10, pady=10, bd=2, relief=tk.GROOVE)
        self.extract_frame.pack(pady=10, fill=tk.X)
        
        self.mz_tolerance_label = tk.Label(self.extract_frame, text="MZ tolerance (Da):")
        self.mz_tolerance_label.grid(row=0, column=0, pady=(10, 0))
        self.mz_tolerance_entry = tk.Entry(self.extract_frame)
        self.mz_tolerance_entry.insert(0,0.005)
        self.mz_tolerance_entry.grid(row=0, column=1, pady=(10, 0))


        self.min_intensity_label = tk.Label(self.extract_frame, text="Minimum intensity threshold:")
        self.min_intensity_label.grid(row=1, column=0)
        self.min_intensity_entry = tk.Entry(self.extract_frame)
        self.min_intensity_entry.insert(0,1000)
        self.min_intensity_entry.grid(row=1, column=1)


        # self.min_peak_height_label = tk.Label(self.extract_frame, text="Minimum peak height threshold:")
        # self.min_peak_height_label.grid(row=2, column=0)
        # self.min_peak_height_entry = tk.Entry(self.extract_frame)
        # self.min_peak_height_entry.insert(0,1000)
        # self.min_peak_height_entry.grid(row=2, column=1)

        self.star_button = tk.Button(self.extract_frame, text="Extract EIC", command=self.run_star)
        self.star_button.grid(row=3, column=1, columnspan=2, pady=(10, 0))

        # Part 3 - Peak Detection
        self.peak_frame = tk.Frame(master, padx=10, pady=10, bd=2, relief=tk.GROOVE)
        self.peak_frame.pack(pady=10, fill=tk.X)

        self.gaussian_fit_threshold_label = tk.Label(self.peak_frame, text="        Gaussian fit threshold:       ")
        self.gaussian_fit_threshold_label.grid(row=0, column=0, pady=(10, 0))
        self.gaussian_fit_threshold_entry = tk.Entry(self.peak_frame)
        self.gaussian_fit_threshold_entry.insert(0,0.5)
        self.gaussian_fit_threshold_entry.grid(row=0, column=1, pady=(10, 0))

        self.sn = tk.Label(self.peak_frame, text="S/N:")
        self.sn.grid(row=1, column=0)
        self.sn = tk.Entry(self.peak_frame)
        self.sn.insert(0,3)
        self.sn.grid(row=1, column=1)

        self.peak_width_label = tk.Label(self.peak_frame, text="Peak width:")
        self.peak_width_label.grid(row=2, column=0)
        self.peak_width_entry = tk.Entry(self.peak_frame)
        self.peak_width_entry.insert(0,5)
        self.peak_width_entry.grid(row=2, column=1)

        self.peak_button = tk.Button(self.peak_frame, text="Peak Detection", command=self.run_peak_detection)
        self.peak_button.grid(row=3, column=1, columnspan=2, pady=(10, 0))

        self.output_text = tk.Text(master, height=10, wrap=tk.WORD)
        self.output_text.insert("insert","------Thanks for using this software------")
        self.output_text.pack(pady=10, fill=tk.X)
        sys.stdout = StdoutRedirector(self.output_text)
    
    def open_file(self):
        global result_dict
        result_dict = { "mz_tolerance":[],
                "min_intensity":[],
                "min_peak_height":[],
                "gaussian_fit_threshold":[],
                "sn":[],
                "peak_width":[],
                "ms":[],
                "EIC":[],
                }
        mzml_file = filedialog.askopenfilename(
            initialdir=(os.getcwd()),
            filetypes=[("mzML file", ".mzML .mzml")],
            title=("Select mzml file"),
            )
        if mzml_file:
            self.file_label.config(text=f"Selected File: {mzml_file}")
            result_dict["file_path"]=mzml_file
            mzml_file = os.path.abspath(mzml_file)
            ms=pymzml.run.Reader((mzml_file),MS_precisions =  {
                    1 : 5e-6,
                    2 : 20e-6
                })
            num=0
            ms1=[]
            ms1_i=[]
            ms1_mz=[]
            ms2=[]
            ms1_id=[]
            ms1_ii=[]
            ms1_rt=[]
            Base_peak=np.array([])
            for i in ms :
                if i.ms_level == 1: 
                    ms1.append(i)
                    ms1_i.append(i.i)
                    ms1_id.append(i.ID)
                    ms1_mz.append(i.mz)
                    ms1_ii.append(num)
                    ms1_rt.append(i.scan_time[0])
                    num+=1
                    Base_peak=np.append(Base_peak,max(i.i))
                else:
                    ms2.append(i)
            ms1_={"id":ms1_id,"intensity":ms1_i,"mz":ms1_mz,"ii":ms1_ii,"rt":ms1_rt,"BPC":Base_peak}
            result_dict["ms"]=ms1_
            print("---The mzml file is loaded---")
            print("-------------------------------------------",f"Ms1 number of scans: {len(ms1)}",f"Ms2 number of scans: {len(ms2)}",f"RT range: {'%.2f'%ms1_rt[0]} - {'%.2f'%ms1_rt[-1]}","-------------------------------------------")
        else:
            messagebox.showinfo("Warning", "Please select the correct file format")
            return

    def run_star(self):
        if result_dict["ms"]!=[]:
            try:
                mz_tolerance = float(self.mz_tolerance_entry.get())
                result_dict["mz_tolerance"]=mz_tolerance
            except:
                messagebox.showinfo("Warning", "Please enter the correct mz_tolrance (int or float)")
                return
            try:
                min_intensity = float(self.min_intensity_entry.get())
                result_dict["min_intensity"]=min_intensity
            except:
                messagebox.showinfo("Warning", "Please enter the correct min intensity (int or float)")
                return
            result_dict["min_peak_height"]=1000
            # try:
            #     min_peak_height = float(self.min_peak_height_entry.get())
            #     result_dict["min_peak_height"]=min_peak_height
            # except:
            #     messagebox.showinfo("Warning", "Please enter the correct min peak height (int or float)")
            #     return
            self.process_data(result_dict["ms"], mz_tolerance, min_intensity, 1000,"EIC")
        else:
            messagebox.showinfo("Warning", "Please import mzml file first")
            return


    def run_peak_detection(self):
        if result_dict["ms"]!=[]:
            if result_dict["EIC"]!=[]:
                try:
                    gaussian_fit_threshold = float(self.gaussian_fit_threshold_entry.get())
                    result_dict["gaussian_fit_threshold"]=gaussian_fit_threshold
                except:
                    messagebox.showinfo("Warning", "Please enter the correct gaussian fit threshold (int or float )")
                    return
                try:
                    sn_value = float(self.sn.get())
                    result_dict["sn"]=sn_value
                except:
                    messagebox.showinfo("Warning", "Please enter the correct sn value (int or float )")
                    return
                try:
                    peak_width = float(self.peak_width_entry.get())
                    result_dict["peak_width"]=peak_width
                except:
                    messagebox.showinfo("Warning", "Please enter the correct peak width (int or float )")
                    return
                if result_dict["gaussian_fit_threshold"] and result_dict["sn"] and result_dict["peak_width"]:
                    self.process_data(result_dict["EIC"],gaussian_fit_threshold,sn_value, peak_width,  "peak_detection")
                else:
                    messagebox.showinfo("Warning", "Please complete EIC extraction first")
                    return
            else:
                messagebox.showinfo("Warning", "Please complete EIC extraction first")
                return
        else:
            messagebox.showinfo("Warning", "Please import mzml file firs")
            return


    def process_data(self, param1, param2, param3, param4, param5):
        if param5=="EIC":
            print("---EIC extraction in progress---")
            EIC=extract_EIC_LAGF(ms1_=param1, mz_tolerance=param2, min_intensity=param3, min_peak_height=param4)
            result_dict["EIC"]=EIC ###dict{ 'rt_numbers': rt_numbers,'rt_times': ms1_["rt"],'tracks': updated_tracks,'mz_range': mz_}
            min_peak_height=estimate_min_peak_height(result_dict["file_path"],EIC)
            result_dict["min_peak_height"]=min_peak_height
            print("---The extraction process of EIC has been successfully completed---")
            print("-------------------------------------------",f"mz range:{'%.4f'%EIC['mz_range'][0]}-{'%.4f'%EIC['mz_range'][1]}","-------------------------------------------")
        elif param5=="peak_detection":
            print("---peak detection in progress---")
            alldata,feature=Peaks(track_mass=param1,min_peak_height=result_dict["min_peak_height"],goodness_fitting=result_dict["gaussian_fit_threshold"],s_n=int(result_dict["sn"]),peak_width=int(result_dict["peak_width"])) ##[mz, index, cent_index, left_index, right_index, pro]
            for i in range(len(feature)) :
                feature[i].insert(0,i)
            Feature=[i[:] for i in feature]
            result_dict["feature"]=feature
            result_dict["Feature"]=Feature
            for i in range(len(feature)) :
                result_dict["Feature"][i][3]=result_dict["ms"]["rt"][feature[i][3]]
                result_dict["Feature"][i][4]=result_dict["ms"]["rt"][feature[i][4]]
                result_dict["Feature"][i][5]=result_dict["ms"]["rt"][feature[i][5]]
                result_dict["Feature"][i][2]=int(result_dict["Feature"][i][2])
            self.peak_df=pd.DataFrame(result_dict["Feature"],columns=["index","mz","EIC_id","RT_cent","RT_left","RT_right","prominences","goodness_fitting","area_removebaseline","area","SN"])
            self.peak_df["index"]=self.peak_df["index"].astype(int)
            self.peak_df["EIC_id"]=self.peak_df["EIC_id"].astype(int)
            result_dict["alldata"]=alldata
            print("---peak detection has been successfully completed---")
        # return 


    def show_peak_table(self):
        ##Popup interface
        try:
            self.peak_df
        except:
            messagebox.showinfo("Warning", "Please complete Feature Detection first.")
            return

        new_window = tk.Toplevel(self.master)
        new_window.title("Feature Table")
        
        ##menu 
        menubar1 = tk.Menu(new_window)

        fileMenu = tk.Menu(menubar1)
        fileMenu.add_command(label="Save the current peak table as CSV", command=self.save_peak_table)
        menubar1.add_cascade(label="Save File", menu=fileMenu)
        new_window.config(menu=menubar1)

        # Part 1: retrieval function
        self.search_frame = ttk.LabelFrame(new_window, text="Search Function")
        self.search_frame.pack(padx=10, pady=10, fill="x")

        self.mz_label = ttk.Label(self.search_frame, text="m/z:")
        self.mz_label.grid(row=0, column=0, padx=(0, 10))

        self.sn_label = ttk.Label(self.search_frame, text="SN:")
        self.sn_label.grid(row=0, column=2, padx=(0, 10))

        self.goodness_label = ttk.Label(self.search_frame, text="Goodness:")
        self.goodness_label.grid(row=0, column=4, padx=(0, 10))

        self.mz_entry = ttk.Entry(self.search_frame)
        self.mz_entry.grid(row=0, column=1)

        self.sn_entry = ttk.Entry(self.search_frame)
        self.sn_entry.grid(row=0, column=3)

        self.goodness_entry = ttk.Entry(self.search_frame)
        self.goodness_entry.grid(row=0, column=5)

        self.search_button = ttk.Button(self.search_frame, text="Search", command=self.perform_search)
        self.search_button.grid(row=0, column=6, padx=(10, 0))

        # Part 2: The Treeview displays the peak table information
        self.treeview_frame = ttk.LabelFrame(new_window, text="Peak Table")
        self.treeview_frame.pack(padx=10, pady=10, fill="both", expand=True)

        self.treeview = ttk.Treeview(self.treeview_frame, columns=("c1", "c2", "c3", "c4", "c5","c6", "c7", "c8", "c9","c10","c11"),show= 'headings',selectmode="browse")
        self.treeview.column("# 1",anchor='center',width=100)
        self.treeview.heading("# 1", text= "index")
        self.treeview.column("# 2",anchor='center',width=100)
        self.treeview.heading("# 2", text= "mz")
        self.treeview.column("# 3", anchor= 'center',width=100)
        self.treeview.heading("# 3", text= "EIC_id")
        self.treeview.column("# 4", anchor= 'center',width=100)
        self.treeview.heading("# 4", text="RT_cent")
        self.treeview.column("# 5", anchor= 'center',width=100)
        self.treeview.heading("# 5", text="RT_left")
        self.treeview.column("# 6", anchor= 'center',width=100)
        self.treeview.heading("# 6", text="RT_right")
        self.treeview.column("# 7", anchor= 'center',width=200)
        self.treeview.heading("# 7", text="prominences")
        self.treeview.column("# 8", anchor= 'center',width=100)
        self.treeview.heading("# 8", text="GF")
        self.treeview.column("# 9", anchor= 'center',width=100)
        self.treeview.heading("# 9", text="area_rmBl")
        self.treeview.column("# 10", anchor= 'center',width=100)
        self.treeview.heading("# 10", text="area")
        self.treeview.column("# 11", anchor= 'center',width=100)
        self.treeview.heading("# 11", text="SN")
        self.treeview.grid(row=0, column=0, sticky="nsew")

        self.treeview.bind("<Button-3>", self.click)

        self.treeview_scrollbar = ttk.Scrollbar(self.treeview_frame, orient="vertical", command=self.treeview.yview)
        self.treeview.configure(yscrollcommand=self.treeview_scrollbar.set)
        self.treeview_scrollbar.grid(row=0, column=1, sticky="ns")

        
        self.peak_temp_df=self.peak_df.round({'mz': 4, 'RT_cent': 2, 'RT_left': 2,'RT_right': 2,"prominences": 2, "area_removebaseline": 2, "area" : 2,"goodness_fitting": 2, "SN":2 })
        for index, row in self.peak_temp_df.iterrows():
            datas = row.tolist()
            datas[0]=int(datas[0])
            datas[2]=int(datas[2])
            self.treeview.insert('', 'end', text='', values=datas)

    

        # Initialize the right-click menu
        self.popup_menu = tk.Menu(self.treeview, tearoff=0)
        self.popup_menu.add_command(label="Plot EIC", command=self.plot_eic)
        
    def perform_search(self):
        try:
            mz = float(self.mz_entry.get()) 
        except:
            mz=0
        try:
            sn = float(self.sn_entry.get()) 
        except:
            sn=0
        try:
            goodness = float(self.goodness_entry.get())
        except:
            goodness=0
        if mz!=0 :
            diff = abs(self.peak_df['mz'] - mz)
            self.peak_temp_df=self.peak_df[diff == diff.min()]
            if sn!=0:
                self.peak_temp_df=self.peak_temp_df.loc[self.peak_temp_df.SN >= sn]
            if goodness!=0:
                self.peak_temp_df=self.peak_temp_df.loc[self.peak_temp_df["goodness_fitting"] >= goodness]
        else:
            self.peak_temp_df=self.peak_df
            if sn!=0:
                self.peak_temp_df=self.peak_temp_df.loc[self.peak_temp_df.SN >= sn]
            if goodness!=0:
                self.peak_temp_df=self.peak_temp_df.loc[self.peak_temp_df["goodness_fitting"] >= goodness]
        self.peak_temp_df=self.peak_temp_df.round({'mz': 4, 'RT_cent': 2, 'RT_left': 2,'RT_right': 2,"prominences": 2, "area": 2,"goodness_fitting": 2, "SN":2 })
        self.peak_temp_df["index"].astype(int)
        self.peak_temp_df["EIC_id"].astype(int)

        self.refresh_treeview()

    def refresh_treeview(self):

        for item in self.treeview.get_children():
            self.treeview.delete(item)

        for index, row in self.peak_temp_df.iterrows():
            datas = row.tolist()
            datas[0]=int(datas[0])
            datas[2]=int(datas[2])
            self.treeview.insert('', 'end', text='', values=datas)

    def click(self, event):
        self.popup_menu.post(event.x_root, event.y_root)
        
    def plot_eic(self):
        selected_item = self.treeview.selection()[0]

        selected_values = self.treeview.item(selected_item, "values")

        EIC_id=int(selected_values[2])
        index = int(selected_values[0])
        eic=result_dict["alldata"][EIC_id]["intensity_baseline"]
        x=np.array(result_dict["ms"]["rt"])
        left_index=int(result_dict["feature"][index][4])
        right_index=int(result_dict["feature"][index][5])
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_title(f"EIC mz="+str(float(selected_values[1])))
        ax.set_xlabel("Retention Time")
        ax.set_ylabel("Intensity")
        ax.plot(x,eic)
        ax.fill_between(x,0,eic,where=(x[left_index]<=x)&(x<=x[right_index]),facecolor="green",alpha=0.5)
        fig.show()

    def save_peak_table(self):

        file_path = filedialog.asksaveasfilename(defaultextension="peak table.csv", filetypes=[("Excel files", "*.csv")])
        if file_path:
            self.peak_temp_df.to_csv(file_path, index=False)
            messagebox.showinfo("Info", "File saved successfully!")
        else:
            return

if __name__ == "__main__":

    root = tk.Tk()

    my_gui = MyGUI(root)

    root.mainloop()
