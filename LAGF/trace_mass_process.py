from tkinter import filedialog
import os
from tqdm import tqdm
import numpy as np 
import pandas as pd
import pymzml
from scipy.interpolate import interp1d
from joblib import Parallel, delayed
from scipy.ndimage import uniform_filter1d
from scipy.signal import savgol_filter

class MS2_isocheck:
    def __init__(self,pepmass,pep_iso,mz,i,rt,ID,charge,iso_id) :
        self.pepmass=pepmass
        self.pep_iso=pep_iso
        self.mz=mz
        self.i=i
        self.rt=rt
        self.ID=ID
        self.char=charge
        self.iso_id=iso_id


def read_mzmlfile():
    mzml_file = filedialog.askopenfilename(
    initialdir=(os.getcwd()),
    filetypes=[("mzML file", ".mzML .mzml")],
    title=("Select mzml file"),
    )
    mzml_file = os.path.abspath(mzml_file)
    ##获取文件路径
    mzml_dirname, mzml_filename = os.path.split(os.path.abspath(mzml_file))
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
    for i in ms :
        if i.ms_level == 1: 
            ms1.append(i)
            ms1_i.append(i.i)
            ms1_id.append(i.ID)
            ms1_mz.append(i.mz)
            ms1_ii.append(num)
            ms1_rt.append(i.scan_time[0])
            num+=1
        else:
            ms2.append(i)
    ms1_={"id":ms1_id,"intensity":ms1_i,"mz":ms1_mz,"ii":ms1_ii,"rt":ms1_rt}
    
    return ms1,ms2,ms1_,mzml_file

def np_move_avg(a,n):
    a_=np.zeros(((n-1)))
    a_=np.insert(a_,int((n-1)/2),a)
    b=np.zeros(len(a))
    for i in range(len(a)):
        v=a_[i:i+n]
        if len(np.where(v==0)[0])<=int((n-1)/2):
            b[i]=v[int((n-1)/2)]
    return b

def smooth_moving_average(list_intensity, size=9):
    '''
    Smooth data of a noisy mass track using simple moving average.

    Parameters
    ----------
    list_intensity : list[int]
        list of intensity values from a mass track.
    size : int, optional, default: 9
        window size for moving average.

    Returns
    -------
    New list of smoothed intensity values.

    
    Note
    ----
        For very noise data, one may use smooth_lowess.
    '''
    return uniform_filter1d(list_intensity, size, mode='nearest')

def smooth_moving_SG(list_intensity, size=15,k=2):
    return savgol_filter(list_intensity,size,k)

def func_interp(lis,trace_mass):
    trace_mass_new=np.zeros_like(trace_mass)
    if len(lis)==1 and lis[0]>1 and lis[0]+2<=len(trace_mass):
        trace_mass_new[lis[0]]=(trace_mass[lis[0]-1]+trace_mass[lis[0]+1])/2
    elif len(lis)>1 and lis[0]>1 and lis[-1]+2<=len(trace_mass):
        x=[lis[0]-1,lis[-1]+1]
        xnew=[lis[0]-1]+lis+[lis[-1]+1]
        y=[trace_mass[lis[0]-1],trace_mass[lis[-1]+1]]
        f=interp1d(x,y,"linear")
        ynew=f(xnew)
        trace_mass_new[lis[0]:lis[-1]+1]=ynew[1:-1]
    return trace_mass_new

def Trace_mass_fill(trace_mass,pepmass,index,fill_gap=2,soomth_wind=9):
    lis=[]
    num=0
    trace_mass_new=np.array([])
    trace_mass_set=[]
    trace_mass=np_move_avg(trace_mass,soomth_wind)
    while num+1<=len(trace_mass)-1:
        lis_=[]
        if trace_mass[num]==0:
            lis_.append(num)
            num+=1
            check=True
            while check==True and trace_mass[num]==0  :
                lis_.append(num)
                num+=1
                if num>len(trace_mass)-1:
                    check=False
        else:
            num+=1
            
        if len(lis_)>0 and len(lis_)<=fill_gap:
            lis.append(lis_)
    trace_mass_new=np.append(trace_mass,trace_mass_new)
    for i in range(len(lis)):
        trace_mass_set.append(trace_mass)
    alldata_=Parallel(n_jobs=-1)(
    delayed(func_interp)(
        lis_,trace_mass_
        )
        for lis_,trace_mass_ in zip(
            lis,
            trace_mass_set
            
            )
    ) 
    del trace_mass_set,lis
    for i in alldata_:
        trace_mass_new+=i
    # trace_mass_new_mean=np_move_avg(trace_mass_new,soomth_wind)
    # trace_mass_new_mean=smooth_moving_SG(trace_mass_new,size=soomth_wind)
    return (pepmass,trace_mass_new,index)


def feature_detection(ms_list,feature_list): 
    ###return ms2 grouping by feature ions
    ms2_group=[]
    for spec in ms_list:
        find_result=[]
        for i in feature_list :
            peak_to_find=spec.has_peak(i)
            if len(peak_to_find)==0:
                find_result.append("False")
            else :
                find_result.append("TURE")
        if "TURE" in find_result:
            ms2_group.append(spec)
        else:
            continue
    return ms2_group

def Found_iso(ms2,ms1,ms1_id,rt_wind=11,max_charge=3,SN=True,drop_point=2):
    ms1_id_list=ms1_id
    pep_id=np.searchsorted(ms1_id_list,int(ms2.selected_precursors[0]["precursor id"]))
    pepmass=float(ms2.selected_precursors[0]["mz"])
    diff=np.absolute(ms1[pep_id].mz-pepmass)
    index = diff.argmin()
    pepmass=ms1[pep_id].mz[index]
    iso_set=[]
    iso_id=[]
    iso_num=0
    pep_mz=[]
    pep_i=[]
    
    if SN :
        for mz_,int_ in ms2.highest_peaks(50):
            pep_mz.append(mz_)
            pep_i.append(int_)
        d=np.argsort(pep_mz)
        pep_mz=np.array(pep_mz)[d]
        pep_i=np.array(pep_i)[d]
    else:
        pep_mz=ms2.mz    
        pep_i=ms2.i
    
    if "charge" in ms2.selected_precursors[0].keys():
        charge=ms2.selected_precursors[0]["charge"]
        iso_check=True
        pep_id_=pep_id
        pepmass_=pepmass
        pepmass_i=ms1[pep_id].i[index]
        while iso_check:
            p=[1]
            r=[1]
            iso=[]
            iso_i=[]
            iso_id_=[]
            iso_mz=[]
            pepmass_mass_lis=[pepmass_]
            pepmass_ID_lis=[pep_id_]
            pepmass_i_lis=[pepmass_i]
            num=1
            p_check=True
            r_check=True
            ms2_i=np.where(ms1[pep_id_].mz<=(pepmass_-1.003/charge)*(1+5e-6))[0]
            ms2_o=np.where(ms1[pep_id_].mz>=(pepmass_-1.003/charge)*(1-5e-6))[0]
            ms2_in=np.intersect1d(ms2_i,ms2_o)
            if len(ms2_in)>=1:
                iso.append(1)
                iso_i.append(np.max(ms1[pep_id_].i[ms2_in]))
                iso_id_.append(pep_id_)
                iso_mz.append(ms1[pep_id_].mz[ms2_in[np.argmax(ms1[pep_id_].i[ms2_in])]])
            while p_check:
                if pep_id_+num<=len(ms1)-1:
                    ms2_i=np.where(ms1[pep_id_+num].mz<=(pepmass_)*(1+5e-6))[0]
                    ms2_o=np.where(ms1[pep_id_+num].mz>=(pepmass_)*(1-5e-6))[0]
                    ms2_in=np.intersect1d(ms2_i,ms2_o)
                    if len(ms2_in)>=1:
                        p.append(1)
                        pepmass_ID_lis.append(pep_id_+num)
                        pepmass_mass_lis.append(ms1[pep_id_+num].mz[ms2_in[np.argmax(ms1[pep_id_+num].i[ms2_in])]])
                        pepmass_i_lis.append(np.max(ms1[pep_id_+num].i[ms2_in]))
                    else:
                        p.append(0)
                    ms2_i=np.where(ms1[pep_id_+num].mz<=(pepmass_-1.003/charge)*(1+5e-6))[0]
                    ms2_o=np.where(ms1[pep_id_+num].mz>=(pepmass_-1.003/charge)*(1-5e-6))[0]
                    ms2_in=np.intersect1d(ms2_i,ms2_o)
                    if len(ms2_in)>=1:
                        iso.append(1)
                        iso_i.append(np.max(ms1[pep_id_+num].i[ms2_in]))
                        iso_id_.append(pep_id_+num)
                        iso_mz.append(ms1[pep_id_+num].mz[ms2_in[np.argmax(ms1[pep_id_+num].i[ms2_in])]])
                    if (1 not in p[-drop_point:] or
                        p.count(1)-1==(rt_wind-1)/2):
                        p_check=False
                    num+=1
                else: p_check=False
            num=1
            while r_check:
                if pep_id_-num>=0:
                    ms2_i=np.where(ms1[pep_id_-num].mz<=(pepmass_)*(1+5e-6))[0]
                    ms2_o=np.where(ms1[pep_id_-num].mz>=(pepmass_)*(1-5e-6))[0]
                    ms2_in=np.intersect1d(ms2_i,ms2_o)
                    if len(ms2_in)>=1:
                        r.append(1)
                        pepmass_ID_lis.insert(0,pep_id_-num)
                        pepmass_mass_lis.insert(0,ms1[pep_id_-num].mz[ms2_in[np.argmax(ms1[pep_id_-num].i[ms2_in])]])
                        pepmass_i_lis.insert(0,np.max(ms1[pep_id_-num].i[ms2_in]))
                    else:
                        r.append(0)
                    ms2_i=np.where(ms1[pep_id_-num].mz<=(pepmass_-1.003/charge)*(1+5e-6))[0]
                    ms2_o=np.where(ms1[pep_id_-num].mz>=(pepmass_-1.003/charge)*(1-5e-6))[0]
                    ms2_in=np.intersect1d(ms2_i,ms2_o)
                    if len(ms2_in)>=1:
                        iso.insert(0,1)
                        iso_i.insert(0,np.max(ms1[pep_id_-num].i[ms2_in]))
                        iso_id_.insert(0,pep_id_-num)
                        iso_mz.insert(0,ms1[pep_id_-num].mz[ms2_in[np.argmax(ms1[pep_id_-num].i[ms2_in])]])
                    if (1 not in r[-drop_point:] or
                        r.count(1)-1==(rt_wind-1)/2):
                        r_check=False
                    num+=1
                else: r_check=False
            
            if len(iso)/rt_wind>=0.7:
                ind=iso_i.index(max(iso_i))
                iso_set.append(iso_mz[ind])
                iso_id.append(ms1_id_list[iso_id_[ind]])
                pep_id_=iso_id_[ind]
                pepmass_i=max(iso_i)
                pepmass_=iso_mz[ind]
                iso_num+=1
            else:
                iso_check=False
            
    else:
        for n in np.linspace(max_charge,1,max_charge):
            charge=int(n)
            iso_check=True
            pep_id_=pep_id
            pepmass_=pepmass
            pepmass_i=ms1[pep_id].i[index]
            while iso_check:
                p=[1]
                r=[1]
                iso=[]
                iso_i=[]
                iso_id_=[]
                iso_mz=[]
                pepmass_mass_lis=[pepmass_]
                pepmass_ID_lis=[pep_id_]
                pepmass_i_lis=[pepmass_i]
                num=1
                p_check=True
                r_check=True
                ms2_i=np.where(ms1[pep_id_].mz<=(pepmass_-1.003/charge)*(1+5e-6))[0]
                ms2_o=np.where(ms1[pep_id_].mz>=(pepmass_-1.003/charge)*(1-5e-6))[0]
                ms2_in=np.intersect1d(ms2_i,ms2_o)
                if len(ms2_in)>=1:
                    iso.append(1)
                    iso_i.append(np.max(ms1[pep_id_].i[ms2_in]))
                    iso_id_.append(pep_id_)
                    iso_mz.append(ms1[pep_id_].mz[ms2_in[np.argmax(ms1[pep_id_].i[ms2_in])]])
                while p_check:
                    if pep_id_+num<=len(ms1)-1:
                        ms2_i=np.where(ms1[pep_id_+num].mz<=(pepmass_)*(1+5e-6))[0]
                        ms2_o=np.where(ms1[pep_id_+num].mz>=(pepmass_)*(1-5e-6))[0]
                        ms2_in=np.intersect1d(ms2_i,ms2_o)
                        if len(ms2_in)>=1:
                            p.append(1)
                            pepmass_ID_lis.append(pep_id_+num)
                            pepmass_mass_lis.append(ms1[pep_id_+num].mz[ms2_in[np.argmax(ms1[pep_id_+num].i[ms2_in])]])
                            pepmass_i_lis.append(np.max(ms1[pep_id_+num].i[ms2_in]))
                        else:
                            p.append(0)
                        ms2_i=np.where(ms1[pep_id_+num].mz<=(pepmass_-1.003/charge)*(1+5e-6))[0]
                        ms2_o=np.where(ms1[pep_id_+num].mz>=(pepmass_-1.003/charge)*(1-5e-6))[0]
                        ms2_in=np.intersect1d(ms2_i,ms2_o)
                        if len(ms2_in)>=1:
                            iso.append(1)
                            iso_i.append(np.max(ms1[pep_id_+num].i[ms2_in]))
                            iso_id_.append(pep_id_+num)
                            iso_mz.append(ms1[pep_id_+num].mz[ms2_in[np.argmax(ms1[pep_id_+num].i[ms2_in])]])
                        if (1 not in p[-drop_point:] or
                            p.count(1)-1==(rt_wind-1)/2):
                            p_check=False
                        num+=1
                    else: p_check=False
                num=1
                while r_check:
                    if pep_id_-num>=0:
                        ms2_i=np.where(ms1[pep_id_-num].mz<=(pepmass_)*(1+5e-6))[0]
                        ms2_o=np.where(ms1[pep_id_-num].mz>=(pepmass_)*(1-5e-6))[0]
                        ms2_in=np.intersect1d(ms2_i,ms2_o)
                        if len(ms2_in)>=1:
                            r.append(1)
                            pepmass_ID_lis.insert(0,pep_id_-num)
                            pepmass_mass_lis.insert(0,ms1[pep_id_-num].mz[ms2_in[np.argmax(ms1[pep_id_-num].i[ms2_in])]])
                            pepmass_i_lis.insert(0,np.max(ms1[pep_id_-num].i[ms2_in]))
                        else:
                            r.append(0)
                        ms2_i=np.where(ms1[pep_id_-num].mz<=(pepmass_-1.003/charge)*(1+5e-6))[0]
                        ms2_o=np.where(ms1[pep_id_-num].mz>=(pepmass_-1.003/charge)*(1-5e-6))[0]
                        ms2_in=np.intersect1d(ms2_i,ms2_o)
                        if len(ms2_in)>=1:
                            iso.insert(0,1)
                            iso_i.insert(0,np.max(ms1[pep_id_-num].i[ms2_in]))
                            iso_id_.insert(0,pep_id_-num)
                            iso_mz.insert(0,ms1[pep_id_-num].mz[ms2_in[np.argmax(ms1[pep_id_-num].i[ms2_in])]])
                        if (1 not in r[-drop_point:] or
                            r.count(1)-1==(rt_wind-1)/2):
                            r_check=False
                        num+=1
                    else: r_check=False
                
                if len(iso)/rt_wind>=0.7:
                    ind=iso_i.index(max(iso_i))
                    iso_set.append(iso_mz[ind])
                    iso_id.append(ms1_id_list[iso_id_[ind]])
                    pep_id_=iso_id_[ind]
                    pepmass_i=max(iso_i)
                    pepmass_=iso_mz[ind]
                    iso_num+=1
                else:
                    iso_check=False
            if iso_set!=[]:
                break
        if iso_set==[]:
            charge="no"

    return MS2_isocheck(float(ms2.selected_precursors[0]["mz"]),iso_set,pep_mz,pep_i,ms2.scan_time[0],int(ms2.selected_precursors[0]["precursor id"]),charge,iso_id)