from scipy.ndimage import gaussian_filter
import numpy as np
import math
from .baseline_remove import AirPLS,optimize_lambda
from scipy.signal import find_peaks,peak_prominences,peak_widths
from .guassan import goodness_fitting__
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib
import pandas as pd 
from scipy.optimize import curve_fit
from scipy.ndimage import uniform_filter1d
np.seterr(divide='ignore',invalid='ignore')

def Func_track_mass_tolist(trace_mass):
    lis = []
    for i in range(len(trace_mass["tracks"])):
        dic = {}
        dic["id_number"] = i
        dic["mz"] = trace_mass["tracks"][i][0]
        dic["rt_scan_numbers"] = trace_mass["rt_numbers"]
        dic["intensity"] = trace_mass["tracks"][i][1]
        dic["mz_list"]= trace_mass["tracks"][i][2]
        lis.append(dic)
    return lis

def np_move_avg(a,n):
    a_=np.zeros(((n-1)))
    a_=np.insert(a_,int((n-1)/2),a)
    b=np.zeros(len(a))
    for i in range(len(a)):
        v=a_[i:i+n]
        if len(np.where(v==0)[0])<=int((n-1)/2):
            b[i]=v[int((n-1)/2)]
    return b


def stats_detect_elution_peaks___(
    mass_track,
    min_peak_height,
    index,
    goodness_fitting,
    peak_width,
    s_n,
):

    
    def peak_check(cent,feature_list):
        check=True
        for i in feature_list:
            if cent>=i[1] and cent<=i[2]:
                check=False
                break
        return check
        
    def check_overlap(my_list):
        for i in range(len(my_list)):
            current_interval = my_list[i]
            
            for j in range(i + 1, len(my_list)):
                other_interval = my_list[j]

                if current_interval[2] >= other_interval[1] and current_interval[1] <= other_interval[2]:
                    overlap_start = max(current_interval[1], other_interval[1])
                    overlap_end = min(current_interval[2], other_interval[2])
                    
                    if current_interval[1] < other_interval[1]:
                        if current_interval[0]!=overlap_start :
                            current_interval[2] = overlap_start
                        else:
                            current_interval[2] = current_interval[0]+1
                    else:
                        if current_interval[0]!=overlap_end :
                            current_interval[1] = overlap_end 
                        else:
                            current_interval[1] = overlap_end-1
        return my_list
    
    
    def asymmetric_gaussian(x,  sigma_r, sigma_p):
        return np.piecewise(x, [x <= miu, x > miu],
                            [lambda x:a * np.exp(-(x - miu)**2 / (2 * sigma_r**2)),
                            lambda x:a * np.exp(-(x - miu)**2 / (2 * sigma_p**2))])
    
    def calculate_snr(signal, peaks, noise_min):
        signal_=np.append([],signal)
        for peak in peaks:
            signal_[peak[1]:peak[2]]=0
        for peak in peaks:
            noise_std_list=[]
            peak_top, peak_start, peak_end = peak[0:3]
            peak_width = peak_end - peak_start
            for i in range(6*peak_width):
                left_window_start = max(0, peak_start - 2 * peak_width - i)
                left_window_end = peak_start
                right_window_start = peak_end
                right_window_end = min(len(signal), peak_end + 2 * peak_width + i)
                
                left_noise = signal_[left_window_start: left_window_end].tolist()
                right_noise = signal_[right_window_start: right_window_end].tolist()
                noise=left_noise+right_noise
            
                noise_std_list.append(np.std(noise))
            peak.append(signal[peak_top]/max(min(noise_std_list),noise_min))

        return peaks
    
    def detect_peak_edges(signal):
        """
        Detect blank signals at both ends of the peak signal

        Parameters:
        signal: The peak signal is a list or array of intensity values

        Return value:
        start_index: Start index of the peak signal
        end_index: indicates the end index of the peak signal
        """
        start_index = 0
        end_index = len(signal) - 1
        
        for i, intensity in enumerate(signal):
            if intensity != 0:
                start_index = max(0, i - 1)
                break
        for i in range(len(signal) - 1, -1, -1):
            if signal[i] != 0:
                end_index = min(len(signal) - 1, i + 1)
                break
        
        return start_index, end_index

    def interpolate_signal(signal, target_length=100):
        """
        Interpolates the input signal into a signal of a specified length

        Parameters:
        signal: The raw signal to be interpolated, as a list or array of intensity values
        target_length: indicates the target length

        Return value:
        interpolated_signal: The interpolated signal is an array of length target_length
        """
        original_length = len(signal)
        
        original_indices = np.linspace(0, original_length - 1, original_length)
        
        target_indices = np.linspace(0, original_length - 1, target_length)
        
        interpolated_signal = np.interp(target_indices, original_indices, signal)
        
        return interpolated_signal
    
    def closest(mylist, Number):
        answer = []
        for i in mylist:
            answer.append(abs(Number-i))
        return answer.index(min(answer))
    
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
    
    def audit_mass_track(list_intensity,  min_peak_height, peak_width):
        
        LOW = min_peak_height/10
        noise_level = LOW  
        max_intensity, median_intensity = list_intensity.max(), np.median(list_intensity)
        if median_intensity > LOW:
            bottom_x_perc = list_intensity[ list_intensity < LOW + np.quantile(list_intensity, 0.25)]
            noise_level = bottom_x_perc.std()
        
        noise_level = max(noise_level, LOW)
        if peak_width>5:
            list_intensity=gaussian_filter(list_intensity, sigma=2)
        else:
            list_intensity = smooth_moving_average(list_intensity, size=peak_width)
        return noise_level, list_intensity
    
    ##------------------------------------------------------
    if peak_width<=6:
        a_ = np_move_avg(
            mass_track,
            2*peak_width-3
        )
    else:
        a_ = np_move_avg(
            mass_track,
            9
        )
    y__=AirPLS(a_, optimize_lambda(a_))
    noise_min, y  = audit_mass_track(y__,min_peak_height,peak_width)
    feature=[]
    initial_guess = [0.5, 1.5]
    check1=True
    peaks, _ = find_peaks(y,height=min_peak_height,width=peak_width,prominence=min_peak_height/3)
    # if len(peaks)<=peak_num:
    peak_num=len(peaks)
    if peak_num>0:
        prominences = peak_prominences(y, peaks)[0]
        prominences_sort=np.argsort(-prominences)
        prominences=prominences[prominences_sort]
        cent=peaks[prominences_sort]
        rel_heights=np.linspace(1,0.3,8)
        for peak in range(peak_num):
            if  feature!=[]:
                # print("**",cent)
                check1=peak_check(cent[peak],feature)
            left_check=True
            right_check=True
            if check1:
                for rel_height in rel_heights:
                    results_full = peak_widths(y, peaks, rel_height=rel_height)
                    if  left_check :
                        left_index=results_full[2][prominences_sort].astype(int)
                        r=left_index[peak]
                    if right_check :
                        right_index=(results_full[3][prominences_sort]+1.5).astype(int)
                        p=right_index[peak]
                    if p-r>=peak_width:
                        miu=cent[peak]-r
                        y_=y[r:p]
                        a=y_[miu]
                        x_data=np.linspace(0, len(y_) - 1, len(y_))
                        try:
                            optimized_params, _ = curve_fit(asymmetric_gaussian, x_data, y_, p0=initial_guess,maxfev = 10000)
                            optimized_params=[abs(i) for i in optimized_params]
                        except:
                            break
                        y_fit = asymmetric_gaussian(x_data, *optimized_params)
                        if cent[peak]-math.ceil(3*optimized_params[0])>r:
                            goodness_r=goodness_fitting__(y_[miu-math.ceil(3*optimized_params[0]):miu],y_fit[miu-math.ceil(3*optimized_params[0]):miu])
                            r=cent[peak]-math.ceil(3*optimized_params[0])
                        else:
                            goodness_r=goodness_fitting__(y_[0:miu],y_fit[0:miu])
                        if cent[peak]+math.ceil(3*optimized_params[1])+1<=p:
                            goodness_p=goodness_fitting__(y_[miu:miu+math.ceil(3*optimized_params[1])],y_fit[miu:miu+math.ceil(3*optimized_params[1])])
                            p=cent[peak]+math.ceil(3*optimized_params[1])+1
                        else:
                            goodness_p=goodness_fitting__(y_[miu:],y_fit[miu:])

                        if goodness_r>=0.75 and left_check:
                            left_check=False
                        if goodness_p>=0.75 and right_check:
                            right_check=False
                            if p>len(y)-1:
                                p=len(y)-1
                        if left_check==False and right_check==False:
                            Peak=y__[r:p]
                            start, end = detect_peak_edges(Peak)
                            p=r+end+1
                            r=r+start
                            Peak=y__[r:p]
                            if end-start+1>=peak_width:
                                miu=cent[peak]-r
                                a=Peak[miu]
                                interpolated_signal = interpolate_signal(Peak)
                                x_data=np.linspace(0, len(interpolated_signal) - 1, len(interpolated_signal))
                                miu=closest(interpolated_signal,a)
                                try:
                                    optimized_params, _ = curve_fit(asymmetric_gaussian, x_data, interpolated_signal, p0=initial_guess, maxfev = 10000)
                                    optimized_params=[abs(i) for i in optimized_params]
                                except:
                                    optimized_params=[0.01,0.01]
                                y_fit_ = asymmetric_gaussian(x_data, *optimized_params)
                                goodness=goodness_fitting__(interpolated_signal,y_fit_)
                                if goodness >= goodness_fitting:
                                    feature.append([cent[peak],r,p,y__[cent[peak]],goodness,sum(Peak),sum(mass_track[r:p])]) ##(cent,r,p,height,goodness,area_removebaseline,area,optimized_params)
                                break
                            else:
                                break
                    else:
                        break


    Feature=[]
    if len(feature)>1:
        feature=check_overlap(feature)
    feature=calculate_snr(y__,feature,noise_min) ###(cent,r,p,height,goodness,area,S/N)
    Feature=[i for i in feature if i[-1]>=s_n and i[-5]>=goodness_fitting]
    
    return {"index": index,
            "intensity_baseline":y__,
            "intensity":mass_track,
            "intensity_baseline_smooth":y,
            "feature": Feature,
            }


def Peaks(track_mass,
        min_peak_height,
        goodness_fitting=0.5,
        peak_width=5,
        s_n=3
        ):
    
    EIC_ = Func_track_mass_tolist(trace_mass=track_mass)
    min_peak_height_list = [min_peak_height for _ in range(len(EIC_))]
    goodness_fitting__list=[ goodness_fitting for _ in range(len(EIC_))]
    peak_width_list=[peak_width for _ in range(len(EIC_))]
    s_n_list=[s_n for _ in range(len(EIC_))]
    with tqdm_joblib(desc="Peak detection:", total=len(EIC_)) as progress_bar:
        alldata=Parallel(n_jobs=-1)(delayed(stats_detect_elution_peaks___)(i["intensity"], g, n, a, b, c)
        for i, g, n, a, b, c in zip(EIC_, min_peak_height_list,range(len(EIC_)),goodness_fitting__list,peak_width_list,s_n_list)
        )
    Feature=[]
    for i in range(len(alldata)):
        if alldata[i]["feature"]:
            for ii in alldata[i]["feature"]:
                
                Feature.append([EIC_[alldata[i]["index"]]["mz_list"][ii[0]],alldata[i]["index"]]+ii)
    return alldata,Feature