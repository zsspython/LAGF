from trace_mass_process import Trace_mass_fill
from scipy.signal import detrend
from auto_min_peakheight import estimate_min_peak_height
from guassan import Func_main_find_peak,func_gaosi,func_self_fit_gaussian,goodness_fitting__
import numpy as np
from joblib import Parallel, delayed

def audit_mass_track(list_intensity, 
                    min_intensity_threshold, 
                    min_peak_height ):

    scaling_factor, LOW, HIGH = 1, min_intensity_threshold, 1E8
    _baseline_, noise_level = LOW, LOW                     # will not change on a clean track
    max_intensity, median_intensity = list_intensity.max(), np.median(list_intensity)
    if max_intensity > HIGH:
        scaling_factor = max_intensity/HIGH
        list_intensity = list_intensity/scaling_factor

    if median_intensity > LOW: 
        LL = list_intensity[list_intensity > LOW]
        if len(LL) > len(list_intensity) * 0.5 and median_intensity > 10 * min_peak_height:
            list_intensity = detrend(list_intensity)        # detrend
        bottom_x_perc = list_intensity[list_intensity < LOW + np.quantile(list_intensity, 0.25)]
        _baseline_, noise_level = bottom_x_perc.mean(), bottom_x_perc.std()
    # if 100 * noise_level > max_intensity or max_intensity < 10 * min_peak_height:
    #     print(noise_level)
    #     list_intensity=Trace_mass_fill(list_intensity,1,1)[1]
    _baseline_, noise_level = max(_baseline_, LOW), max(noise_level, LOW)

    # decision on smoothing
    list_intensity = list_intensity - _baseline_

    return  noise_level,list_intensity

def extend_ROI(ROI, number_of_scans):
    left = [x for x in [ROI[0]-3, ROI[0]-2, ROI[0]-1] if x >=0]
    # todo - should we add *up to* 3 datapoints instead of 3 datapoints in the case that the difference is less than 3?
    right = [x for x in [ROI[-1]+1, ROI[-1]+2, ROI[-1]+3] if x < number_of_scans]
    return left + ROI + right


def stats_detect_elution_peaks(mass_track,  
                min_peak_height, min_fwhm=3, 
                min_intensity_threshold=1000,SN_threshold=3,goodness_threshold=0.5,min_sigma=1,cent_sigma=200,max_sigma=210,peak_length=7):
    
    def __S_N_check__(EIC,R,SN_threshold):
        """check the SN of ROI 

        Args:
            EIC (np.array): mass_track["intensity"]
            R (list): ROI's index in EIC
            SN_threshold (int): the threshold of SN

        Returns:
            SN_check: True or False
            SN 
        """
        SN_check=True
        y=EIC[R]
        S=np.std(y)
        noise_min=1000
        noise_list=[]
        for i in range(len(y)*8):
            if R[1]+len(y)+i<=len(EIC)-1:
                noise_wind_p=EIC[R[-1]+i:R[-1]+len(y)+i]
                std_dev = np.std(noise_wind_p)
                noise_list.append(std_dev)
            if R[0]-len(y)-i>=0:
                noise_wind_r=EIC[R[0]-len(y)-i:R[0]-i]
                std_dev = np.std(noise_wind_r)
                noise_list.append(std_dev)
        if noise_list!=[] and min(noise_list)>=noise_min:
            noise_min=min(noise_list)
        if S/noise_min < SN_threshold:
            SN_check=False
        return SN_check,S/noise_min
    
    def __func_goodness_check__(
        list_intensity,
        peak_list,
        EIC,
        min_peak_height,
        goodness_threshold,
        peak_length,
        min_sigma,
        cent_sigma,
        max_sigma,
        noise_min=1000,
        SN_threshold=10,
    ):
    
        result = []
        for i in peak_list:
            if i[1]==i[2]:
                i[1]+=1
            y=list_intensity[i[0] : i[1]]
            y_ = EIC[i[0] : i[1]]
            noise_list = []
            for n in range(len(y_) * 8):
                if i[1] + len(y_) + n <= len(list_intensity) - 1:
                    noise_wind_p = EIC[i[1] + n : i[1] + len(y_) + i[1]]
                    std_dev = np.std(noise_wind_p)
                    noise_list.append(std_dev)
                if i[0] - len(y_) - n >= 0:
                    noise_wind_r = EIC[i[0] - len(y_) - n : i[0] - n]
                    std_dev = np.std(noise_wind_r)
                    noise_list.append(std_dev)
            if noise_list != [] and min(noise_list) >= noise_min:
                noise_min = min(noise_list)
            S = max(y_)
            S_N = S / noise_min
            area = sum(y_)
            miu = i[2] - i[0]
            a=y[miu]
            re, x_r, x_p = func_self_fit_gaussian(
                y,
                miu,
                min_sigma,
                cent_sigma,
                max_sigma,
                goodness_fitting=goodness_threshold,
            )
            x = np.linspace(0, len(y_) - 1, len(y_))
            y1 = func_gaosi(x, miu, x_r, a)
            y2 = func_gaosi(x, miu, x_p, a)
            y_new =np .append(y1[0:miu],y2[miu:])
            p=goodness_fitting__(y, y_new)
            check=False
            if x_r>=min_sigma and x_r<=cent_sigma and x_p>=min_sigma and x_p<=cent_sigma:
                check=True
            if (
                p >= goodness_threshold
                and a >= min_peak_height
                and len(y_) >= peak_length
                and S_N >= SN_threshold
                and check
            ):
                
                i.append(area)
                i.append(S)
                i.append(p)
                i.append(S_N)
                result.append(i)
                    
        return result  ###[r,p,cent,area,max_i,p,SN]

    ##------------------------------------------------------
    list_scans = np.arange(len(mass_track["rt_scan_numbers"]))
    a = Trace_mass_fill(
        mass_track["intensity"],
        mass_track["mz"],
        mass_track["id_number"],
        soomth_wind=13,
    )[1]
    noise_level, list_intensity= audit_mass_track(
        a, min_intensity_threshold, min_peak_height
    )
    # # get ROIs by separation/filtering with noise_level, allowing 2 gap
    __selected_scans__ = list_scans[list_intensity > noise_level]
    ROIs_ = []
    peak_list = []
    if __selected_scans__.any():
        ROIs = []
        tmp = [__selected_scans__[0]]
        for ii in __selected_scans__[1:]:
            if ii - tmp[-1] < 3:
                tmp += range(tmp[-1] + 1, ii + 1)
            else:
                ROIs.append(tmp)
                tmp = [ii]

        ROIs.append(tmp)
        ROIs = [r for r in ROIs if len(r) >= min_fwhm + 2]
        if ROIs:
            lis = np.zeros(len(ROIs))
            for R in ROIs:
                i = ROIs.index(R)
                if len(R) < 3 * min_fwhm:  # extend if too short - help narrow peaks
                    R = extend_ROI(R, len(mass_track["rt_scan_numbers"]))
                list_intensity_roi = list_intensity[R]
                lis[i] = max(list_intensity_roi)

            lis_sort = np.argsort(-lis)
            ROIs_ = [ROIs[i] for i in lis_sort]

            # peak_detection
            max_int = []
            for R in ROIs_:
                SN_check, SN = __S_N_check__(mass_track["intensity"], R, SN_threshold)
                if SN_check:
                    peak_list_ = Func_main_find_peak(
                        list_intensity,
                        R,
                        min_peak_height,
                        min_sigma,
                        cent_sigma,
                        max_sigma,
                        goodness_threshold=0.80,
                    )
                    if peak_list_:
                        for peak in peak_list_:
                            if peak[1] == len(R):
                                peak[1] = len(R) - 1
                            peak_list.append([R[peak[0]], R[peak[1]], R[peak[2]]])
                            max_int.append(mass_track["intensity"][R[peak[2]]])

    if peak_list:
        if min_peak_height<max(max_int)/20:
            min_peak_height=max(max_int)/20
        peak_list = __func_goodness_check__(
            list_intensity,
            peak_list,
            mass_track["intensity"],
            min_peak_height,
            goodness_threshold,
            peak_length,
            min_sigma,
            cent_sigma,
            max_sigma,
        )

    mass_track["intensity_soomth"] = list_intensity
    mass_track["peak"] = peak_list
    return mass_track

