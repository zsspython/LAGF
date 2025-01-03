import numpy as np
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib

def extract_EIC_LAGF(ms1_, traget_check=False,traget_array=[],
                        mz_tolerance=0.01, min_intensity=1000, min_timepoints=5, 
                        min_peak_height=1000):

    def mz_list_check(mz,mz_array,mz_tolerance_ppm=10):
        mz_check=False
        if min(abs(mz_array-mz))/mz<=mz_tolerance_ppm*0.000001:
            mz_check=True
        return mz_check
    
    def func_alldata_set_traget(ii,mz,intens,traget_array,min_intensity):
        alldata = [[] for _ in range(10000)]
        intensities = intens
        good_positions = intensities > min_intensity
        intensities = intensities[good_positions]
        mzs = mz[good_positions]
        for mz, inten in zip(mzs, intensities):
            if mz_list_check(mz,traget_array):
                index=int(mz)
                alldata[index].append((mz,ii,inten))
        dtype = np.dtype([('mz', float), ('rt', int),('intensity', float)])
        alldata=[np.array(sorted(i,key=lambda x:(-x[2],x[1])),dtype=dtype) for i in alldata]
        return alldata
    
    def __check_two_mz__(mz1,mz2,mz_tolerance=mz_tolerance):
        check=False
        if abs(mz1-mz2)<=mz_tolerance:
            check=True
        return check
    
    traget_array_=[]
    min_intensity_=[]
    for i in range(len(ms1_["id"])):
        min_intensity_.append(min_intensity)
    if traget_check:
        for i in range(len(ms1_["id"])):
            traget_array_.append(traget_array)
        alldata_=Parallel(n_jobs=-1)(
        delayed(func_alldata_set_traget)(
            ii,mz,i,traget_array,min_intensity
        )
        for ii,mz,i,traget_array,min_intensity in zip(
            ms1_["ii"],
            ms1_["mz"],
            ms1_["intensity"],
            traget_array_,
            min_intensity_
            )
        )
        alldata=[]
        for i in alldata_:
            if i!=[]:
                for ii in i:
                    alldata.append(ii)
        del alldata_
    
    else:
        alldata = [[] for _ in range(10000)]
        for ii in ms1_["ii"]:
            intensities = ms1_["intensity"][ii]
            good_positions = intensities > min_intensity
            intensities = intensities[good_positions]
            mzs =ms1_["mz"][ii][good_positions]
            #alldata += [{'mz': mz,'rt': ii, 'intensity':inten} for mz, inten in zip(mzs, intensities)]
            for mz, inten in zip(mzs, intensities):
                index=int(mz)
                alldata[index].append((mz,ii,inten))
        dtype = np.dtype([('mz', float), ('rt', int),('intensity', float)])
        alldata=[np.array(sorted(i,key=lambda x:(-x[2],x[1])),dtype=dtype) for i in alldata]
    
    def build_eics_numpy(data_points, epsilon_mz=mz_tolerance/2,ms1_len=len(ms1_["intensity"]),min_peak_height=min_peak_height,min_timepoints=min_timepoints):
        
        def __rough_check_consecutive_scans__(data_point_rt, gap_allowed=2, min_timepoints=min_timepoints):
            _checked = True
            check_max_len = 4 * min_timepoints                 
            if len(data_point_rt) < check_max_len:
                min_check_val = gap_allowed + min_timepoints -1 
                rts =data_point_rt
                if len(rts)>=min_timepoints:
                    steps = [rts[ii]-rts[ii-min_timepoints+1] for ii in range(min_timepoints-1, len(rts))]
                    if min(steps) > min_check_val:
                        _checked = False
                else:
                    _checked = False
            return _checked
        
        def __check_min_peak_height__(data_point_i, min_peak_height):
            return max(data_point_i) >= min_peak_height
        
        
        def __check_mz_range_overlap__(mz_range,mz,epsilon_mz=epsilon_mz):
            result=(mz-epsilon_mz,mz+epsilon_mz,mz)
            if mz_range:
                for i in mz_range:
                    if mz>=i[2]+epsilon_mz and mz<=i[2]+epsilon_mz*2:
                        result[0]=i[1]
                        break
                    elif mz<i[2]-epsilon_mz and mz>=i[2]-epsilon_mz*2:
                        result[1]=i[0]
                        break
            return result
            
        eics = []
        mz_range=[]
        for data_point in data_points:
            mz = data_point['mz']
            rt = data_point["rt"]
            added_to_existing_eic = False
            for eic in eics:
                if (eic['mz_range'][0] - epsilon_mz <= mz < eic['mz_range'][1] + epsilon_mz):
                    if rt not in eic["rt"] :
                        eic['data_points'].append(data_point)
                        eic["rt"].append(rt)
                        eic["mz"].append(mz)
                        added_to_existing_eic = True
                        break
                    else: 
                        added_to_existing_eic = True
                        break

            if not added_to_existing_eic :
                a=__check_mz_range_overlap__(mz_range,mz,epsilon_mz=epsilon_mz)
                new_eic = {'mz_range': a, 'data_points': [data_point], "rt":[rt] ,"mz": [mz]}
                eics.append(new_eic)
                mz_range.append(new_eic['mz_range'])
                
        if eics!=[]:
            eic_new=[]
            for i in eics:
                i["data_points"].sort(key=lambda x:x["rt"])
                data_point_i=[x["intensity"] for x in i["data_points"]]
                data_point_rt=[x["rt"] for x in i["data_points"]]
                data_point_mz=[x["mz"] for x in i["data_points"]]
                if __rough_check_consecutive_scans__(data_point_rt) and __check_min_peak_height__(data_point_i,min_peak_height):
                    eic_i=np.zeros(ms1_len)
                    eic_mz=np.zeros(ms1_len)
                    eic_mz.fill(i["mz_range"][2])
                    for ii in range(len(data_point_rt)):
                        eic_i[data_point_rt[ii]]=data_point_i[ii]
                        eic_mz[data_point_rt[ii]]=data_point_mz[ii]
                    i["tracks"]=eic_i
                    i["mz_list"]=eic_mz
                    eic_new.append(i)
        else:
            eic_new=[]
        return eic_new
    
    with tqdm_joblib(desc="EIC Extraction:", total=len(alldata)) as progress_bar:
        EIC_=Parallel(n_jobs=-1)(
            delayed(build_eics_numpy)(
                data_point
            )
            for data_point in alldata)
    
    del alldata
    
    mz_range_index=[]
    for i,j in enumerate(EIC_):
        if j:
            mz_range=[x["mz_range"][2] for x in j]
            a=mz_range.index(min(mz_range))
            c=mz_range.index(max(mz_range))
            mz_range_index.append((i,a,c))


    merge_index=[]
    temp_data_points=[]
    EIC=[]
    for i,j in enumerate(mz_range_index[0:-1]):
        if mz_range_index[i+1][0] - mz_range_index[i][0] == 1:
            if __check_two_mz__(EIC_[mz_range_index[i+1][0]][mz_range_index[i+1][1]]["mz_range"][2], EIC_[mz_range_index[i][0]][mz_range_index[i][2]]["mz_range"][2]):
                temp_data_points=EIC_[mz_range_index[i+1][0]][mz_range_index[i+1][1]]["data_points"] + EIC_[mz_range_index[i][0]][mz_range_index[i][2]]["data_points"]
                merge_two_EIC=build_eics_numpy(temp_data_points)
                merge_index.append((mz_range_index[i+1][0],mz_range_index[i+1][1]))
                merge_index.append((mz_range_index[i][0],mz_range_index[i][2]))
                for x in merge_two_EIC:
                    EIC.append(x)
                
    
    for i in mz_range_index:
        for x in EIC_[i[0]]:
            if (i[0],EIC_[i[0]].index(x)) not in merge_index:
                EIC.append(x)
    del EIC_
    
    EIC.sort(key=lambda x:x["mz_range"][2])
    rt_numbers = list(range(len(ms1_["rt"])))
    updated_tracks=[(x["mz_range"][2], x["tracks"], x["mz_list"]) for x in EIC]
    mz_=[updated_tracks[0][0],updated_tracks[-1][0]]
    return {
        'rt_numbers': rt_numbers,
        'rt_times': ms1_["rt"],
        'tracks': updated_tracks,
        'mz_range': mz_
    }
