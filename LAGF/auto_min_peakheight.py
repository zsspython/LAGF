def find_mzdiff(list_mass, list_mz_diff=[1.003355, 21.9820], mz_tolerance_ppm=5):

    def build_centurion(list_peaks):
        d = {}
        for p in list_peaks:
            cent = int(100 * p['mz'])
            if cent in d:
                d[cent].append(p)
            else:
                d[cent] = [p]
        return d
    
    def find_best_match(query_mz, mz_centurion_tree, limit_ppm=2):
        q = int(query_mz * 100)
        mz_tol = query_mz * limit_ppm * 0.000001
        result = (None, 999)
        for ii in (q-1, q, q+1):
            L = mz_centurion_tree.get(ii, [])
            for peak in L:
                _d = abs(peak['mz']-query_mz)
                if _d < min(result[1], mz_tol):     # enforce mz_tol here
                    result = (peak, _d)
                    
        return result[0]
    
    pairs = []
    # list_mass_tracks has similar format as list_peaks.
    mztree = build_centurion(list_mass)
    for mzdiff in list_mz_diff:
        for x in list_mass:
            y = find_best_match(x['mz'] + mzdiff, mztree, mz_tolerance_ppm)
            if y:
                pairs.append((x['id_number'], y['id_number']))

    return pairs

def estimate_min_peak_height(infile, EIC_,
                        mz_tolerance_ppm=5,
                        ):
    
    def flatten_tuplelist(L):
        return list(set([x[0] for x in L] + [x[1] for x in L]))

    new = {'sample_id': infile, 'input_file': infile, 'ion_mode': '',}
    list_mass = []
    xdict=EIC_
    new['list_scan_numbers'] = xdict['rt_numbers']            # list of scans, starting from 0
    new['list_retention_time'] = xdict['rt_times']        # full RT time points in sample
    ii = 0
    # already in ascending order of m/z from extract_massTracks_, get_thousandth_regions
    for track in xdict['tracks']:                         
        list_mass.append( {
            'id_number': ii, 
            'mz': track[0],
            'intensity': track[1], 
            } )
        ii += 1

    new['list_mass_tracks'] = list_mass
    anchor_mz_pairs = find_mzdiff(list_mass, 
                            list_mz_diff = [1.003355,], mz_tolerance_ppm=mz_tolerance_ppm)
    _mz_landmarks_ = flatten_tuplelist(anchor_mz_pairs)
    # down scale list_mass_tracks to verified by _mz_landmarks_
    list_mass = [list_mass[ii] for ii in _mz_landmarks_]
    peak_heights = [x['intensity'].max() for x in list_mass]
    min_peak_height_ = min(peak_heights)
    return min_peak_height_


