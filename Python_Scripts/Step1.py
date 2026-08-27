import pandas as pd
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import os
import h5py
import glob
import numpy as np
import yaml
import argparse
from scipy.ndimage import uniform_filter1d
import resource

def _rss_gb():
    """Current process peak resident set size, in GB (Linux ru_maxrss is in KB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)

def thd_correct(array):
    indices = np.arange(0, 25) * 25
    start_indices, end_indices = indices[:-1], indices[1:]
    segment_range = np.arange(25)
    index_array = start_indices[:, None] + segment_range
    sliced_data = array[..., index_array]

    ranges = np.abs(np.ptp(sliced_data, axis=-1))
    means = np.mean(sliced_data, axis=-1)
    del sliced_data  # full-size copy no longer needed -- free before allocating more

    smallest_ordering = np.argsort(ranges, axis=-1)
    mask_zero = (ranges != 0)
    ranges = np.where(mask_zero, ranges, np.nan)
    means = np.where(mask_zero, means, np.nan)
    sorted_means = np.take_along_axis(means, smallest_ordering, axis=-1)
    average_mean = np.mean(sorted_means[..., 1:3], axis=-1)
    expanded_mean = average_mean[..., None]  # shape (..., 1)

    # broadcasts against the full trailing axis
    filtered_wvfm = (array - expanded_mean).astype(np.float32)  # float32 halves this array vs default float64
    return filtered_wvfm

def pedestal(array):

    # Define start and end indices
    indices = np.arange(0, 13) * 50  # (39,)
    start_indices, end_indices = indices[:-1], indices[1:]  # (39,)

    segment_range = np.arange(50)  # Shape: (25,)
    index_array = start_indices[:, None] + segment_range  # Shape: (39, 25)

    # Extract data using advanced indexing
    sliced_data = array[..., index_array]
    
    ranges = np.ptp(sliced_data, axis=-1)  # Compute range (n, 8, 64, 39)

    # Find ordering based on the smallest range
    smallest_region = np.argmin(ranges, axis=-1) * 50 # Shape (n, 8, 64, 39)
    pedestal_region = array[smallest_region:smallest_region+50]

    return pedestal_region

def pedestal_batch(traces):
    """Vectorized pedestal(): traces shape (n, 600) -> quietest 50-sample segment per row, shape (n, 50)."""
    indices = np.arange(0, 13) * 50
    start_indices = indices[:-1]
    segment_range = np.arange(50)
    index_array = start_indices[:, None] + segment_range        # (12, 50)

    sliced = traces[:, index_array]                              # (n, 12, 50)
    ranges = np.ptp(sliced, axis=-1)                              # (n, 12)
    smallest_region = np.argmin(ranges, axis=-1) * 50              # (n,)

    seg_idx = smallest_region[:, None] + np.arange(50)[None, :]   # (n, 50)
    return np.take_along_axis(traces, seg_idx, axis=1)

def kill_weirdos(array):
    #good_mask = (array[:,:,:,-1] > (array[:,:,:,0] - 120))
    absolute_mean = np.mean(np.abs(array[:,:,:,:] - (np.mean(array[:,:,:,:50],axis=-1))[:, :, :, np.newaxis]), axis=-1)
    normal_mean = np.mean(array[:,:,:,:50], axis=-1)
    #mean_abs_array = np.mean(absolute_array, axis=-1)
    #mean_norm_array = np.mean(array, axis=-1)
    #print(absolute_mean - normal_mean)
    good_mask = ((absolute_mean - normal_mean) < 50)
    print("Good Mask Shape:", np.shape(good_mask))
    print("Good Mask:", np.sum(good_mask, axis=0))
    #broadcasted_mask = np.tile(good_mask, (1, 1000)) 
    filtered_array = array * good_mask[:, :, :, np.newaxis]

    return filtered_array

def peak_finder(wvfm, n_noise_factor,
                    n_bins_rolled,
                    n_sqrt_rt_factor,
                    pe_weight,
                    use_rising_edge=True):
        # height = flat threshold over noise (n*sigma)
        height = n_noise_factor[..., np.newaxis, np.newaxis] * np.ones(wvfm.shape[-1]) #* noise[..., np.newaxis] * np.ones(wvfm.shape[-1])
        # dynamic_threshold = rolling threshold of previous 5 bins + n*sqrt(rolling threshold)
        wvfm_rolled = np.roll(wvfm, n_bins_rolled)
        rolling_average = uniform_filter1d(wvfm_rolled, size=n_bins_rolled)
        sqrt_rolling_average = np.sqrt(np.abs(rolling_average) * pe_weight**2)
        sqrt_rolling_average[sqrt_rolling_average == 0] = 1
        dynamic_threshold = rolling_average + n_sqrt_rt_factor*sqrt_rolling_average
        # find bins over dynamic threshold and noise floor
        bins_over_dynamic_threshold = (wvfm > dynamic_threshold) & (wvfm > height)
        # Find first bins over threshold (rising edge)
        first_bins_over = bins_over_dynamic_threshold.copy()
        first_bins_over[..., 1:] &= ~bins_over_dynamic_threshold[..., :-1]
        if use_rising_edge:
            return first_bins_over
        
def tag_dark_counts(FileList, ADCnum,Channelnum, MaxPulsenum, Voltage, Run, Window):
     
     with open(FileList, "r") as file:
          file_list = [line.strip() for line in file if line.strip() and not line.lstrip().startswith("#")]

     N_Files = len(file_list)

     print(f"Numer of input files: {N_Files}")

     dark_count_wvfm = np.zeros((MaxPulsenum,8,64,200), dtype=np.int16)
     count = np.zeros((8,64))
     sipm_channels = ([4,5,6,7,8,9] + \
                     [10,11,12,13,14,15] + \
                     [20,21,22,23,24,25] + \
                     [26,27,28,29,30,31] + \
                     [36,37,38,39,40,41] + \
                     [42,43,44,45,46,47] + \
                     [52,53,54,55,56,57] + \
                     [58,59,60,61,62,63])
     print('aaa')
    
     #file_list = glob.glob("/global/cfs/cdirs/dune/www/data/2x2/nearline/flowed_light/data_bin004/*.FLOW.hdf5")
     #file_list = glob.glob("/global/cfs/cdirs/dune/users/ajwhite/2x2_Data/2x2_Filtered/LRS_FLOW/mpd_run_hvramp_rctl_105_p1*")
     #for file in file_list:
     #start_value = 500 # for the radon injection run
     start_value = 0
     for Nf in range(N_Files):

          file = file_list[Nf]
          file_num = 0

          print("File exists: ", file)
          if file_num < N_Files: #508:
               print('file number:', file_num)
               print(count)
               file_num += 1
               if not os.path.isfile(file):
                    print("WARNING: File does not exits: ", file)
                    continue
               else:
                    with h5py.File(file, 'r') as h5:
                         #print('bbb')
                         #light_wvfms = h5['light/wvfm/data']['samples']
                         #light_wvfms_rwm = np.max(light_wvfms[:,0,18,:], axis=-1)
                         #offbeam_mask = np.where(light_wvfms_rwm < 1e4)[0]
                         #offbeam_wvfm_v1 =  light_wvfms[offbeam_mask, :, :, :]
                         offbeam_wvfm_v1 =  h5['light/wvfm/data']['samples']
                         print(f"[MEM] after reading raw 'samples' field: {_rss_gb():.2f} GB | shape {offbeam_wvfm_v1.shape} dtype {offbeam_wvfm_v1.dtype}", flush=True)
                         #print('a')
                         #del light_wvfms_rwm
                         #del light_wvfms
                         #del offbeam_mask
                         offbeam_wvfm_v2 =  thd_correct(offbeam_wvfm_v1)
                         print(f"[MEM] after thd_correct: {_rss_gb():.2f} GB", flush=True)
                         #print('b')
                         del offbeam_wvfm_v1
                         #offbeam_wvfm_v3 = kill_weirdos(offbeam_wvfm_v2)
                         #print('c')
                         #del offbeam_wvfm_v2
                         #del offbeam_wvfm_v1
                         offbeam_wvfm_v4 =  offbeam_wvfm_v2[:, :, sipm_channels, :] #* gain_array[:, :, np.newaxis]
                         print(f"[MEM] after slicing to sipm_channels: {_rss_gb():.2f} GB | shape {offbeam_wvfm_v4.shape}", flush=True)
                         #offbeam_wvfm_v4 =  offbeam_wvfm_v3[:, :, sipm_channels, :] #* gain_array[:, :, np.newaxis]
                         #print('d')
                         #del offbeam_wvfm_v3
                         del offbeam_wvfm_v2
                         #n_noise_factor=np.array([130, 130, 210, 210, 210, 210, 210, 210]) # Decommissioning values
                         n_noise_factor=np.array([110, 110, 190, 190, 190, 190, 190, 190]) # June 2025 values
                         
                         first_bins = peak_finder(wvfm=offbeam_wvfm_v4[:, :, :, 20:], n_noise_factor=n_noise_factor, n_bins_rolled=1, n_sqrt_rt_factor=0, pe_weight=0, use_rising_edge=True)
                         #print('Shape of first_bins:', np.shape(first_bins))
                         #print(first_bins)
                         #del noise_thresholds

                         #print(np.shape(np.sum(np.sum(np.sum(first_bins, axis=-1), axis=-1), axis=-1)))
                         #print(np.sum(np.sum(np.sum(first_bins, axis=-1), axis=-1), axis=-1))
                         #per_event_hits = np.sum(np.sum(np.sum(first_bins, axis=-1), axis=-1), axis=-1)
                         #print(np.sum(per_event_hits > 0))

     last_tick = offbeam_wvfm_v4.shape[-1] - (Window - 2)

     # 1. Every candidate hit across the whole (event, adc, channel) space, one call
     evt_i, adc_i, ch_i, tick_i = np.where(first_bins)
     hit_value = tick_i + 20  # undo the [..., 20:] offset applied before peak_finder

     # cut_3, plus the Channelnum limit the original `range(Channelnum)` enforced
     keep = (hit_value <= last_tick) & (ch_i < Channelnum)
     evt_i, adc_i, ch_i, hit_value = evt_i[keep], adc_i[keep], ch_i[keep], hit_value[keep]

     if len(evt_i) > 0:
          # 2. Extract every candidate's window in one fancy-index call
          window_offsets = np.arange(-3, Window - 3)
          sample_idx = hit_value[:, None] + window_offsets[None, :]
          forms = offbeam_wvfm_v4[evt_i[:, None], adc_i[:, None], ch_i[:, None], sample_idx]

          # 3. cuts 4-9, vectorized across every candidate at once
          is_mod123 = adc_i >= 2
          limit = np.where(is_mod123, 80, 100)
          cut_4 = np.max(forms[:, -6:], axis=1) < limit
          cut_5 = forms[:, 0] < limit
          cut_6 = np.min(forms, axis=1) > -limit
          rises = ((forms[:, 2] > forms[:, 3]).astype(np.int8)
                 + (forms[:, 3] > forms[:, 4]).astype(np.int8)
                 + (forms[:, 4] > forms[:, 5]).astype(np.int8)
                 + (forms[:, 5] > forms[:, 6]).astype(np.int8))
          cut_7 = rises < 2
          cut_8 = np.all(forms[:, 3:5] > n_noise_factor[adc_i][:, None], axis=1)
          sums = np.sum(forms, axis=1)
          cut_9 = np.where(is_mod123, (sums < 2.5e4) & (sums > -1.5e3), sums > -1.5e3)
          passed_49 = cut_4 & cut_5 & cut_6 & cut_7 & cut_8 & cut_9

          # 4. Pedestal check -- only for candidates that already passed cuts 4-9
          would_be_accepted = np.zeros(len(evt_i), dtype=bool)
          idx_49 = np.where(passed_49)[0]
          ped_regions = np.empty((0, 50), dtype=np.int16)
          if len(idx_49) > 0:
               traces_49 = offbeam_wvfm_v4[evt_i[idx_49], adc_i[idx_49], ch_i[idx_49], :]
               ped_regions = pedestal_batch(traces_49).astype(np.int16)
               ped_ok = ((np.abs(np.mean(ped_regions, axis=1)) < 20)
                        & (np.min(ped_regions, axis=1) > -100)
                        & (np.max(ped_regions, axis=1) < 100))
               would_be_accepted[idx_49] = ped_ok

          # 5. Apply the per-channel MaxPulsenum cap in the SAME traversal order the
          #    original loop used (adc outer, channel next, event next, tick innermost) --
          #    reproduces exactly which hits get kept once a channel fills up.
          adc_channel_i = np.asarray(sipm_channels)[ch_i]
          group_id = adc_i.astype(np.int64) * 64 + adc_channel_i.astype(np.int64)
          order = np.lexsort((hit_value, evt_i, ch_i, adc_i))

          g_sorted = group_id[order]
          acc_sorted = would_be_accepted[order]
          df = pd.DataFrame({'group': g_sorted, 'accepted': acc_sorted})
          cum_before = (df.groupby('group')['accepted'].cumsum() - df['accepted']).to_numpy()

          final_mask_sorted = acc_sorted & (cum_before < MaxPulsenum)
          final_positions = order[final_mask_sorted]
          final_slots = cum_before[final_mask_sorted].astype(np.int64)
          final_adc = adc_i[final_positions]
          final_adc_channel = adc_channel_i[final_positions]

          # 6. Write the final, capped set of accepted hits into dark_count_wvfm
          pos_to_49 = -np.ones(len(evt_i), dtype=np.int64)
          pos_to_49[idx_49] = np.arange(len(idx_49))
          ped_rows = pos_to_49[final_positions]

          dark_count_wvfm[final_slots, final_adc, final_adc_channel, 60:(60 + Window)] += \
               forms[final_positions].astype(np.int16)
          dark_count_wvfm[final_slots, final_adc, final_adc_channel, 0:50] += \
               ped_regions[ped_rows]

          np.add.at(count, (final_adc, final_adc_channel), 1)

     del offbeam_wvfm_v4
     del first_bins
     print(count)
     del count

     return dark_count_wvfm

def main(output_file, file_list, ADCnum=None, Channelnum=None, MaxPulsenum = None, voltage=None, run=None, window=50):
    
    new_wvfms = tag_dark_counts(FileList=file_list, ADCnum=ADCnum, Channelnum = Channelnum, MaxPulsenum = MaxPulsenum, Voltage=voltage, Run=run, Window=window)
    np.savez(output_file, data=new_wvfms)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--file_list', required=True, type=str, \
                        help='''string corresponding to text file containing one input HDF5 path per line including path and name''')
    parser.add_argument('-o', '--output_file', default=None, required=True, type=str, \
                        help='''string corresponding to desired output file path and name''')
    parser.add_argument('-a', '--ADCnum', default=None, required=True, type=int, \
                        help='''int corresponding to ADC number''')
    parser.add_argument('-c', '--Channelnum', default=None, required=True, type=int, \
                        help='''int corresponding to number of channels''')
    parser.add_argument('-p', '--MaxPulsenum', default=None, required=True, type=int, \
                        help='''int corresponding to Max Pulse number''')
    parser.add_argument('-v', '--voltage', default=None, required=True, type=int, \
                        help='''int corresponding to operating voltage''')
    parser.add_argument('-r', '--run', default=None, required=True, type=int, \
                        help='''int corresponding to light file run number''')
    parser.add_argument('-w', '--window', default=None, required=True, type=int, \
                        help='''int corresponding to tick width of signal integration window''')
    args = parser.parse_args()
    main(**vars(args))
