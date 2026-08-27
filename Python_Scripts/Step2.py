import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
import scipy
from scipy.integrate import simpson
from scipy.stats import poisson, norm, exponnorm
from scipy.optimize import curve_fit
from matplotlib.backends.backend_pdf import PdfPages
import yaml
from scipy.signal import find_peaks
from glob import glob

SAMPLE_WIDTH_NS = 16
N_PHOTOELECTRONS = 2

def load_channel_waveforms(file_paths, adc, ch):
    waveform_chunks = []
    pulses_per_file = []

    for path in file_paths:
        print(f"Reading ADC {adc}, channel {ch} from: ", path)
        with np.load(path) as npz_file:
            data = npz_file["data"]
            current_waveforms = data[:, adc, ch, :].copy()
            del data

        nonempty_mask = np.any(current_waveforms != 0, axis=1)
        current_waveforms = current_waveforms[nonempty_mask]

        print("Pulses found in this file: ", len(current_waveforms))
        pulses_per_file.append(len(current_waveforms))

        if len(current_waveforms) > 0:
            waveform_chunks.append(current_waveforms)
            #return np.empty((0,200), dtype=np.int16)#, pulses_per_file

    if len(waveform_chunks) == 0:
        return np.empty((0,200), dtype=np.int16)
    
    combined_waveforms = np.concatenate(waveform_chunks, axis=0)
    return combined_waveforms


def clean_histogram(hist, zero_run_threshold=10):
    hist = np.asarray(hist)
    zero_mask = (hist == 0)

    # Identify leading zero run
    leading_zeros = 0
    for count in zero_mask:
        if count:
            leading_zeros += 1
        else:
            break

    # Identify trailing zero run
    trailing_zeros = 0
    for count in zero_mask[::-1]:
        if count:
            trailing_zeros += 1
        else:
            break

    to_keep = np.ones_like(hist, dtype=bool)

    # Mask leading zeros if long enough
    if leading_zeros >= zero_run_threshold:
        to_keep[:leading_zeros] = False

    # Mask trailing zeros if long enough
    if trailing_zeros >= zero_run_threshold:
        to_keep[-trailing_zeros:] = False

    cleaned_hist = hist[to_keep]
    cleaned_indices = np.where(to_keep)[0]
    return cleaned_hist, cleaned_indices


def fit_pedestal(hist_vals, bin_edges, fixed_amp=None, fixed_mu=None):
    centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    # Use defaults if not supplied
    if fixed_amp is None:
        fixed_amp = max(hist_vals)
        #print('Ped Amplitude:', fixed_amp)
    if fixed_mu is None:
        fixed_mu = np.average(centers, weights=hist_vals[:-1])
        #print('Ped Mu:', fixed_mu)

    # Model with fixed amplitude and mean
    def fixed_gaussian(x, sigma):
        weight = fixed_amp * (np.sqrt(2*np.pi)*sigma)
        return weight * norm.pdf(x, fixed_mu, sigma)

    # Initial guess for sigma
    var = np.sum((centers - fixed_mu)**2) / len(centers)
    sigma0 = np.sqrt(var)

    # Fit only sigma
    popt, pcov = curve_fit(fixed_gaussian, centers, hist_vals[:-1], p0=[sigma0])
    #print('p0 Pedestal:', popt)

    return fixed_amp, fixed_mu, popt[0]

def spe_model(q, A1, mu, q_spe, sigma_spe, A_lump, mu_lump, sigma_lump, A0, mu0, sigma0, n_max=3):
  q = np.asarray(q)
  total = np.zeros_like(q, dtype=np.float64)
  components = []
  lobes = []

  for n in range(n_max + 1):
    if n == 0:
      # Pedestal (0 charge peak) with freely floating width and amplitude
      weight = A0 * (np.sqrt(2*np.pi)*sigma0)  # Freely floating amplitude for pedestal
      sigma_n = sigma0  # Freely floating width for pedestal
      mean = mu0  # Initialize mean for the pedestal
      component = weight * norm.pdf(q, loc=mean, scale=sigma_n)
      total += component
      components.append(component)
      #print(str(n)+' has run through')
    else:
      mean = n * q_spe
      #mean = q0 + (n - 1) * q_spe
      sigma_n = np.sqrt(sigma0**2 + (n * sigma_spe**2))
      weight = A1 * (np.sqrt(2*np.pi)*sigma_n) * poisson.pmf(n-1, mu)#* np.exp(-(n-1) / mu)
      weight_2 = weight * A_lump #* poisson.pmf(n-1, mu)
      #mean_2 = q0 + ((n - 1) * q_spe) + mu_lump
      mean_2 = n * q_spe + mu_lump
      sigma_n_2 = sigma_lump
    # Gaussian distribution for each component
      component = weight * norm.pdf(q, loc=mean, scale=sigma_n) #+ weight_2 * norm.pdf(q, loc=mean_2, scale=sigma_n_2)
      component_2 = weight_2 * norm.pdf(q, loc=mean_2, scale=sigma_n_2)
      #print(str(n)+' has run through')
      total += component
      components.append(component)
      total += component_2
      lobes.append(component_2)
    #print(str(n)+' has run through')

  return total, components, lobes


def spe_model_exp(q, A1, mu, q_spe, sigma_spe, A_lump, mu_lump, sigma_lump, A0, mu0, sigma0, q0, lump_shape, n_max=3):
    """ 
    Same idea, but with an exponential.
    Larger lump shape is more dominant tail.
    """
    q = np.asarray(q)
    total = np.zeros_like(q, dtype=np.float64)
    components = []
    lobes = []

    for n in range(n_max + 1):
        if n == 0:
            # Pedestal (0 charge peak) with freely floating width and amplitude
            weight = A0 * (np.sqrt(2*np.pi)*sigma0)  # Freely floating amplitude for pedestal
            sigma_n = sigma0  # Freely floating width for pedestal
            mean = mu0  # Initialize mean for the pedestal
            component = weight * norm.pdf(q, loc=mean, scale=sigma_n)
            total += component
            components.append(component)
            #print(str(n)+' has run through')
        # I'm only attempting this for 1PE
        elif n == 1:
            mean = q0 + (n - 1) * q_spe
            sigma_n = np.sqrt(sigma0**2 + (n * sigma_spe**2))
            weight = A1 * (np.sqrt(2*np.pi)*sigma_n) * poisson.pmf(n-1, mu)#* np.exp(-(n-1) / mu)
            weight_2 = weight * A_lump #* poisson.pmf(n-1, mu)
            mean_2 = q0 + ((n - 1) * q_spe) + mu_lump
            sigma_n_2 = sigma_lump
            # Gaussian distribution for the PE peaks
            component = weight * norm.pdf(q, loc=mean, scale=sigma_n) #+ weight_2 * norm.pdf(q, loc=mean_2, scale=sigma_n_2)
            
            # Exponential Gaussian
            component_2 = weight_2 * exponnorm.pdf(q, lump_shape, loc=mean_2, scale=sigma_n_2)
            
            total += component
            components.append(component)
            total += component_2
            lobes.append(component_2)
        else:
            mean = q0 + (n - 1) * q_spe
            sigma_n = np.sqrt(sigma0**2 + (n * sigma_spe**2))
            weight = A1 * (np.sqrt(2*np.pi)*sigma_n) * poisson.pmf(n-1, mu)#* np.exp(-(n-1) / mu)
            weight_2 = weight * A_lump #* poisson.pmf(n-1, mu)
            mean_2 = q0 + ((n - 1) * q_spe) + mu_lump
            sigma_n_2 = sigma_lump
            # Gaussian distribution for each component
            component = weight * norm.pdf(q, loc=mean, scale=sigma_n) #+ weight_2 * norm.pdf(q, loc=mean_2, scale=sigma_n_2)
            component_2 = weight_2 * norm.pdf(q, loc=mean_2, scale=sigma_n_2)
            #print(str(n)+' has run through')
            total += component
            components.append(component)
            total += component_2
            lobes.append(component_2)
            #print(str(n)+' has run through')

    return total, components, lobes

def spe_peak(q, A1, mu, q_spe, sigma_spe, sigma0, n_max=3):
    q = np.asarray(q)
    total = np.zeros_like(q, dtype=np.float64)
    components = []

    for n in range(n_max + 1):
        #mean = q0 + (n * q_spe)
        mean = (n+1) * q_spe
        sigma_n = np.sqrt(sigma0**2 + ((n+1) * sigma_spe**2))
        weight = A1 * (np.sqrt(2*np.pi)*sigma_n) * poisson.pmf(n, mu)#* np.exp(-(n-1) / mu)
        # Gaussian distribution for each component
        component = weight * norm.pdf(q, loc=mean, scale=sigma_n)

        total += component
        components.append(component)

    return total, components

# Fit SPE spectrum using the model
def fit_lump_spectrum(hist, bins, npe=N_PHOTOELECTRONS, A0=None, mu0=None, sigma0=None, q_spe=None, p0=None):
    centers = 0.5 * (bins[1:] + bins[:-1])
    # NB: uses old value for q_spe as bounds for the new value
    popt, pcov = curve_fit(
        lambda q, A1, mu, q_spe, sigma_spe, A_lump, mu_lump, sigma_lump: spe_model(q, A1, mu, q_spe, sigma_spe, A_lump, mu_lump, sigma_lump, A0, mu0, sigma0, npe)[0],
        centers,
        hist[:-1],
        p0=p0,
        # A1, mu, q_spe, sigma_spe, A_lump, mu_lump, sigma_lump
    bounds=([0.75*p0[0], 0.05, q_spe*0.7, 100, 0.01, 0.2*q_spe, 1], [p0[0]+(0.25*p0[0]), 5, q_spe*1.3, 2e3, 1.1, 0.7*q_spe, 1.5e3]) # Old: ([p0[0]-(0.25*p0[0]), 0.05, q0*0.7, 1, 0.01, 0.2*q0, 1], [p0[0]+(0.25*p0[0]), 5, q0*1.3, 1e3, 1.1, 0.7*q0, 1e3])
    )
    return popt, pcov, centers

def fit_exp_spectrum(hist, bins, npe=N_PHOTOELECTRONS, A0=None, mu0=None, sigma0=None, q0=None, p0=None):
    centers = 0.5 * (bins[1:] + bins[:-1])
    #print('entering SPE fit')
    #print('Lower Bounds:',[p0[0]-100, 0.05, q0*0.7, 1, 1, 0.1*q0, 1.1*p0[3]])
    #print('Upper Bounds:',[p0[0]+100, 5, q0*1.5, 1e3, p0[0]*0.5, 0.5*q0, 1e3])
    popt, pcov = curve_fit(
        lambda q, A1, mu, q_spe, sigma_spe, A_lump, mu_lump, sigma_lump: spe_model_exp(q, A1, mu, q_spe, sigma_spe, A_lump, mu_lump, sigma_lump, A0, mu0, sigma0, q0, npe)[0],
        centers,
        hist[:-1],
        p0=p0,
      # very good for 47p5 V  bounds=([p0[0]-(0.25*p0[0]), 0.05, q0*0.7, 1, 0.01, 0.2*q0, 1,0.1], [p0[0]+(0.25*p0[0]), 5, q0*1.3, 4e3, 1.1, 0.7*q0, 1e3,5]) # Old: ([p0[0]-(0.25*p0[0]), 0.05, q0*0.7, 1, 0.01, 0.2*q0, 1], [p0[0]+(0.25*p0[0]), 5, q0*1.3, 1e3, 1.1, 0.7*q0, 1e3])
    bounds=([p0[0]-(0.25*p0[0]), 0.05, q0*0.7, 1, 0.01, 0.2*q0, 1, 0.1], [p0[0]+(0.35*p0[0]), 5, q0*1.3, 4e3, 1.1, 0.7*q0, 1e3,5]) # Old: ([p0[0]-(0.25*p0[0]), 0.05, q0*0.7, 1, 0.01, 0.2*q0, 1], [p0[0]+(0.25*p0[0]), 5, q0*1.3, 1e3, 1.1, 0.7*q0, 1e3])

    )

    return popt, pcov, centers

# Fit PE peaks only
def fit_spe_spectrum(hist, bins, sigma0=None, npe=N_PHOTOELECTRONS, p0=None):
    centers = 0.5 * (bins[1:] + bins[:-1])
    #print('entering SPE fit')
    popt, pcov = curve_fit(
        lambda q, A1, mu, q_spe, sigma_spe: spe_peak(q, A1, mu, q_spe, sigma_spe, sigma0, npe)[0],
        centers,
        hist[:-1],
        p0=p0,
        #p0: A1, mu, q_spe, sigma_spe, sigma0
        bounds=([0.75*p0[0], 0.1, p0[2]*0.7, 1], [1.25*p0[0], 5, p0[2]*1.5, 4e3]) 
    )
    return popt, pcov, centers

# Chi-squared per degree of freedom
def chi_squared(y_obs, y_fit, dof=None, min_expected=5):
    # Only use bins with decent statistics
    mask = y_fit >= min_expected
    if np.sum(mask) < 2:
        return np.inf  # Not enough points

    residuals = y_obs[mask] - y_fit[mask]
    chi2 = np.sum(residuals**2 / y_fit[mask])
    if dof is None:
        dof = np.sum(mask) - 1  # default to N - 1
    return chi2 / dof


def analyze_spe(file_path, output_file, initial_guess_v0=None, adc_num=None, window=None):

    results = []

    # Configuration

    # Here, we guess how many pe we have
    N_PHOTOELECTRONS = 3
    fits_failed = np.zeros(8)
    dead = np.zeros(8)
    pdf_filename = output_file
    gain_outputs = np.zeros((8,64,3), dtype=np.float64)
    #with h5py.File(file_path, "r") as f
    #waveform_data = []
    #for file in file_path:
    #    waveform_data_temp = np.load(file)
    #    waveform_data.append(waveform_data_temp[list(waveform_data_temp.keys())[0]])

    #waveforms = np.concatenate(waveform_data)
    #print("Combined waveform shape: ", waveforms.shape)
    #waveform_data = np.load(file_path)
    #waveforms = waveform_data[list(waveform_data.keys())[0]] # Code for one single file.
    ##########################My Update
    #with open('/global/cfs/cdirs/dune/users/ajwhite/2x2_LRS_DataAssess/2025_Calibration/AFIViewer/WaveformCalib_Mod0Estimates.yaml', 'r') as file:
    #    data = yaml.safe_load(file)

    # Access the gain dictionary
    #gain_dict = data['params']['gain']
    gain_dict = {} 
#####################Ends here

    with PdfPages(pdf_filename) as pdf_output:
        #_, nadcs, nchans, _ = waveforms.shape
        #for adc in range(nadcs):
        sipm_channels = (list(range(4,16)) + list(range(20,32)) + list(range(36,48)) + list(range(52,64)))

        for adc in range(8):#[adc_num]: 
            for ch in sipm_channels:

                #wfs_v1 = waveforms[:,adc,ch,:]
                wfs = load_channel_waveforms(file_path, adc, ch)
                #charge_mask = (np.mean(wfs_v1, axis=-1)!=0)
                print(f"ADC {adc}, Channel {ch}")
                print(f"{len(wfs)} total pulses")
                #print(f"Pulses per NPZ file: ", pulses_per_file)
                if len(wfs) > 0: 
                    # Use separate initial guesses for Mod0 and Mod123
                    if adc < 2:
                        bin_width, sigmaspe_est, combo_max_clip, sig_max_clip = initial_guess_v0[:4]
                    else:
                        bin_width, sigmaspe_est, combo_max_clip, sig_max_clip = initial_guess_v0[4:]

                    #wfs = wfs_v1[charge_mask==1]
                    #del wfs_v1, charge_mask

                    # Dependent on input waveforms
                    ped_integrals = np.trapz(wfs[:,0:50], axis=-1)
                    #sig_integrals = np.trapz(wfs[:,60:85], axis=-1)
                    sig_integrals = np.trapz(wfs[:,60:(60+window)], axis=-1)
                    #sig_subtracted = sig_integrals -  (np.mean(wfs[:,0:50], axis=-1)*25)
                    sig_subtracted = sig_integrals -  (np.mean(wfs[:,0:50], axis=-1)*window)
                    del sig_integrals

                    combo_integrals = np.concatenate((sig_subtracted, ped_integrals), axis=0)

                    # Fit pedestal
                    ped_min = np.min(ped_integrals)
                    ped_max = np.max(ped_integrals)
                    if ped_max > 1.2e4:
                        ped_max = 1.2e4
                    ped_bins_pre = np.arange(start=ped_min, stop=ped_max + bin_width, step=bin_width)
                    ped_hist_1, ped_bins_1 = np.histogram(ped_integrals, bins=ped_bins_pre) # define a bin width instead
                    ped_hist, ped_bins_idx = clean_histogram(ped_hist_1, zero_run_threshold=5)
                    ped_bins = ped_bins_1[ped_bins_idx]
                    del ped_hist_1, ped_bins_1
                    del ped_integrals
                    try:
                        A0, mu0, sigma0 = fit_pedestal(ped_hist, ped_bins)
                        del ped_bins
                    except: 
                        A0, mu0, sigma0 = 1, 0, 1
                    del ped_hist

                    # Fit signal spectrum
                    sig_min = np.min(sig_subtracted)
                    sig_max = np.max(sig_subtracted)

                    if sig_max > sig_max_clip:
                        sig_max = sig_max_clip
                    
                    sig_bins_pre = np.arange(start=sig_min, stop=sig_max + bin_width, step=bin_width)

                    sig_hist_1, sig_bins_1 = np.histogram(sig_subtracted, bins=sig_bins_pre)
                    sig_hist, sig_bins_idx = clean_histogram(sig_hist_1, zero_run_threshold=5)
                    sig_bins = sig_bins_1[sig_bins_idx]
                    del sig_hist_1, sig_bins_1
                    try:
                        A1 = max(sig_hist)
                    except ValueError as e:
                        print(f"{e} for ADC {adc}, channel {ch}")
                        continue

                    peak_indices, properties = find_peaks(sig_hist, prominence = 0.05 * np.max(sig_hist))

                    if len(peak_indices) == 0:
                        print("WARNING: No signal peak found using maximum bin")
                    else:
                        qspe_est = sig_bins[peak_indices[0]]
                    print("Detected SPE peak position: ", sig_bins[peak_indices])
                    print("Selected qspe_est: ", qspe_est)

                    #qspe_est = sig_bins[np.argmax(sig_hist)]
                    mu_est = 0.2 # old: 0.5 #2400 #sig_bins[np.argmax(sig_hist)]

                    initial_guess = [A1, mu_est, qspe_est, sigmaspe_est]
    

                    try:
                        popt_spe_1, pcov_spe_1, centers_1 = fit_spe_spectrum(sig_hist, sig_bins, sigma0=sigma0, npe=N_PHOTOELECTRONS, p0=initial_guess)
                        new_npe = N_PHOTOELECTRONS
                        print(f"Fitted gain first fit: {popt_spe_1[2]}")
                        print(initial_guess)
                    except:
                        popt_spe_1 = [A1, mu_est, qspe_est, sigmaspe_est]
                        new_npe = 2
                        print("Fit unsuccessful, using hardcoded values")
                    print('Made it through First pass')
                    del sig_hist, sig_bins
                    mu_lump = 0.3*popt_spe_1[2] # mu lump is basically the distance between the peak of the lump and the peak of the PE
                    sigma_lump = 1.4*popt_spe_1[3]
                    A_lump = 0.4
                    second_guess = [A1, popt_spe_1[1], qspe_est, sigmaspe_est, A_lump, mu_lump, sigma_lump]
                    bounds = [second_guess[0]-(0.25*second_guess[0]), 0.05, popt_spe_1[2]*0.7, 1, 0.01, 0.2*popt_spe_1[2], 1.3*second_guess[3]], [second_guess[0]+(0.25*second_guess[0]), 5, popt_spe_1[2]*1.3, 3e3, 1.1, 0.7*popt_spe_1[2],5e3]

                    combo_min = np.min(combo_integrals)
                    combo_max = np.max(combo_integrals)
                    if combo_max > combo_max_clip: # Defined separately for Module 0 and Module 123
                        combo_max = combo_max_clip

                    combo_bins_pre = np.arange(start=combo_min, stop=combo_max + bin_width, step=bin_width)
                    combo_hist_1, combo_bins_1 = np.histogram(combo_integrals, bins=combo_bins_pre)
                    combo_hist, combo_bins_idx = clean_histogram(combo_hist_1, zero_run_threshold=5)
                    combo_bins = combo_bins_1[combo_bins_idx]
                    print('Combo_hist bins:', np.unique(combo_bins[1:] - combo_bins[:-1]))
                    del combo_hist_1, combo_bins_1

                    try:
                        popt_spe, pcov_spe, centers = fit_lump_spectrum(combo_hist, combo_bins, npe=(new_npe+1), A0=A0, mu0=mu0, sigma0=sigma0, q_spe=qspe_est,  p0=second_guess)
                        
                        print('Made it through second pass')

                        fit_vals, components, lobes = spe_model(centers, A0=A0, mu0=mu0, sigma0=sigma0, n_max=new_npe, *popt_spe)
                        print(f"optimal values: for ADC {adc} ch {ch}", popt_spe)
                        #print('Made it through the fit')
                        chi2_ndf = chi_squared(combo_hist[:-1], fit_vals, len(combo_hist[:-1]) - len(popt_spe))
                        gain_outputs[adc,ch,0] += np.float64(popt_spe[2]/4)
                        gain_outputs[adc,ch,1] += np.float64(chi2_ndf)
                        gain_outputs[adc,ch,2] += np.float64(np.sqrt(np.diag(pcov_spe))[2])
                        # Plot results
                        #plt.figure(figsize=(10, 5))
                        fig, axs = plt.subplots(3, 1, figsize=(10, 12),gridspec_kw={'height_ratios': [2, 1, 1]})
                        # --- First subplot: SPE fit histogram ---
                        axs[0].step(combo_bins, combo_hist, where='mid', label="Signal Data", color="black")
                        colors = plt.cm.tab20(np.linspace(0, 1, len(components)))
                        for i, comp in enumerate(components):
                            axs[0].plot(combo_bins[:-1], comp, '--', color=colors[i], label=f'{i} PE')
                        for i, lobe in enumerate(lobes):
                            axs[0].plot(combo_bins[:-1], lobe, ':', color=colors[i+1], label=f'{i+1} PE Lobe')
                        axs[0].plot(centers, fit_vals, 'g--', label=f"SPE Fit\nμ={popt_spe[1]:.2f}, Gain={popt_spe[2]:.2f}, χ²/ndf={chi2_ndf:.2f}")
                        axs[0].set_title(f"SPE Fit for ADC {adc}, Channel {ch}")
                        axs[0].set_xlabel("Integrated Charge [a.u.]")
                        axs[0].set_ylabel("Counts")
                        axs[0].axvline(qspe_est, color='red',label=f"hist peak: {qspe_est}")
                        
                        for v in range(1, 4):
                            axs[0].axvline(popt_spe[2]*v, color=f'C{v+6}', lw=3, label=f"{v}PE peak")

                        axs[0].legend()
                        axs[0].grid(True)

                        # --- Second subplot: Overlaid waveforms ---
                        for wf in wfs:  # optional: show only first 100
                            axs[1].plot(wf, alpha=0.3, color='blue', rasterized=True)
                        axs[1].set_title(f"ADC {adc}, Channel {ch}: Dark Count Waveforms")
                        axs[1].set_xlabel("Sample")
                        axs[1].set_ylabel("Raw ADC Counts")
                        axs[1].grid(True)

                        gain_integrals = (sig_subtracted) / np.int32(popt_spe[2])
                        #gain_integrals = sig_subtracted / np.int32(popt_spe[2])
                        og_gain = gain_dict.get(adc, {}).get(ch, 0.0)
                        #gain_og_int = ((sig_subtracted - np.int32(popt_spe[1])) / 4) * og_gain
                        gain_og_int = (sig_subtracted / 4) * og_gain
                        #bin_min = 0
                        #bin_max = 10
                        # Create shared bin edges
                        #bins = np.linspace(bin_min, bin_max + 0.1, 0.1)
                        bin_width = 0.1
                        data_min = 0
                        data_max = 10
                        bins = np.arange(np.floor(data_min), np.ceil(data_max) + bin_width, step=bin_width)
                        for k in range(9):
                            axs[2].axvline(x=k+1, linewidth=3, color='yellowgreen', alpha=0.5)
                        # Plot both histograms using identical bins on axs[2]
                        axs[2].hist(gain_og_int, bins=bins, histtype='stepfilled', label='LED Calibrated', alpha=0.3, color="mediumblue")
                        axs[2].hist(gain_integrals, bins=bins, histtype='step', label='DC Calibrated', color='black')
                        #gain_hist, _ = np.histogram(gain_integrals, bins=bins)
                        #og_gain_hist, _ = np.histogram(gain_og_int, bins=bins)
                        #centers = 0.5 * (bins[:-1] + bins[1:])
                        #axs[2].fill_between(centers, og_gain_hist, step="mid", alpha=0.3, color="mediumblue", label="LED Calibrated")
                        #axs[2].hist(og_gain_bins, og_gain_hist, color="mediumblue", histtype='stepfilled', alpha=0.3, label="LED Calibrated")
                        #axs[2].step(og_gain_bins[:-1], og_gain_hist, where='mid', label="LED Calibrated", color="mediumblue")
                        #axs[2].step(centers, gain_hist, step="mid", color="black", label="DC Calibrated")
                        axs[2].set_title(f"ADC {adc}, Channel {ch}: Integral of Calibrated Dark Counts")
                        axs[2].set_xlabel("Integral [p.e.]")
                        axs[2].set_ylabel("Frequency [log]")
                        #axs[2].set_yscale('log')
                        axs[2].grid(True)
                        axs[2].legend()
                        
                        plt.tight_layout()
                        pdf_output.savefig(fig)
                        plt.close(fig)

                    except Exception as e:
                        print(f"Fit failed for ADC {adc} CH {ch}")
                        print(f"{type(e).__name__}")
                        fits_failed[adc] += 1
                        fig, axs = plt.subplots(2, 1, figsize=(10, 6),gridspec_kw={'height_ratios': [1, 1]})
                        for wf in wfs:
                            axs[0].plot(wf, alpha=0.3, color='blue', rasterized=True)
                        axs[0].set_title(f"ADC {adc}, Channel {ch}: Fit Failure on Dark Counts")
                        axs[0].set_xlabel("Sample")
                        axs[0].set_ylabel("Raw ADC Counts")
                        axs[0].grid(True)
                        try: 
                            data_min = np.sort(combo_integrals)[10]
                            data_max = np.sort(combo_integrals)[-10]
                            bins = np.arange(np.floor(data_min), np.ceil(data_max) + bin_width, step=bin_width)
                            axs[1].hist(combo_integrals, bins=bins, histtype='step', label='Pedestal and Signal Integrals', color='black')
                            axs[1].set_title(f"ADC {adc}, Channel {ch}: Fit Failure on Dark Counts")
                            axs[1].set_xlabel("Integral")
                            axs[1].set_ylabel("Frequency")
                            axs[1].grid(True)

                            plt.tight_layout()
                            pdf_output.savefig(fig)
                            plt.close(fig)

                            second_guess_rounded = [round(val, 2) for val in second_guess]
                            lower_bounds_rounded = [round(val, 2) for val in bounds[0]]
                            upper_bounds_rounded = [round(val, 2) for val in bounds[1]]

                            # Create text lines
                            lines = []
                            for idx, (val, low, high) in enumerate(zip(second_guess_rounded, lower_bounds_rounded, upper_bounds_rounded)):
                                line = f"Param {idx+1}: {low} < **{val}** < {high}"
                                lines.append(line)
                            # Plot the text box
                            fig, ax = plt.subplots(figsize=(6, len(lines)*0.5 + 1))
                            ax.axis('off')

                            for i, line in enumerate(lines):
                                ax.text(0.05, 1 - (i + 1) * 0.1, line, fontsize=12, verticalalignment='top')

                            plt.tight_layout()
                            pdf_output.savefig(fig)
                            plt.close(fig)

                        except:
                            data_min = np.min(combo_integrals)
                            data_max = np.max(combo_integrals)
                            bins = np.arange(np.floor(data_min), np.ceil(data_max) + bin_width, step=bin_width)
                            axs[1].hist(combo_integrals, bins=bins, histtype='step', label='Pedestal and Signal Integrals', color='black')
                            axs[1].set_title(f"ADC {adc}, Channel {ch}: Fit Failure on Dark Counts")
                            axs[1].set_xlabel("Integral")
                            axs[1].set_ylabel("Frequency")
                            axs[1].grid(True)

                            plt.tight_layout()
                            pdf_output.savefig(fig)
                            plt.close(fig)

                        #del initial_guess, combo_hist, combo_bins
                    try:
                        lump_shape = 1.5
                        second_guess_exp = [A1, popt_spe_1[2], popt_spe_1[3], sigmaspe_est, A_lump, mu_lump, sigma_lump, lump_shape]
                        bounds_exp = [second_guess_exp[0]-(0.25*second_guess_exp[0]), 0.05, popt_spe_1[0]*0.7, 1, 0.01, 0.2*popt_spe_1[0], 1.3*second_guess_exp[3], 0.1], [second_guess_exp[0]+(0.25*second_guess_exp[0]), 5, popt_spe_1[0]*1.3, 3e3, 1.1, 0.7*popt_spe_1[0], 3e3,5]
                        popt_spe_exp, pcov_spe_exp, centers_exp = fit_exp_spectrum(combo_hist, combo_bins, npe=(new_npe+1), A0=A0, mu0=mu0, sigma0=sigma0, q0=q0,  p0=second_guess_exp)
                        
                        print('Made it through second pass')

                        fit_vals_exp, components_exp, lobes_exp = spe_model_exp(centers_exp, A0=A0, mu0=mu0, sigma0=sigma0, q0=q0, n_max=new_npe, *popt_spe_exp)
                        print("optimal values: ", popt_spe_exp)

                    except:
                        pass
                else:
                    print(f"Inactive or Dead for ADC {adc} CH {ch}")
                    dead[adc] += 1
            print(f"Fits failed for ADC {adc}: {fits_failed[adc]}")    
    #del waveform_data, waveforms
    print("# Fits failed: ", fits_failed, "total: ", np.sum(fits_failed))
    print("# Dead channels: ", dead, "total: ", np.sum(dead))
    return gain_outputs

def main(file_list, output_file, gain_output, adc=None, bias_voltage=None, run=None, window=50):
    # Either provide an input file with the bash script or provide them here
    bias_voltage = float(bias_voltage)
    inputs = {
        45.0: ['/global/cfs/cdirs/dune/users/ajwhite/2x2_LRS_DataAssess/2025_Calibration/102025_FinalDC_npzs/Run2_DC_100125_45_24dB_90f_v16_wvfms.npz'],
        45.5: ['/global/cfs/cdirs/dune/users/ajwhite/2x2_LRS_DataAssess/2025_Calibration/Run2_DC_102225_45p5_24dB_200f_v03_wvfms.npz'],
        46.0: ['/global/cfs/cdirs/dune/users/ajwhite/2x2_LRS_DataAssess/2025_Calibration/Run2_DC_102225_46_24dB_200f_v03_wvfms.npz'],
        46.5: ['/global/cfs/cdirs/dune/users/ajwhite/2x2_LRS_DataAssess/2025_Calibration/102025_FinalDC_npzs/Run2_DC_100125_46p5_24dB_90f_v16_wvfms.npz'],
        46.7: ['/global/cfs/cdirs/dune/users/ajwhite/2x2_LRS_DataAssess/2025_Calibration/Run2_10dB_SPE_Co66754_50_nominal_50tick_v01_wvfms.npz'],
        47.0: ['/global/cfs/cdirs/dune/users/ajwhite/2x2_LRS_DataAssess/2025_Calibration/102025_FinalDC_npzs/Run2_DC_100125_47_24dB_90f_v16_wvfms.npz'],
        47.5: ['/global/cfs/cdirs/dune/users/ajwhite/2x2_LRS_DataAssess/2025_Calibration/102025_FinalDC_npzs/Run2_DC_100125_47p5_24dB_90f_v16_wvfms.npz'],
        48.0: ['/global/cfs/cdirs/dune/users/ajwhite/2x2_LRS_DataAssess/2025_Calibration/Run2_DC_102225_48_24dB_200f_v03_wvfms.npz'], #/dune/users/ajwhite/2x2_LRS_DataAssess/2025_Calibration/102025_FinalDC_npzs/Run2_DC_100125_48_24dB_90f_v16_wvfms.npz
        48.5: ['/global/cfs/cdirs/dune/users/ajwhite/2x2_LRS_DataAssess/2025_Calibration/102025_FinalDC_npzs/Run2_DC_100125_48p5_24dB_90f_v16_wvfms.npz'],
        49.0: ['/global/cfs/cdirs/dune/users/ajwhite/2x2_LRS_DataAssess/2025_Calibration/102025_FinalDC_npzs/Run2_DC_100125_49_24dB_90f_v16_wvfms.npz']
        #43.0: ['/global/cfs/cdirs/dune/users/ajwhite/2x2_LRS_DataAssess/2025_Calibration/Run2_gamma_103025_43V_test_10dB_18f_v06_wvfms.npz'],
        #44.0: ['/global/cfs/cdirs/dune/users/ajwhite/2x2_LRS_DataAssess/2025_Calibration/Run2_gamma_103025_44V_test_10dB_18f_v12_wvfms.npz'],
        #45.0: ['/global/cfs/cdirs/dune/users/ajwhite/2x2_LRS_DataAssess/2025_Calibration/Run2_gamma_103025_45V_test_10dB_18f_v06_wvfms.npz'],
        #46.0: ['/global/cfs/cdirs/dune/users/ajwhite/2x2_LRS_DataAssess/2025_Calibration/Run2_gamma_103025_46V_test_10dB_18f_v06_wvfms.npz'],
        #46.7: ['/global/cfs/cdirs/dune/users/ajwhite/2x2_LRS_DataAssess/2025_Calibration/102025_FinalDC_npzs/Run2_DC_100125_45_24dB_90f_v16_wvfms.npz'],
        #47.0: ['/global/cfs/cdirs/dune/users/ajwhite/2x2_LRS_DataAssess/2025_Calibration/Run2_gamma_103025_47V_test_10dB_18f_v12_wvfms.npz']
    }

    initial_guess = { 
                51.0: [40, 100, 4.5e3, 4.5e3, 150, 160, 5.0e4, 3.5e4],
                51.5: [40, 100, 4.5e3, 4.5e3, 150, 160, 5.0e4, 3.5e4],
                52.0: [40, 100, 4.5e3, 4.5e3, 150, 160, 5.0e4, 3.5e4],
                52.5: [40, 100, 4.5e3, 4.5e3, 150, 160, 5.0e4, 3.5e4],
                53.0: [40, 100, 4.5e3, 4.5e3, 150, 160, 5.0e4, 3.5e4],
                53.5: [40, 100, 4.5e3, 4.5e3, 150, 160, 5.0e4, 3.5e4],
                54.0: [40, 100, 4.5e3, 4.5e3, 150, 160, 5.0e4, 3.5e4],
                54.5: [40, 100, 4.5e3, 4.5e3, 150, 160, 5.0e4, 3.5e4],
                55.0: [40, 100, 4.5e3, 4.5e3, 150, 160, 5.0e4, 3.5e4],
                55.5: [40, 100, 4.5e3, 4.5e3, 150, 160, 5.0e4, 3.5e4],
                56.0: [40, 100, 4.5e3, 4.5e3, 150, 160, 5.0e4, 3.5e4],
                56.5: [40, 100, 4.5e3, 4.5e3, 150, 160, 5.0e4, 3.5e4],
                57.0: [40, 100, 4.5e3, 4.5e3, 150, 160, 5.0e4, 3.5e4],
                57.5: [40, 100, 4.5e3, 4.5e3, 150, 160, 5.0e4, 3.5e4],
                58.0: [40, 100, 4.5e3, 4.5e3, 150, 160, 5.0e4, 3.5e4] 
    }

    try:

        #input_file = [f'/global/cfs/cdirs/dune/users/ajwhite/2x2_LRS_DataAssess/2025_Calibration/Co60_Files/Run2_10dB_SPE_{int(bias_voltage)}V_Co66{run}_{int(window)}tick_v01_wvfms.npz'] #inputs[bias_voltage]
        #input_file = [f'/exp/dune/app/users/hbagdu/2x2/2x2_LRS_OperatingScripts/SiPM_Operation_Commissioning/edit/step1_output/single_channel/step1_run1115_56V_p00001_SingleChannel.npz'] #inputs[bias_voltage]
        #input_file = [f'/exp/dune/app/users/hbagdu/2x2/2x2_LRS_OperatingScripts/SiPM_Operation_Commissioning/edit/step1_output/single_channel/step1_run1115_56V_p00019_SingleChannel.npz'] #inputs[bias_voltage]
        #input_file = [f'/exp/dune/app/users/hbagdu/2x2/2x2_LRS_OperatingScripts/SiPM_Operation_Commissioning/edit/step1_output/single_channel/step1_run1116_56V_p00001_SingleChannel.npz'] #inputs[bias_voltage]
        #input_file = [f'/exp/dune/app/users/hbagdu/2x2/2x2_LRS_OperatingScripts/SiPM_Operation_Commissioning/edit/step1_output/single_channel/step1_run1117_56V_p00001_SingleChannel.npz'] #inputs[bias_voltage]
        #input_file = [f'/exp/dune/app/users/hbagdu/2x2/2x2_LRS_OperatingScripts/SiPM_Operation_Commissioning/edit/step1_output/single_channel/allparts/step1_run1116_56V_SingleChannel_allparts.npz'] #inputs[bias_voltage]
        #input_file = [f'/exp/dune/app/users/hbagdu/2x2/2x2_LRS_OperatingScripts/SiPM_Operation_Commissioning/edit/step1_output/single_channel/allparts/step1_run1115_56V_SingleChannel_allparts.npz'] #inputs[bias_voltage]
        #input_file = [f'/exp/dune/app/users/hbagdu/2x2/2x2_LRS_OperatingScripts/SiPM_Operation_Commissioning/edit/step1_output/single_ADCs/ADC7/step1_run1115_56V_AllChannels_ADC7_allparts.npz'] #inputs[bias_voltage]
        #input_file = sorted(
        #    glob(
        #        "../step1_output/single_ADCs/ADC3/run1116_parts/"
        #        "step1_run1116_56V_ADC3_p*_max5000.npz"
        #    )
        #)
        #input_pattern = (f"../step1_output/single_ADCs/ADC{adc}/run{run}_parts/")
        #input_file = sorted(glob(input_pattern))
        with open(file_list, "r") as f:
            input_file = [line.strip() for line in f if line.strip()]
        print("Number of Step 1 NPZ files: ",len(input_file))

        for path in input_file:
            print(path)
        if len(input_file) == 0:
            raise FileNotFoundError("No Step 1 NPZ files were found!")

        #output_file = f'/global/cfs/cdirs/dune/users/ajwhite/2x2_LRS_DataAssess/2025_Calibration/Co60_Files/Run2_10dB_SPE_{int(bias_voltage)}V_Co66{run}_{int(window)}tick_v01_fitplots.pdf'
        #output_directory = (f"../step2_output/single_ADC/ADC{adc}")

        #output_file = (f'{output_directory}/step2_run{run}_{int(bias_voltage)}V_fitplots_ADC{adc}_AllChannels_allparts_maxPeak5000.pdf')

        #os.makedirs(output_directory, exist_ok=True)

        print("SPE Analysis at bias voltage ", bias_voltage)
        print("Initial guesses: ", initial_guess[bias_voltage])

        gain_values = analyze_spe(input_file, output_file, initial_guess[bias_voltage], adc_num=adc, window=window)
        #gain_output_file = (f'{output_directory}/step2_run{run}_{int(bias_voltage)}_V_fitplots_ADC{adc}_AllChannels_allparts_maxPek5000_gainvalues.npz')
        np.savez(gain_output, data=gain_values)

    except ValueError:
        print("ERROR")

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--file_list', required=True, type=str, \
                        help='''List containin one Step 1 NPZ path per line''')
    parser.add_argument('-o', '--output_file', required=True, type=str, \
                        help='''Output PDF file''')
    parser.add_argument('-g', '--gain_output', required=True, type=str, \
                        help='''Output NPZ file containint fitted fain values''')
    parser.add_argument('-a', '--adc', default=None, required=True, type=int, \
                        help='''int corresponding to ADC number to analyze''')
    parser.add_argument('-r', '--run', default=None, required=True, type=int, \
                        help='''int corresponding to light file run number''')
    parser.add_argument('-b', '--bias_voltage', default=None, required=False, type=float, \
                        help='''string corresponding to running bias voltage. Used for file management.''')
    parser.add_argument('-w', '--window', default=None, required=False, type=int, \
                        help='''int corresponding to number of ticks in signal window.''')
    args = parser.parse_args()
    main(**vars(args))

  
