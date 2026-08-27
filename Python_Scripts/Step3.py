import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import glob
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from matplotlib.colors import LogNorm, TwoSlopeNorm


sipm_channels = ([4,5,6,7,8,9] + \
                [10,11,12,13,14,15] + \
                [20,21,22,23,24,25] + \
                [26,27,28,29,30,31] + \
                [36,37,38,39,40,41] + \
                [42,43,44,45,46,47] + \
                [52,53,54,55,56,57] + \
                [58,59,60,61,62,63]) # Active channels

# Aggregate data for different voltages
def agg_data(file_names, voltages):

    # Check if the values provided for voltage match the file names (in number) # Might add check for correct values
    if len(file_names) != len(voltages):
        raise ValueError("File names and provided voltages do not match in length")
        return

    gains = np.zeros((len(file_names),8,64,3)) # First axis is bias voltages
    for idx, file_name in enumerate(file_names):

        data = np.load(file_name)
        gains[idx,...] = data['data']

    return gains

# Check chi squared value
def check_chisquared(array, sipm_channels):
    """ Provide a numpy array of shape (V,8,64,2) for (n_voltages, n_adc, n_ch, dim2) with in the second dimension (gain, chi squared)"""
    gains = array[...,0]
    chisquared = array[...,1]

    # Loop over active channels and count the fits that are between 0.5 and 1.5 for each channel
    for adc in range(0, gains.shape[1]):

        for ch in range(0, gains.shape[2]):
            if ch not in sipm_channels:
                pass
            else:
                #print(f"ADC {adc}, ch {ch}, {chisquared[:,adc,ch]}")
                
                if sum(0.5 <= x <= 1.5 for x in chisquared[:,adc,ch]) >= 4:
                    print(adc, ch)
                

    return

# Fitting function
def linear_fit(gains, voltages, pdf_filename, plot=False, verbose=False):
    """ WLS fit based on reduced chi squared values obtained from the multiple Gaussian model."""
    sipm_channels = ([4,5,6,7,8,9] + \
                     [10,11,12,13,14,15] + \
                     [20,21,22,23,24,25] + \
                     [26,27,28,29,30,31] + \
                     [36,37,38,39,40,41] + \
                     [42,43,44,45,46,47] + \
                     [52,53,54,55,56,57] + \
                     [58,59,60,61,62,63]) # Active channels
    
    voltages = np.array(voltages)
    fit_res = np.zeros((8,64,2)) # Store the coefficients of the linear fit per adc per channel here
    r_squared = np.zeros((8,64)) # Coefficient of determination
    pvalues = np.zeros((8,64,2)) # Check if coefficients are significantly different from zero
    bdv_value = np.zeros((8,64))
   
    plots_per_fig = 4
    no_fit = 0
    with PdfPages(pdf_filename) as pdf_output:
        for adc in range(0, 8):
        #for adc in range(5,6):

            for ch in range(0,gains.shape[2]):
                print('Current channel', ch)
                if ch not in sipm_channels: 
                    pass # Skip inactive channels

                else: # Fit for each channel separately, because the nr of good fits varies per channel
                    gains_per_chan = gains[:,adc,ch,:] # 2D array with gain+stats for all voltages (voltages, gains/stats)
                    print(gains_per_chan)
                    if np.sum(gains_per_chan) == 0:
                        if verbose:
                            print(f"No values available for ADC {adc} ch {ch}")

                    else:
                        cut_1 = (gains_per_chan[:,0] !=0) # Take out zero gains bc those are failed/bad fits
                        cut_2 = (gains_per_chan[:,1] < 1.5) & (gains_per_chan[:,1] > .5) # We only want to look at fitted gains that had a 'good' chi squared value
                        cut_combi =  cut_1 #* cut_2
                        gains_per_chan_v1 = gains_per_chan[cut_combi] 
                        if (gains_per_chan[~cut_combi].size != 0) and verbose:
                            print(f"We don't take these values into account for ADC {adc} ch {ch} \n", gains_per_chan[~cut_combi], "\n At voltages ", voltages[~cut_combi])

                        voltages_v1 = voltages[cut_combi]

                        cut_3 = gains_per_chan_v1.size > 0  # Skip arrays that are empty after our first cut
                        if cut_3 == 1:

                            # Now check if for increasing voltages, the gain decreases. If that happens, something went wrong and we don't want to use that value in the fit.
                            gains_per_chan_v2 = np.array([gains_per_chan_v1[0]]) # Store the first gain value in a new array
                            voltages_v2 = np.array(voltages_v1[0])

                            for idx_v in range(1,len(voltages_v1)): # Start checking from the second value
                                if gains_per_chan_v1[idx_v,0] < gains_per_chan_v2[-1,0]: # Skip if the gain is lower than the last one that passed the cut
                                    if verbose:
                                        print(f"Probably {voltages_v1[idx_v]} is a bad fit for ADC {adc} ch {ch}")
                                        print(f"Gain for {voltages_v1[idx_v-1]}V: {gains_per_chan_v1[idx_v-1,0]}")
                                        print(f"Gain for {voltages_v1[idx_v]}V: {gains_per_chan_v1[idx_v,0]}")
                                else:
                                    gains_per_chan_v2 = np.vstack([gains_per_chan_v2,gains_per_chan_v1[idx_v,:]]) # Collect approved gain values
                                    voltages_v2 = np.append(voltages_v2, voltages_v1[idx_v])
                        
                            # Temporarily remove monotone inscrease requirement
                            gains_per_chan_v2 = gains_per_chan_v1
                            voltages_v2 = voltages_v1
                            cut_4 = (len(gains_per_chan_v2) > 1)  # Only attempt a fit for 3+ datapoints
                            if cut_4 == 1:
                                try:
                                    # Weighted least squares with chi2
                                    volt_w_const = sm.add_constant(voltages_v2)
                                    weights = 1/(gains_per_chan_v2[:,2]**2)
                                    model_WLS = sm.WLS(gains_per_chan_v2[:,0], volt_w_const, weights=weights)
                                    res = model_WLS.fit()


                                    fit_res[adc, ch, :] = res.params

                                    r_squared[adc, ch] = res.rsquared
                                    pvalues[adc, ch,:] = res.pvalues
                                except:
                                    if verbose:
                                        print('some sort of issue')

                            else:
                                if verbose:
                                    print(f"No fit for adc {adc} channel {ch}")
                
                    # For plotting
                    if plot:
                        
                                        
                        if ch % plots_per_fig == 0: # Creates a new figure for every 4 channels

                            fig, axs = plt.subplots(2,2, figsize=(10,10))
                            axs = axs.flatten()

                        ax = axs[ch % plots_per_fig]
                        if np.sum(fit_res[adc, ch, :]) != 0: # If the model results are zero, no fit was made
                            x_fit = np.linspace(48,60,100)
                            y_fit = res.params[1]*x_fit + res.params[0]
                            ax.set_title(f"ADC {adc} Channel {ch}, $V_{{BD}}$={round(-1*res.params[0]/res.params[1],2)}")
                            bdv_value[adc,ch] = round(-1*res.params[0]/res.params[1],2)
                            ax.plot(x_fit, y_fit,linestyle='--', label=f"fit of ADC {adc} ch {ch}, $R^2$={round(res.rsquared,3)}")
                            ax.errorbar(voltages_v2, gains_per_chan_v2[:,0], yerr=gains_per_chan_v2[:,2],color='black', fmt='x', capsize=3,label=f"ADC {adc} ch {ch}")
                            ax.set_xlabel("$V_{bias}$")
                            ax.set_ylabel("Gain")                                    
                            ax.legend()
                            if (adc == 0) or (adc == 1):
                                ax.set_ylim(0,1500)
                            else:
                                ax.set_ylim(0, 4000)
                            ax.set_xlim(48,60)
                        
                        else: 
                            ax.text(0.5, 0.5, f"No WLS fit possible for ADC {adc} ch {ch}", ha='center', va='center')
                            no_fit += 1
                        # Saves the figure per 4 channels
                        if ch % plots_per_fig == (plots_per_fig-1) or (ch == 63): 
                            fig.tight_layout()    
                            pdf_output.savefig(fig)
                            plt.close(fig)            
    np.savez("fit_res_Co60_v01_10dB_50tick.npz", bdv_value=bdv_value) #TODO: Come up with a better name lol
    print("Number of channels for which the linear fit failed is: ", no_fit)
    return fit_res, r_squared, pvalues

# Load Data hanconet
#file_list = ["/exp/dune/app/users/hanconet/2x2_LRS_OperatingScripts/SiPM_Operation_Commissioning/myoutput/001128.npz", "/exp/dune/app/users/hanconet/2x2_LRS_OperatingScripts/SiPM_Operation_Commissioning/myoutput/001129.npz", "/exp/dune/app/users/hanconet/2x2_LRS_OperatingScripts/SiPM_Operation_Commissioning/myoutput/001130.npz", "/exp/dune/app/users/hanconet/2x2_LRS_OperatingScripts/SiPM_Operation_Commissioning/myoutput/001131.npz", "/exp/dune/app/users/hanconet/2x2_LRS_OperatingScripts/SiPM_Operation_Commissioning/myoutput/001132.npz", "/exp/dune/app/users/hanconet/2x2_LRS_OperatingScripts/SiPM_Operation_Commissioning/myoutput/001133.npz", "/exp/dune/app/users/hanconet/2x2_LRS_OperatingScripts/SiPM_Operation_Commissioning/myoutput/001134.npz", "/exp/dune/app/users/hanconet/2x2_LRS_OperatingScripts/SiPM_Operation_Commissioning/myoutput/001135.npz"]
#TODO: Need to edit this to point towards real file list from step2.py
file_list = sorted(glob.glob("/exp/dune/app/users/hanconet/2x2_LRS_OperatingScripts/SiPM_Operation_Commissioning/myoutputv2/0011*.npz")) #
print(file_list)

#voltages = [51.0, 51.5, 52.0, 52.5, 53.0, 53.5, 54, 54.5, 55, 55.5, 56, 56.5, 57.0, 57.5, 58.0]
#Runs 1124 (52.5V) and 1125 (53.0V) not working do to statistics issues (I think) with the peak finder
#voltages = [51.0, 51.5, 52.0, 53.5, 54, 54.5, 55, 55.5, 56, 56.5, 57.0, 57.5, 58.0]
voltages = [54.5, 55, 55.5, 56, 56.5, 57.0, 57.5, 58.0]
gains = agg_data(file_list[1:], voltages[1:])

# Check chi squared requirement 
#check_chisquared(gains, sipm_channels)

# Do the linear fit
file_name = 'mytest.pdf'
fit_res, _,_ = linear_fit(gains, voltages[1:], file_name, plot=True, verbose=False)