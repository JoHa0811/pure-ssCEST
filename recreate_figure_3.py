#%%
from simulate_data import  process_folder
## simulate pure-ss sequence on Barbituric Acid phantom with different B1 power and saturation times
saturation_time_series = "./pulseq_sequences/fig3_zSpectra_simulation/saturation_time_series/"
b1_power_series = "./pulseq_sequences/fig3_zSpectra_simulation/b1_power_series"

config_file_path = "./simulation_phantoms/barbituric_acid_3T_bmsim.yaml"

#%%
sat_time_series_sim = process_folder(saturation_time_series, config_file_path, return_simulated_data=True, plot_z_spectra=True, show_plot=True, store_dynamics=0)
b1_power_series_sim = process_folder(b1_power_series, config_file_path, return_simulated_data=True, plot_z_spectra=False, show_plot=False, store_dynamics=0)
# %%
## plot z-Spectra
folderpath_sim_b1 = "./pulseq_sequences/fig3_zSpectra_simulation/b1_power_series/"
folderpath_sim_sat = "./pulseq_sequences/fig3_zSpectra_simulation/saturation_time_series/"
#%%