#%%
#from reconstruct_data import reconstruct_folder, reconstruct_file
from simulate_data import process_folder, simulate_file_mag_development
from visualize import plot_magnetization_development
from bmctool.utils.eval import plot_z


## simulate mixed-ss and pure-ss sequences on Barbituric Acid phantom
mixed_ss_filepath = "./pulseq_sequences/20250317_BA_simulationCEST_spiral_multi_2D_256mm_1227k0_1slices_10interleaves_0p1ms_0.5uT_12spirals_10angle.seq"
pure_ss_filepath = "./pulseq_sequences/20250317_BACEST_spiral_mixed_ss_256mm_1227k0_1slices_10interleaves_0p1ms_0.8uT_12spirals.seq"

config_file_path = "./simulation_phantoms/barbituric_acid_3T_bmsim.yaml"

#%%
mixed_ss_sim = simulate_file_mag_development(mixed_ss_filepath, config_file_path, return_simulated_data=True, plot_z=True, show_plot=True)
pure_ss_sim = simulate_file_mag_development(pure_ss_filepath, config_file_path, return_simulated_data=True, plot_z=False, show_plot=False)
# %%
## plot magnetization development
fig_mixed_ss, ax_mixed_ss = plot_magnetization_development(mixed_ss_sim, return_plot=True, plot=True)
fig_pure_ss, ax_pure_ss = plot_magnetization_development(pure_ss_sim, return_plot=True, plot=True)
# %%
