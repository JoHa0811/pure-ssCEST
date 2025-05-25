#%%
from simulate_data import simulate_file_mag_development
from visualize import plot_magnetization_development
from bmctool.utils.eval import plot_z


## simulate mixed-ss and pure-ss sequences on Barbituric Acid phantom
mixed_ss_filepath = "./pulseq_sequences/fig2_longitudinal_magnetisation/20250328_BACEST_spiral_mixed_ss_256mm_1227k0_1slices_12interleaves_0p1ms_1.2uT_12spirals.seq"
pure_ss_filepath = "./pulseq_sequences/fig2_longitudinal_magnetisation/20250328_BA_simulationCEST_spiral_multi_2D_256mm_1227k0_1slices_12interleaves_0p1ms_1.2uT_12spirals_10angle.seq"

config_file_path = "./simulation_phantoms/barbituric_acid_3T_bmsim.yaml"

#%%
mixed_ss_sim = simulate_file_mag_development(mixed_ss_filepath, config_file_path, return_simulated_data=True, plot_z_spectra=True, show_plot=True, store_dynamics=2)
pure_ss_sim = simulate_file_mag_development(pure_ss_filepath, config_file_path, return_simulated_data=True, plot_z_spectra=False, show_plot=False, store_dynamics=2)
# %%
## plot magnetization development
fig_mixed_ss, ax_mixed_ss = plot_magnetization_development(mixed_ss_sim, return_plot=True, plot=True)
fig_pure_ss, ax_pure_ss = plot_magnetization_development(pure_ss_sim, return_plot=True, plot=True)
# %%
