#%%
import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from bmctool.utils.eval import plot_z
from bmctool.simulation import simulate

def setup_logging():
    """Set up logging for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

def simulate_file_mag_development(file: Path, config_file: Path, return_simulated_data: bool = False, show_plot: bool = True, plot_z_spectra: bool = True ,store_dynamics: int = 2):
    """
    Simulate and process a single .seq file.
    
    Args:
        file (Path): Path to the .seq file.
        config_file (Path): Path to the simulation configuration file.
    """
    # try:
    logging.info(f"Processing: {file}")
    simulated_returns_ss = simulate(
        config_file=config_file,
        seq_file=file,
        show_plot=show_plot,
        store_dynamics=store_dynamics,
    )
    offsets, m_z = simulated_returns_ss.get_zspec()
    if plot_z_spectra:
        plot_z(m_z=m_z, offsets=offsets)
        plt.plot(simulated_returns_ss.t_dyn, simulated_returns_ss.m_dyn.T[:, 4])
        plt.show()
    
    np.save(Path(file).with_suffix(".npy"), simulated_returns_ss, allow_pickle=True)
    logging.info(f"Saved simulation results: {Path(file).with_suffix('.npy')}")

    if return_simulated_data:
        return simulated_returns_ss
    
    # except Exception as e:
    #     logging.error(f"Error processing {file}: {e}")
        
    
#%%
def process_folder(folderpath: str, config_file_path: str, return_simulated_data = False, show_plot = True, plot_z_spectra = True ,store_dynamics = 2):
    
    folder_sims = []
    
    """
    Process all .seq files in the specified folder.
    
    Args:
        folderpath (str): Path to the folder containing .seq files.
        config_file_path (str): Path to the simulation configuration file.
    """
    folder = Path(folderpath)
    config_file = Path(config_file_path)
    
    if not folder.exists():
        logging.error(f"Folder does not exist: {folderpath}")
        return
    
    if not config_file.exists():
        logging.error(f"Config file does not exist: {config_file_path}")
        return
    
    for file in folder.rglob("*.seq"):
        folder_sims.append(simulate_file_mag_development(file, config_file, return_simulated_data, show_plot, plot_z_spectra, store_dynamics))
        logging.info(f"Processed file: {file}")
        
    return folder_sims
#%%
def main():
    """Main function to set paths and start processing."""
    folderpath = "/echo/hammac01/SpiralssCEST/pulseq_sequences/fig2_longitudinal_magnetisation/"
    config_file_path = "/echo/hammac01/SpiralssCEST/simulation_phantoms/barbituric_acid_3T_bmsim.yaml"
    
    process_folder(folderpath, config_file_path)

if __name__ == "__main__":
    setup_logging()
    main()

# %%
