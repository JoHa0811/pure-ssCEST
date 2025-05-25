#%%
import logging
from pathlib import Path
import numpy as np
import colorcet as cc
import matplotlib.pyplot as plt
import pandas as pd
from bmctool.utils.eval import plot_z
from bmctool.simulation import simulate

def setup_logging():
    """Set up logging for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(__name__)

logger = setup_logging()

def plot_magnetization_development(simulation_data, plot: bool = True, return_plot: bool = False):
    """
    Plot the magnetization development and return the plot object.

    Args:
        simulation_data: Simulation data containing t_dyn and m_dyn attributes.

    Returns:
        fig, ax: Matplotlib figure and axes objects.
    """
    try:
        logger.info("Creating magnetization development plot.")
        fig, ax = plt.subplots()
        ax.plot(simulation_data.t_dyn, simulation_data.m_dyn.T[:, 4], label="Magnetization Development")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Magnetization (m_z)")
        ax.legend()
        logger.info("Plot created successfully.")
        if plot:
            plt.show()
        if return_plot:
            return fig, ax
    except Exception as e:
        logger.error(f"Error while creating plot: {e}")
        raise e
    
#%%
def process_npy_files(folderpath):
    """
    Process all .npy files in the specified folder and extract offsets and m_z values.
    
    """
    file_names = []
    all_offsets = []
    all_m_z = []
    all_data = []
    
    for file in Path(folderpath).rglob("*.npy"):
        data = np.load(file, allow_pickle=True).item()
        file_names.append(file)
        offsets, m_z = data.get_zspec()
        all_offsets.append(offsets[1:])
        all_m_z.append(m_z[1:])
        all_data.append(data)
        #print(m_z[5])
    
    return all_offsets, all_m_z, file_names, all_data

def calculate_mtr_asymmetry_df(offsets, m_z):
    """
    Calculate MTR asymmetry as the difference between positive and negative offsets.
    
    Args:
        offsets (np.ndarray): Array of offset values.
        m_z (np.ndarray): Array of m_z values corresponding to the offsets.
        
    Returns:
        np.ndarray: Array of symmetric offsets.
        np.ndarray: Array of MTR asymmetry values.
    """
    # Calculate MTR asymmetry as the difference between positive and negative offsets
    mtr_asym = []
    symmetric_offsets = []
    for i in range(len(offsets)):
        if offsets[i] > 0:
            pos_index = i
            neg_index = np.where(offsets == -offsets[i])[0]
            if len(neg_index) > 0:
                neg_index = neg_index[0]
                mtr_asym.append(m_z[neg_index] - m_z[pos_index])
                symmetric_offsets.append(offsets[i])
    return np.array(symmetric_offsets), np.array(mtr_asym)
#%%
def plot_mtr_asym(df, color_dict):
    """
    Plot MTR asymmetry for each column in the DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame containing offset and m_z values.
        color_dict (dict): Dictionary mapping column names to colors.
        
    Returns:
        fig, ax: Matplotlib figure and axes objects.
    """
    
    fig, ax = plt.subplots()
    for i in range(len(df.columns)-1):
        offsets, mtr_asym = calculate_mtr_asymmetry_df(np.array(df["x data"]), np.array(df.iloc[:,i+1]))
        label = df.columns[i+1]
        ax.plot(offsets, mtr_asym, label=label, color=color_dict[df.columns[i+1]])
        ax.scatter(offsets, mtr_asym, color=color_dict[df.columns[i+1]])  
        
    ax.set_xlabel('offsets')
    ax.set_ylabel('MTR asymmetry')
    ax.set_ylim(-0.1, 0.5)
    #ax.legend(loc='lower right')
    ax.invert_xaxis()
    plt.show()
    
    return fig, ax

def plt_z_spectra(df, color_dict):
    """
    Plot z-spectra for each column in the DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame containing offset and m_z values.
        color_dict (dict): Dictionary mapping column names to colors.
        
    Returns:
        fig, ax: Matplotlib figure and axes objects.
    """
    
    fig, ax = plt.subplots()    
    for i in range(len(df.columns)-1):
        label = df.columns[i+1]
        ax.plot(df["x data"], df.iloc[:,i+1], label=label, color=color_dict[df.columns[i+1]])
        ax.scatter(df["x data"], df.iloc[:,i+1], color=color_dict[df.columns[i+1]])
    ax.set_xlabel('offsets')
    ax.set_ylabel('m_z')
    ax.invert_xaxis()
    ax.set_ylim(-0.1, 1.05)
    ax.legend(loc='lower right')
    plt.show()
    return fig, ax

#%%
def plot_z_spectra_from_sims(folderpath_sim_b1: str, folderpath_sim_sat: str):
    """
    Plot z-spectra from simulation data for different B1 power and saturation times.
    
    Args:
        folderpath_sim_b1 (str): Path to the folder containing B1 power simulation data.
        folderpath_sim_sat (str): Path to the folder containing saturation time simulation data.
        
    Returns:
        fig_b1, ax_b1: Matplotlib figure and axes objects for B1 power simulation.
        fig_sat, ax_sat: Matplotlib figure and axes objects for saturation time simulation.
    """
    
    color_dict_sat_times = {"25ms": "#8c3bff", "50ms": "#018700", "75ms": "#d60000", "100ms": "#00acc6"}
    color_dict_b1 = {"0p5": "#8c3bff", "0p8": "#018700", "1p2": "#d60000", "2p0": "#00acc6"}

    # load sim data for b1 power series and sat time series
    all_offsets_sim_b1, all_m_z_sim_b1, file_names_sim_b1, all_data_sim_b1 = process_npy_files(folderpath_sim_b1)
    all_offsets_sim_sat, all_m_z_sim_sat, file_names_sim_sat, all_data_sim_sat = process_npy_files(folderpath_sim_sat)

    # # create df for sim data
    # # b1
    df_sim_b1 = pd.DataFrame()
    df_sim_b1["x data"] = all_offsets_sim_b1[0]
    df_sim_b1["0p5"] = all_m_z_sim_b1[3]
    df_sim_b1["0p8"] = all_m_z_sim_b1[0]
    df_sim_b1["1p2"] = all_m_z_sim_b1[2]
    df_sim_b1["2p0"] = all_m_z_sim_b1[1]

    # # sat times
    df_sim_sat = pd.DataFrame()
    df_sim_sat["x data"] = all_offsets_sim_sat[0]
    df_sim_sat["25ms"] = all_m_z_sim_sat[2]
    df_sim_sat["50ms"] = all_m_z_sim_sat[0]
    df_sim_sat["75ms"] = all_m_z_sim_sat[3]
    df_sim_sat["100ms"] = all_m_z_sim_sat[1]
    
    fig_b1, ax_b1 = plt_z_spectra(df_sim_b1, color_dict_b1)
    fig_sat, ax_sat = plt_z_spectra(df_sim_sat, color_dict_sat_times)
    
    return fig_b1, ax_b1, fig_sat, ax_sat
#%%
