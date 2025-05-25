import requests
import matplotlib.pyplot as plt
import colorcet as cc
from visualize import plot_mtr_asym, calculate_mtr_asymmetry_df
from reconstruct_data import reconstruct_folder, reconstruct_file

#%%
# reconstruct the data
# access data from Zenodo

record_id = '15489482'
api_url = f"https://zenodo.org/api/records/{record_id}"
response = requests.get(api_url)
data = response.json()

# List all files
for file in data['files']:
    print(file['key'], ":", file['links']['self'])

# load data


# reconstruct the data

# visualize the data

#%%