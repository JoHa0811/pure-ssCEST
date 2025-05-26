#%%
import requests
import matplotlib.pyplot as plt
import colorcet as cc
from visualize import plot_mtr_asym, calculate_mtr_asymmetry_df
from reconstruct_data import reconstruct_folder, reconstruct_file
import os

#%%
# reconstruct the data
# access data from Zenodo

record_id = '15498440'
api_url = f"https://zenodo.org/api/records/{record_id}"
response = requests.get(api_url)
data = response.json()

#%%
# Create a folder to store the downloaded files
output_dir = f"zenodo_data"
os.makedirs(output_dir, exist_ok=True)

# Download all files
for file in data['files']:
    filename = file['key']
    url = file['links']['self']
    if filename.endswith('with_traj.h5'):
        print(f"Downloading {filename}...")

        response = requests.get(url)
        file_path = os.path.join(output_dir, filename)

        if os.path.exists(file_path):
            print(f"Already exists: {file_path} — skipping download.")
            continue
        
        with open(file_path, 'wb') as f:
            f.write(response.content)

        print(f"Saved to {file_path}")

#%%

# reconstruct data
reconstruct_folder(folderpath=output_dir)

#%%