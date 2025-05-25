# %%
import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from mrpro.data import KData, SpatialDimension
from mrpro.data.traj_calculators import KTrajectoryIsmrmrd
from mrpro.algorithms.reconstruction import DirectReconstruction


def setup_logging():
    """Set up logging for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def reconstruct_file(file: Path):
    """
    Reconstruct a single HDF5 file into a NIfTI image.

    Args:
        file (Path): Path to the HDF5 file.
    """
    try:
        logging.info(f"Reconstructing: {file}")
        kdata = KData.from_file(file, KTrajectoryIsmrmrd())
        logging.info(f"KData shape: {kdata.data.shape}")

        nx, ny = kdata.header.recon_matrix.x, kdata.header.recon_matrix.y
        n_slices = kdata.data.shape[2]
        kdata.header.recon_matrix = SpatialDimension(n_slices, nx, ny)
        kdata.header.encoding_matrix = SpatialDimension(n_slices, nx, ny)

        images = []
        for offset in range(kdata.data.shape[0]):
            logging.info(f"Reconstructing Offset {offset + 1}/{kdata.data.shape[0]}")
            kdata_sub = kdata.select_other_subset(
                torch.tensor([offset]), subset_label="repetition"
            )
            reconstruction = DirectReconstruction(kdata_sub)
            img = reconstruction(kdata_sub)
            images.append(img)

        # Stack and save the reconstructed images
        image_stack = torch.stack([x.data for x in images]).squeeze(1).squeeze(1)
        ni_img = nib.Nifti1Image(
            np.abs(image_stack.cpu().permute(-1, -2, -3, -4).numpy()), affine=np.eye(4)
        )
        output_path = str(file).split(".")[0] + ".nii"
        nib.save(ni_img, output_path)
        logging.info(f"Saved reconstructed image to: {output_path}")

    except Exception as e:
        logging.error(f"Error reconstructing {file}: {e}")


def reconstruct_folder(folderpath: str):
    """
    Process all HDF5 files in the specified folder.

    Args:
        folderpath (str): Path to the folder containing HDF5 files.
    """
    folder = Path(folderpath)
    if not folder.exists():
        logging.error(f"Folder does not exist: {folderpath}")
        return

    for file in folder.rglob("*_traj.h5"):
        output_file = Path(f"{str(file).split('.h5')[0]}.nii")
        if not output_file.exists():
            reconstruct_file(file)
        else:
            logging.info(f"Reconstruction already exists: {output_file}")


def main():
    """Main function to parse arguments and start processing."""
    parser = argparse.ArgumentParser(
        description="Reconstruct HDF5 trajectory files into NIfTI images."
    )
    parser.add_argument(
        "folderpath",
        type=str,
        help="Path to the folder containing HDF5 trajectory files.",
    )
    args = parser.parse_args()

    reconstruct_folder(args.folderpath)


if __name__ == "__main__":
    setup_logging()
    main()
