from mp_api.client import MPRester
import pandas as pd
import numpy as np
import pymatgen
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from pymatgen.electronic_structure.plotter import BSPlotter
import emmet
from emmet.core.summary import HasProps
from emmet.core.mpid import MPID
import scipy.interpolate as inter
import time
import argparse
import sys

parser = argparse.ArgumentParser(description='Near-Fermi Band Flatness Calculation')
parser.add_argument('--api', type=str, help="Materials Project API key")
parser.add_argument("--freq", type=int, required=False, help="Save checkpoint of flatness data (CSV and plot) every n structures")
parser.add_argument("--fermi", type=float, nargs="*", required=False, help="Energy range(s) (eV) from Fermi level to treat as 'near-Fermi'")
parser.add_argument("--mpid", type=str, help="Filepath to csv containing MPIDs to query")
parser.add_argument("--csv_dest", type=str, help="Filename for saving flatness scores")
parser.add_argument("--img_dest", type=str, required=False, help="Filepath to folder for saving flatness plots (if excluded, no plots generated)")
parser.add_argument("--plot_range", type=float, required=False, help="Energy range from Fermi level to consider when plotting band structure. Must be one of the ranges passed to `--fermi`")

args = parser.parse_args(sys.argv[1:])

def main():
    global args

    api_key = args.api
    mpids = pd.read_csv(args.mpid, header=None).iloc[:, 0].to_numpy()
    checkpoint = args.freq if args.freq else len(mpids)//10
    fermi_window = [*args.fermi] if args.fermi else [1, 0.75, 0.5, 0.25, 0.1]
    csv_dest = args.csv_dest
    img_dest = args.img_dest if args.img_dest else None
    plot_bands = img_dest is not None
    plot_window = args.plot_range

    if plot_window:
        assert plot_window in fermi_window, "Argument `--plot_range` must one of ranges passed to `--fermi`"

    
    def band_flatness(bs, fermi_window=[1, 0.75, 0.5, 0.25, 0.1], plot_band=False, plot_window=None, mpid=None):
        """Computes flatness of the flattest band among all bands within `fermi_window` eV of the Fermi level
        Flatness is defined as the standard deviation of the residual between the band (B-spline interpolation of k-points from DFT calculation) and the mean energy of the band
        A perfectly horizontal band has flatness 0; bands with frequent and/or large oscillations have high flatness scores. Stuctures with no bands within `fermi_window` are assigned flatness = infinity
        
        @param bs (pymatgen.electronic_structure.bandstructure.BandStructureSymmLine): Band structure to evaluate
        @param fermi_window (float or lst of floats): Energy range (in eV) from the Fermi level considered "near-Fermi"
        @param plot_band (bool): If True, plots the band diagram with the flattest band highlighted and annotated with flatness score
        @param plot_window (float, optional): Energy range from the Fermi level to plot when generating band diagram. Must be one of the ranges passed to `fermi_window`
        @param mpid (str, optional): Used to annotate band diagram with the MPID it represents

        @returns spin_flatness (float or lst of floats): Flatness scores for each `fermi_window` considered. Takes min flatness across all bands and spin states within the `fermi_window`
        """
        bs_helper = BSPlotter(bs)
        data = bs_helper.bs_plot_data(zero_to_efermi=True) # All bands are expressed relative to E_f = 0
        distances = data["distances"] # Distance scale for MP band structure k-points
        if type(fermi_window)==int:
            # If a single Fermi energy range is passed
            spin_flatness = np.inf
            for spin in bs.bands:
                energies = data["energy"][str(spin)]
                interp_distances, interp_energies = bs_helper._interpolate_bands(distances, energies)
                interp_energies = np.hstack(interp_energies)
                near_fermi = interp_energies[np.any((interp_energies >= -fermi_window) & (interp_energies <= fermi_window), axis=1), :]
                if len(near_fermi) == 0:
                    # No bands lie in the allowed energy range from the Fermi level; assign a flatness score of infinity
                    continue
                flatness = np.min(np.std(near_fermi, axis=1)) # Consider the flattest band of all bands in the Fermi range
                spin_flatness = min(spin_flatness, flatness)
                if spin_flatness == flatness and plot_band: 
                    plot_distances, plot_energies = np.hstack(interp_distances), near_fermi
                    flattest_idx = np.argmin(np.std(near_fermi, axis=1))
                    flattest = plot_energies[flattest_idx, :]
                    plot_energies = np.delete(plot_energies, flattest_idx, axis=0)
            if plot_band:
                plt.figure()
                ax = plt.subplot()
                for band in plot_energies:
                    ax.plot(plot_distances, band, c="tab:blue")
                ax.plot(plot_distances, flattest, c="tab:red")
                ax.set_ylabel("E")
                ax.set_title(f"{str(mpid) if mpid else ''}\nFlatness = {spin_flatness}")
                ax.set_ylim(-1.5, 1.5)
                ax.savefig(f"{img_dest}/{mpid}_flatness_{fermi_window}_eV_from_fermi")
            return spin_flatness
        else:
            # If several Fermi energy ranges are passed
            spin_flatness = np.inf*np.ones(len(fermi_window))
            if plot_window:
                plot_idx = fermi_window.index(plot_window)
            else:
                plot_idx = 0
            for spin in bs.bands:
                energies = data["energy"][str(spin)]
                interp_distances, interp_energies = bs_helper._interpolate_bands(distances, energies)
                interp_energies = np.hstack(interp_energies)
                for i, window in enumerate(fermi_window):
                    near_fermi = interp_energies[np.any((interp_energies >= -window) & (interp_energies <= window), axis=1), :]
                    if len(near_fermi) == 0:
                        # No bands lie in the allowed energy range from the Fermi level; assign a flatness score of infinity
                        continue
                    flatness = np.min(np.std(near_fermi, axis=1))
                    spin_flatness[i] = min(spin_flatness[i], flatness)
                    if spin_flatness[plot_idx] == flatness and i == plot_idx and plot_band:
                        plot_distances, plot_energies = np.hstack(interp_distances), near_fermi
                        flattest_idx = np.argmin(np.std(near_fermi, axis=1))
                        flattest = plot_energies[flattest_idx, :]
                        plot_energies = np.delete(plot_energies, flattest_idx, axis=0)
            if plot_band:
                fig = plt.figure()
                ax = plt.subplot()
                for band in plot_energies:
                    ax.plot(plot_distances, band, c="tab:blue")
                ax.plot(plot_distances, flattest, c="tab:red")
                ax.set_ylabel("E")
                ax.set_title(f"{str(mpid) if mpid else ''}\nFlatness = {spin_flatness[plot_idx]}")
                ax.set_ylim(-1.5, 1.5)
                fig.savefig(f"{img_dest}/{mpid}_flatness_{plot_window}_eV_from_fermi.png")
            return spin_flatness
        
    flatness_vals = np.zeros((len(mpids), len(fermi_window)))
    computed_mpids = []
    failed_mpids = []
    start = time.time()
    print("Start calculation")

    with MPRester(api_key) as mpr:
        for i, mpid in enumerate(mpids):
            if i%checkpoint == 0 and i > 0:
                    # Save a checkpoint copy of the flatness scores and plot the band structure of the most recent MPID for sanity checking
                    print("*"*10, f" Structure {i}/{len(mpids)} ", "*"*10)
                    print(f"Time elapsed: {np.round((time.time() - start)/60, 3)} minutes")
                    pd.DataFrame(flatness_vals[:i, :], index=mpids[:i], columns=[f"{window}_eV" for window in fermi_window]).to_csv(csv_dest)
                    plot_band = plot_bands
            else:
                    plot_band = False
            try:
                bs = mpr.get_bandstructure_by_material_id(mpid)  
                flatness_vals[i, :] = band_flatness(bs, fermi_window=fermi_window, plot_band=plot_band, plot_window=plot_window, mpid=mpid)
                computed_mpids.append(mpid)
            except:
                # Some Materials Project band structures are misformatted and trigger a Unicode error if accessed
                print(f"Corrupted MP band structure for {mpid}, skipping flatness calculation")
                flatness_vals[i, :] = [np.nan]*len(fermi_window)
                failed_mpids.append(mpid)

    flatness_df = pd.DataFrame(flatness_vals, index=mpids, columns=[f"{window}_eV" for window in fermi_window])

    flatness_df.to_csv(csv_dest)


if __name__ == '__main__':
    main()





