from mp_api.client import MPRester
import pandas as pd
import numpy as np
import pymatgen
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from pymatgen.electronic_structure.plotter import BSPlotter
from pymatgen.electronic_structure.bandstructure import BandStructureSymmLine
import emmet
from emmet.core.summary import HasProps
from emmet.core.mpid import MPID
import scipy.interpolate as inter
from itertools import cycle 
import pickle

def get_bands(bs, plot=False):
    bs_helper = BSPlotter(bs)
    data = bs_helper.bs_plot_data(zero_to_efermi=True) # All bands are expressed relative to E_f = 0
    distances = data["distances"] # Distance scale for MP band structure k-points
    all_energies, all_distances, all_spins = None, None, None
    for spin in bs.bands:
        energies = data["energy"][str(spin)]
        interp_distances, interp_energies = bs_helper._interpolate_bands(distances, energies)
        if all_energies is None:
            all_distances, all_energies = np.hstack(interp_distances), np.hstack(interp_energies)
            all_spins = int(spin)*np.ones(len(all_energies))
        else:
            all_energies = np.vstack([all_energies, np.hstack(interp_energies)])
            all_spins = np.hstack([all_spins, int(spin)*np.ones(len(all_energies))])
    if plot:
        return all_distances, all_energies, all_spins, (data["ticks"]["distance"], data["ticks"]["label"])
    else:
        return all_distances, all_energies, all_spins

def plot_bands(distances, energies, spins, tick_data, annotate=None, ylim=(-1.5, 1.5), title=None, img_dest=None):
    if annotate:
        indices = [band[0] for band in annotate]
        labels = [band[1] for band in annotate]
        annot_colors = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'][1:])
        for i in range(2):
            next(annot_colors) #I like starting with red ;)
    fig = plt.figure()
    ax = plt.subplot()
    for i, band in enumerate(energies):
        if not annotate or i not in indices:
            ax.plot(distances, band, c="tab:blue", linestyle="-" if spins[i]==1 else "--")
        elif annotate and i in indices:
            ax.plot(distances, band, label=labels[indices.index(i)], c=next(annot_colors), linestyle="-" if spins[i]==1 else "--")
    ax.set_ylim(*ylim)
    ax.set_xlim(distances[0], distances[-1])
    ax.set_xticks(tick_data[0])
    ax.set_xticklabels(tick_data[1])
    ax.set_title(f"{title if title else ''}")
    ax.set_ylabel(r"$E - E_F$ (eV)")
    if annotate:
        plt.legend(loc="lower right")
    if img_dest:
        fig.savefig(img_dest)
    plt.close()

def near_fermi(energies, fermi_window):
    return np.where(np.any((energies >= -fermi_window) & (energies <= fermi_window), axis=1))

def intersects_fermi(energies):
    return np.any(energies > 0, axis=1)&np.any(energies < 0, axis=1)

def compute_flatness(energies):
    return np.std(energies, axis=1)

def compute_bandwidth(energies):
    return np.max(energies, axis=1) - np.min(energies, axis=1)

def score_variation(scores):
    return np.std(scores)

def score_range(scores):
    return np.max(scores) - np.min(scores)

def is_flat_steep_system(energies, fermi_window, metric, threshold_percent):
    assert metric == "flatness" or metric == "bandwidth", "Allowable scoring metrics are 'flatness' or 'bandwidth'"
    if metric == "flatness":
        all_scores = compute_flatness(energies)
        near_fermi_idx = near_fermi(energies, fermi_window)
        bands_near_fermi = energies[near_fermi_idx]
        window_scores = compute_flatness(bands_near_fermi)
    elif metric == "bandwidth":
        all_scores = compute_bandwidth(energies)
        near_fermi_idx = near_fermi(energies, fermi_window)
        bands_near_fermi = energies[near_fermi_idx]
        window_scores = compute_bandwidth(bands_near_fermi)
    flat_threshold = threshold_percent*np.mean(all_scores)
    has_flat_band = np.any(window_scores <= flat_threshold)
    has_dispersive_band = np.any(intersects_fermi(energies[all_scores > flat_threshold])) #flat band can't count as the dispersive band!
    return has_flat_band and has_dispersive_band

def query_band_structure(mpid, api_key):
    with MPRester(api_key) as mpr:
        bs = mpr.get_bandstructure_by_material_id(mpid)  

    # Directly copied from pymatgen repo
    # By some voodoo magic, the `as_dict` method fails when called from library but runs fine if manually inserted into code
    dct = {
        "@module": type(bs).__module__,
        "@class": type(bs).__name__,
        "lattice_rec": bs.lattice_rec.as_dict(),
        "efermi": bs.efermi,
        "kpoints": [],
    }
    for k in bs.kpoints:
        dct["kpoints"].append(k.as_dict()["fcoords"])

    dct["bands"] = {str(int(spin)): bs.bands[spin].tolist() for spin in bs.bands}
    dct["is_metal"] = bs.is_metal()
    vbm = bs.get_vbm()
    dct["vbm"] = {
        "energy": vbm["energy"],
        "kpoint_index": vbm["kpoint_index"],
        "band_index": {str(int(spin)): vbm["band_index"][spin] for spin in vbm["band_index"]},
        "projections": {str(spin): v.tolist() for spin, v in vbm["projections"].items()},
    }
    cbm = bs.get_cbm()
    dct["cbm"] = {
        "energy": cbm["energy"],
        "kpoint_index": cbm["kpoint_index"],
        "band_index": {str(int(spin)): cbm["band_index"][spin] for spin in cbm["band_index"]},
        "projections": {str(spin): v.tolist() for spin, v in cbm["projections"].items()},
    }
    dct["band_gap"] = bs.get_band_gap()
    dct["labels_dict"] = {}
    dct["is_spin_polarized"] = bs.is_spin_polarized

    # MongoDB does not accept keys starting with $. Add a blank space to fix the problem
    for c, label in bs.labels_dict.items():
        mongo_key = c if not c.startswith("$") else f" {c}"
        dct["labels_dict"][mongo_key] = label.as_dict()["fcoords"]
    dct["projections"] = {}
    if len(bs.projections) != 0:
        dct["structure"] = bs.structure.as_dict()
        dct["projections"] = {str(int(spin)): np.array(v).tolist() for spin, v in bs.projections.items()}
    return bs, dct

def characterize_bands(mpid, api_key, fermi_windows, plot_window, img_dest=None):
    # bs, band_dict = query_band_structure(mpid, api_key)
    with MPRester(api_key) as mpr:
        bs = mpr.get_bandstructure_by_material_id(mpid) 
    assert plot_window in fermi_windows, "`plot_window` must be one of the `fermi_windows` being analyzed!"
    plot_idx = fermi_windows.index(plot_window)
    k_point_distances, band_energies, spins, ticks = get_bands(bs, plot=True)
    data = np.zeros(18*len(fermi_windows))
    for i, fermi_window in enumerate(fermi_windows):
        fermi_window_idx = near_fermi(band_energies, fermi_window)
        fermi_window_energies, fermi_window_spins = band_energies[fermi_window_idx], spins[fermi_window_idx]
        if len(fermi_window_energies) == 0:
            # no bands lie in Fermi window
            data[18*i:18*(i+1)] = [np.inf]*18
            continue
        near_fermi_flatnesses, near_fermi_bandwidths = compute_flatness(fermi_window_energies), compute_bandwidth(fermi_window_energies)
        is_flat_steep_flatness, is_flat_steep_bandwidth = is_flat_steep_system(band_energies, fermi_window, "flatness", 0.2), is_flat_steep_system(band_energies, fermi_window, "bandwidth", 0.2)
        mean_flatness, sd_flatness, range_flatness, min_flatness = np.mean(near_fermi_flatnesses), np.std(near_fermi_flatnesses), score_range(near_fermi_flatnesses), np.min(near_fermi_flatnesses)
        min_relative_flatness = min_flatness/np.mean(compute_flatness(band_energies))
        mean_relative_flatness = np.mean(near_fermi_flatnesses/np.mean(compute_flatness(band_energies)))
        sd_relative_flatness = np.std(near_fermi_flatnesses/np.mean(compute_flatness(band_energies)))
        range_relative_flatness = np.max(near_fermi_flatnesses/np.mean(compute_flatness(band_energies))) - np.min(near_fermi_flatnesses/np.mean(compute_flatness(band_energies)))
        mean_bandwidth, sd_bandwidth, range_bandwidth, min_bandwidth = np.mean(near_fermi_bandwidths), np.std(near_fermi_bandwidths), score_range(near_fermi_bandwidths), np.min(near_fermi_bandwidths)
        min_relative_bandwidth = min_bandwidth/np.mean(compute_bandwidth(band_energies))
        mean_relative_bandwidth = np.mean(near_fermi_bandwidths/np.mean(compute_bandwidth(band_energies)))
        sd_relative_bandwidth = np.std(near_fermi_bandwidths/np.mean(compute_bandwidth(band_energies)))
        range_relative_bandwidth = np.max(near_fermi_bandwidths/np.mean(compute_bandwidth(band_energies))) - np.min(near_fermi_bandwidths/np.mean(compute_bandwidth(band_energies)))
        data[18*i:18*(i+1)] = [min_flatness, min_relative_flatness, mean_flatness, mean_relative_flatness, sd_flatness, sd_relative_flatness, range_flatness, range_relative_flatness, is_flat_steep_flatness, min_bandwidth, min_relative_bandwidth, mean_bandwidth, mean_relative_bandwidth, sd_bandwidth, sd_relative_bandwidth, range_bandwidth, range_relative_bandwidth, is_flat_steep_bandwidth]
        if img_dest and i == plot_idx:
            flattest_idx = np.argmin(near_fermi_flatnesses)
            if not os.path.exists(img_dest):
                os.makedirs(img_dest)
            plot_bands(k_point_distances, fermi_window_energies, fermi_window_spins, ticks, [(flattest_idx, f"Min flatness: {np.round(min_flatness, 5)}")], title=f"{mpid}", img_dest=f"{img_dest}/{mpid}_{plot_window}_eV_from_fermi.png")
    return data#, band_dict


def characterize_bands_from_local(mpid, fermi_windows, filepath):
    bs = load_band_structure(f"{filepath}/{mpid}.pkl")
    k_point_distances, band_energies, spins = get_bands(bs, plot=False)
    data = np.zeros(12*len(fermi_windows))
    for i, fermi_window in enumerate(fermi_windows):
        fermi_window_idx = near_fermi(band_energies, fermi_window)
        fermi_window_energies, fermi_window_spins = band_energies[fermi_window_idx], spins[fermi_window_idx]
        if len(fermi_window_energies) == 0:
            # no bands lie in Fermi window
            data[18*i:18*(i+1)] = [np.inf]*18
            continue
        near_fermi_flatnesses, near_fermi_bandwidths = compute_flatness(fermi_window_energies), compute_bandwidth(fermi_window_energies)
        is_flat_steep_flatness, is_flat_steep_bandwidth = is_flat_steep_system(band_energies, fermi_window, "flatness", 0.2), is_flat_steep_system(band_energies, fermi_window, "bandwidth", 0.2)
        mean_flatness, sd_flatness, range_flatness, flattest_flatness = np.mean(near_fermi_flatnesses), np.std(near_fermi_flatnesses), score_range(near_fermi_flatnesses), np.min(near_fermi_flatnesses)
        relative_flatness = flattest_flatness/np.mean(compute_flatness(band_energies))
        mean_bandwidth, sd_bandwidth, range_bandwidth, flattest_bandwidth = np.mean(near_fermi_bandwidths), np.std(near_fermi_bandwidths), score_range(near_fermi_bandwidths), np.min(near_fermi_bandwidths)
        relative_bandwidth = flattest_bandwidth/np.mean(compute_bandwidth(band_energies))
        data[18*i:18*(i+1)] = [flattest_flatness, relative_flatness, mean_flatness, sd_flatness, range_flatness, is_flat_steep_flatness, flattest_bandwidth, relative_bandwidth, mean_bandwidth, sd_bandwidth, range_bandwidth, is_flat_steep_bandwidth]
    return data

def load_band_structure(filepath):
    with open(filepath, "rb") as f:
        loaded_bs = pickle.load(f)
    return BandStructureSymmLine.from_dict(loaded_bs)