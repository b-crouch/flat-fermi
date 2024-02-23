import time
from datetime import datetime
import argparse
import sys
import numpy as np
import pandas as pd
import pickle
import os
from utils import *
import re

parser = argparse.ArgumentParser(description='Near-Fermi Band Flatness Calculation')
parser.add_argument("--freq", type=int, required=False, help="Save checkpoint of flatness data (CSV of flatness metrics) every n structures")
parser.add_argument("--fermi", type=float, nargs="*", required=False, help="Energy range(s) (eV) from Fermi level to treat as 'near-Fermi'")
parser.add_argument("--filepath", type=str, help="Filepath to directory containing pickled band structures")
parser.add_argument("--csv_dest", type=str, help="Filename for saving flatness scores")

args = parser.parse_args(sys.argv[1:])

def main():
    global args

    checkpoint = args.freq if args.freq else len(mpids)//10
    fermi_windows = [*args.fermi] if args.fermi else [1, 0.75, 0.5, 0.25, 0.1]
    csv_dest = args.csv_dest
    filepath = args.filepath
    mpids = [re.findall(r"(mp-\d+)\.pkl", file)[0] for file in os.listdir(filepath)]

    data = np.zeros((len(mpids), 10*len(fermi_windows)))
    columns = []
    for window in fermi_windows:
        for metric in ["flatness", "bandwidth"]:
            columns.extend([f"min_{metric}_{window}_ev", f"mean_{metric}_{window}_ev", f"sd_{metric}_{window}_ev", f"range_{metric}_{window}_ev", f"system_{metric}_{window}_ev"])
    computed_mpids = []
    failed_mpids = []

    start = time.time()
    print(f"Start calculation: {datetime.now()}")

    for i, mpid in enumerate(mpids):
        if i%checkpoint == 0 and i > 0:
            print("*"*10, f" Structure {i}/{len(mpids)} ", "*"*10)
            print(f"Time elapsed: {np.round((time.time() - start)/60, 3)} minutes")
            df = pd.DataFrame(data[:i, :], index=mpids[:i], columns=columns)
            df.to_csv(csv_dest)

        flatness_data = characterize_bands_from_local(mpid, fermi_windows, filepath)
        data[i, :] = flatness_data
        computed_mpids.append(mpid)

    df = pd.DataFrame(data, index=mpids, columns=columns)
    df.to_csv(csv_dest)



if __name__ == '__main__':
    main()