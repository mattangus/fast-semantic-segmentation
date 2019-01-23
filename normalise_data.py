import numpy as np
import glob
import argparse
import os
import tqdm
import multiprocess as mp
from functools import partial

parser = argparse.ArgumentParser()
parser.add_argument("--eval_dir", "-e", type=str, required=True)

args = parser.parse_args()

files = glob.glob(os.path.join(args.eval_dir, "*.jpg.npy"))

pool = mp.Pool(32)

means = []

print("computing mean")

def load_mean_min_dist(file):
    try:
        # with open(file, "r") as f:
        data = np.load(file).item()["full"]
        min_dist = np.nanmin(data,-1)
        if np.any(np.logical_not(np.isfinite(min_dist))):
            print("not finite:", file)
        return np.mean(min_dist)
    except:
        print("invalid:", file)
    return None

for v in tqdm.tqdm(pool.imap_unordered(load_mean_min_dist, files), total=len(files)):
    if v is not None:
        means.append(v)

mean = np.mean(means)
print("mean:", mean)

var_sum = 0
n = len(files)

print("computing std")

def load_var_min_dist(file, mean):
    try:
        x = np.mean(np.nanmin(np.load(file).item()["full"],-1))
        if np.any(np.logical_not(np.isfinite(x))):
            print("not finite:", file)
        return (x - mean)*(x - mean)
    except:
        print("invalid:", file)
    return None

for v in tqdm.tqdm(pool.imap_unordered(partial(load_var_min_dist, mean=mean), files), total=len(files)):
    if v is not None:
        var_sum += v

std = np.sqrt(var_sum / (n - 1))
print("std", std)

np.save(os.path.join(args.eval_dir, "normalisation.npy"), {"mean": mean, "std": std})