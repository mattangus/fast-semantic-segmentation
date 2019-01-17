import numpy as np
import glob

files = glob.glob("eval_logs/mobile_dim_dist2/dump_LG/*.npz")

means = []

print("computing mean")

for f in files:
    means.append(np.mean(np.load(f)["arr_0"]))

mean = np.mean(means)
print("mean:", mean)

var_sum = 0
n = len(files)

print("computing variance")

for f in files:
    x = np.mean(np.load(f)["arr_0"])
    var_sum += (x - mean)*(x - mean)

variance = var_sum / (n - 1)
print("variance", variance)