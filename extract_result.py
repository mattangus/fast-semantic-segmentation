import numpy as np
import glob
import argparse
import os
import re

parser = argparse.ArgumentParser()
parser.add_argument("--folder", "-f", type=str, required=True)
parser.add_argument("--type", "-t", type=str, required=True)

args = parser.parse_args()

if args.type == "odin":
    pattern = os.path.join(args.folder, "odin_*.log")
    reg = r".*eps_(\d+\.\d+).*t_(\d+).*"
else:
    pattern = os.path.join(args.folder, "eps_*.log")
    reg = r".*eps_(\d+\.\d+).*"


files = glob.glob(pattern)

score_reg = r"area under curve: (\d+\.\d+)"

all_scores = []

for f_name in files:
    params_match = re.match(reg, f_name)
    params = list(map(float, params_match.groups()))

    with open(f_name,"r") as f:
        data = f.readlines()
    
    for l in data:
        match = re.match(score_reg, l)
        if match is not None:
            break

    if match is None:
        print("no match", f_name)
        continue
    score = match.groups()
    assert len(score) == 1, "bad score: " + score

    score = float(score[0])

    all_scores.append((params, score))


sorted_scores = sorted(all_scores, key=lambda x: x[1], reverse=True)
print(sorted_scores[:3])
