import argparse

from experiment_mgr import producer

parser = argparse.ArgumentParser()
parser.add_argument("gpus", nargs='+', type=str)
args = parser.parse_args()

#validate args
try:
    for gpu in args.gpus:
        list(map(int,gpu.split(",")))
except:
    print("invalid gpu list")
    import sys; sys.exit(1)

producer.main(args.gpus)
