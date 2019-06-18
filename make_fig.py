import cv2
import numpy as np
from glob import glob
import os
from pprint import pprint
from matplotlib.pylab import cm
import sys
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="city")
parser.add_argument("--interest", type=str, default=None)

dataset_names = {
    "city": "CityScapes",
    "sun": "SUN2012",
    "lost": "LostAndFoundDataset",
    "wild": "Wilddash",
    "normal": "Normal",
    "perlin": "Perlin",
    "uniform": "Uniform",
}

args = parser.parse_args()
assert args.dataset in dataset_names, "dataset must be one of " + str(list(dataset_names.keys()))
assert args.interest in [None, "good", "bad"]

dataset = dataset_names[args.dataset]

interest = args.interest

annot_files = glob("full_export/Mahal/annot/" + dataset + "/**/*.png", recursive=True)
image_files = glob("full_export/Mahal/image/" + dataset + "/**/*.png", recursive=True)
prediction_files = glob("full_export/*/pred/" + dataset + "/**/*.png", recursive=True)
uncert_files = glob("full_export/*/map/" + dataset + "/**/*.exr", recursive=True)
iou_files = glob("full_export/*/iou/" + dataset + "/**/*.png", recursive=True)
roc_files = glob("full_export/*/roc/" + dataset + "/**/*.png", recursive=True)
meta_files = glob("full_export/*/meta.csv")

fig_type = "raw"
#fig_type = "thresh"

all_files = {}

def get_image(path, flags=cv2.IMREAD_COLOR):
    img = cv2.imread(path, flags)
    return cv2.resize(img, (0,0), fx=0.2, fy=0.2)

for af in annot_files:
    all_files[os.path.basename(af)] = {"annot": af}

for imgf in image_files:
    all_files[os.path.basename(imgf)]["img"] = imgf

for cur_f in prediction_files:
    name = cur_f.split("/")[1]
    base = os.path.basename(cur_f)
    if "pred" not in all_files[base]:
        all_files[base]["pred"] = {name: cur_f}
    else:
        all_files[base]["pred"][name] = cur_f

for cur_f in uncert_files:
    name = cur_f.split("/")[1]
    base = os.path.basename(cur_f).replace(".exr", ".png")
    if "map" not in all_files[base]:
        all_files[base]["map"] = {name: cur_f}
    else:
        all_files[base]["map"][name] = cur_f

for cur_f in uncert_files:
    name = cur_f.split("/")[1]
    base = os.path.basename(cur_f).replace(".exr", ".png")
    if "map" not in all_files[base]:
        all_files[base]["map"] = {name: cur_f}
    else:
        all_files[base]["map"][name] = cur_f

for cur_f in roc_files:
    name = cur_f.split("/")[1]
    base = os.path.basename(cur_f)
    if "roc" not in all_files[base]:

        all_files[base]["roc"] = {name: cur_f}
    else:
        all_files[base]["roc"][name] = cur_f

for cur_f in iou_files:
    name = cur_f.split("/")[1]
    base = os.path.basename(cur_f)
    if "iou" not in all_files[base]:
        all_files[base]["iou"] = {name: cur_f}
    else:
        all_files[base]["iou"][name] = cur_f

if dataset == "sun":
    for cur_f in meta_files:
        with open(cur_f,"r") as csv_file:
            csv_data = [l.strip("\n").split(",") for l in csv_file.readlines()]
        csv_data = csv_data[1:]
        name = cur_f.split("/")[1]
        for line in csv_data:
            base = os.path.basename(line[0])
            metrics = dict(zip(["auroc","aupr","max_iou","fpr_at_tpr","detection_error"],list(map(float,line[1:]))))
            if "metrics" not in all_files[base]:
                all_files[base]["metrics"] = {name: metrics}
            else:
                all_files[base]["metrics"][name] = metrics

gray_values = np.arange(256, dtype=np.uint8)
color_values = map(tuple, cv2.applyColorMap(gray_values, cv2.COLORMAP_JET).reshape(256, 3))
color_to_gray_map = dict(zip(color_values, gray_values))

def reverse_map(img):
    grey = np.apply_along_axis(lambda bgr: color_to_gray_map[tuple(bgr)], 2, img)
    return grey

class FigMaker(object):

    def __init__(self, all_files):
        self.all_files = all_files
        self.all_keys = list(all_files.keys())
        self.cur_file = 0
        self.cur_image = None
        self.result = None

    def current_intresting(self, metric_name):
        if interest == "bad":
            cmp = lambda v: (v > 0.001 and v < 0.25)
        elif interest == "good":
            cmp = lambda v: v > 0.9 and v < 1
        else:
            return True

        metrics = self.all_files[self.all_keys[self.cur_file]]["metrics"]
        values = [metrics[k][metric_name] for k in metrics]

        return cmp(np.mean(values))

    def change_file(self, direction):
        direction = np.sign(direction)
        
        self.cur_file += direction
        self.cur_file = self.cur_file % len(self.all_keys)

        if dataset == "sun":
            metric_name = "auroc"

            while not self.current_intresting(metric_name):
                self.cur_file += direction
                self.cur_file = self.cur_file % len(self.all_keys)
            
            metrics = self.all_files[self.all_keys[self.cur_file]]["metrics"]
            values = [metrics[k][metric_name] for k in metrics]
            print(values)

    def append_image(self):
        if self.result is None:
            self.result = self.cur_image
        else:
            bar = np.ones((2, self.result.shape[1], 3),dtype=np.uint8)*255
            self.result = np.concatenate([self.result, bar, self.cur_image], 0)

    def save(self):
        out_name = input("filename:")
        if ".png" not in out_name:
            out_name += ".png"
        cv2.imwrite(out_name, self.result)

    def process_key(self, key):

        if key == 97: #a
            self.change_file(-1)
        elif key == 100: #d
            self.change_file(1)
        elif key == 115: #s
            self.save()
        elif key == 32: #space
            self.append_image()
        elif key == 27: #esc
            sys.exit(0)

    def make_row_thres(self):
        f_name = self.all_keys[self.cur_file]
        annot = get_image(self.all_files[f_name]["annot"])
        img = get_image(self.all_files[f_name]["img"])

        pred = get_image(self.all_files[f_name]["pred"]["MaxSoftmax"])
        map_order = ["ODIN"]
        ious = [get_image(self.all_files[f_name]["iou"][v]) for v in map_order]
        rocs = [get_image(self.all_files[f_name]["roc"][v]) for v in map_order]

        zeros = np.zeros((img.shape[:-1]))
        # import pdb; pdb.set_trace()

        iou_t = [np.stack([zeros, i[...,0], i[...,0]], -1).astype(np.uint8) for i in ious]
        roc_t = [np.stack([zeros, i[...,0], i[...,0]], -1).astype(np.uint8) for i in rocs]

        img_list = [img, annot] + iou_t + roc_t

        bar = np.ones((img.shape[0], 2, 3),dtype=np.uint8)*255
        result = [bar] * (len(img_list) * 2 - 1)
        result[0::2] = img_list

        disp = np.concatenate(result, 1)
        return disp

    def make_row_raw(self):
        f_name = self.all_keys[self.cur_file]
        annot = get_image(self.all_files[f_name]["annot"])
        img = get_image(self.all_files[f_name]["img"])

        pred = get_image(self.all_files[f_name]["pred"]["MaxSoftmax"])
        map_order = ["MaxSoftmax", "ODIN", "Mahal", "Confidence", "Dropout", "Entropy"]
        maps = [get_image(self.all_files[f_name]["map"][v], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) for v in map_order]
        #import pdb; pdb.set_trace()
        maps = [m - m.min() for m in maps]
        maps = [m/m.max() for m in maps]

        maps = [(cm.inferno(m)[...,:-1][...,::-1]*255).astype(np.uint8) for m in maps]

        img_list = [img, annot, pred] + maps

        bar = np.ones((img.shape[0], 2, 3),dtype=np.uint8)*255
        result = [bar] * (len(img_list) * 2 - 1)
        result[0::2] = img_list

        disp = np.concatenate(result, 1)
        return disp

    def main_loop(self):

        while True:
            if fig_type == "raw":
                self.cur_image = self.make_row_raw()
            elif fig_type == "thresh":
                self.cur_image = self.make_row_thres()

            cv2.imshow("current", self.cur_image)
            if self.result is not None:
                cv2.imshow("result", self.result)
            
            key = cv2.waitKey()

            self.process_key(key)
        
fmaker = FigMaker(all_files)
fmaker.main_loop()
