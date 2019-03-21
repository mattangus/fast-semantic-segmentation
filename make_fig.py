import cv2
import numpy as np
from glob import glob
import os
from pprint import pprint
from matplotlib.pylab import cm
import sys

annot_files = glob("exported/Dropout/annot/*.png")
image_files = glob("exported/Dropout/image/*.png")
prediction_files = glob("exported/*/pred/*.png")
uncert_files = glob("exported/*/map/*.png")

all_files = {}

def get_image(path):
    img = cv2.imread(path)
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
    base = os.path.basename(cur_f)
    if "map" not in all_files[base]:
        all_files[base]["map"] = {name: cur_f}
    else:
        all_files[base]["map"][name] = cur_f

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

    def change_file(self, direction):
        direction = np.sign(direction)

        self.cur_file += direction
        self.cur_file = self.cur_file % len(self.all_keys)

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

    def make_row(self):
        f_name = self.all_keys[self.cur_file]
        annot = get_image(self.all_files[f_name]["annot"])
        img = get_image(self.all_files[f_name]["img"])

        pred = get_image(self.all_files[f_name]["pred"]["MaxSoftmax"])
        map_order = ["MaxSoftmax", "ODIN", "Mahal", "Confidence", "Dropout"]
        maps = [get_image(self.all_files[f_name]["map"][v]) for v in map_order]

        gscales = list(map(reverse_map, maps))
        maps = list(map(lambda x: (cm.inferno(x)[...,:-1][...,::-1]*255).astype(np.uint8), gscales))

        img_list = [img, annot, pred] + maps

        bar = np.ones((img.shape[0], 2, 3),dtype=np.uint8)*255
        result = [bar] * (len(img_list) * 2 - 1)
        result[0::2] = img_list

        disp = np.concatenate(result, 1)
        return disp

    def main_loop(self):

        while True:
            self.cur_image = self.make_row()

            cv2.imshow("current", self.cur_image)
            if self.result is not None:
                cv2.imshow("result", self.result)
            
            key = cv2.waitKey()

            self.process_key(key)
        
fmaker = FigMaker(all_files)
fmaker.main_loop()

# for f_name in 

#     cv2.imshow("disp", disp)

#     # names = [v.split("/")[1] for v in sorted(all_files[f_name]["pred"].values())]

#     # cv2.imshow("annot", annot)
#     # cv2.imshow("img", img)

#     # for n,p in zip(names,preds):
#     #     cv2.imshow(n + "_pred", p)

#     # for n,m in zip(names,gscales):
#     #     temp = cm.viridis(m)[...,:-1]*255
#     #     cv2.imshow(n + "_map", temp.astype(np.uint8))

#     key = cv2.waitKey()
#     if key == 27:
#         break
