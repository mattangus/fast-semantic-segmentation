import tensorflow as tf
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import os

parser = argparse.ArgumentParser()
parser.add_argument("--mean", type=str, required=True)
parser.add_argument("--cov_inv", type=str, default=None)
#parser.add_argument("--output", type=str, required=True)

args = parser.parse_args()

mean = np.load(args.mean)
if "npz" in args.mean:
    mean = mean["arr_0"]
if args.cov_inv:
    cov_inv = np.load(args.cov_inv)
    if "npz" in args.cov_inv:
        cov_inv = cov_inv["arr_0"]
else:
    cov_inv = None
mean = np.swapaxes(mean, 3, 0)
if cov_inv:
    cov_inv = np.swapaxes(cov_inv, 3, 0)
num_class = mean.shape[0]

cm = np.zeros([num_class, num_class])

def mahal(m, cov_inv, p):
    temp = p - m
    cov_inv = np.squeeze(cov_inv)

    left = np.matmul(temp, cov_inv)
    dist = dist = np.squeeze(np.matmul(left, np.transpose(temp, [0,1,3,2])))
    return np.mean(dist)

def l2_norm(m, unused, p):
    return np.sqrt(np.sum(np.square(m-p)))

if cov_inv:
    dist_fn = mahal
else:
    dist_fn = l2_norm

for i in range(num_class):
    for j in range(num_class):
        cur_cov = cov_inv[i] if cov_inv else None
        cm[i,j] = dist_fn(mean[i], cur_cov, mean[j])
        #cm[j,i] = cm[i,j]

xticklabels = ['road','sidewalk','building','wall','fence','pole','traffic light',
                'traffic sign','vegetation','terrain','sky','person','rider','car',
                'truck','bus','train','motorcycle','bicycle']
yticklabels = xticklabels

plt.figure(figsize=[20.48,10.24])
sb.heatmap(cm, annot=True, fmt="g", xticklabels=xticklabels, yticklabels=yticklabels)
plt.savefig(os.path.join(os.path.dirname(args.mean),"mean_dists.png"))
plt.cla() #clear memory since no display happens
plt.clf()


c = [(128, 64,128),(244, 35,232),( 70, 70, 70),(102,102,156),(190,153,153),(153,153,153),(250,170, 30),(220,220,  0),
    (107,142, 35),(152,251,152),( 70,130,180),(220, 20, 60),(255,  0,  0),(  0,  0,142),(  0,  0, 70),(  0, 60,100),
    (  0, 80,100),(  0,  0,230),(119, 11, 32)]
c = np.array(c)/255

mean = np.reshape(mean, [num_class, -1])

pca = PCA(n_components=3)
tformed = pca.fit_transform(mean)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(tformed[:,0], tformed[:,1], tformed[:,2], c=c)
plt.show()
