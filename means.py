import tensorflow as tf
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=True)
#parser.add_argument("--output", type=str, required=True)


args = parser.parse_args()

mat = np.load(args.input)
if "npz" in args.input:
    mat = mat["arr_0"]

mat = np.swapaxes(mat, 3, 0)
num_class = mat.shape[0]
import pdb; pdb.set_trace()
mat = np.reshape(mat, [num_class, -1])

cm = np.zeros([num_class, num_class])

for i in range(num_class):
    for j in range(i, num_class):
        cm[i,j] = np.linalg.norm(mat[i,:] - mat[j,:])
        cm[j,i] = cm[i,j]

xticklabels = ['road','sidewalk','building','wall','fence','pole','traffic light',
                'traffic sign','vegetation','terrain','sky','person','rider','car',
                'truck','bus','train','motorcycle','bicycle']
yticklabels = xticklabels

plt.figure(figsize=[20.48,10.24])
sb.heatmap(cm, annot=True, fmt="g", xticklabels=xticklabels, yticklabels=yticklabels)
plt.savefig("mean_dists.png")
plt.cla() #clear memory since no display happens
plt.clf()


c = [(128, 64,128),(244, 35,232),( 70, 70, 70),(102,102,156),(190,153,153),(153,153,153),(250,170, 30),(220,220,  0),
    (107,142, 35),(152,251,152),( 70,130,180),(220, 20, 60),(255,  0,  0),(  0,  0,142),(  0,  0, 70),(  0, 60,100),
    (  0, 80,100),(  0,  0,230),(119, 11, 32)]
c = np.array(c)/255


pca = PCA(n_components=3)
tformed = pca.fit_transform(mat)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(tformed[:,0], tformed[:,1], tformed[:,2], c=c)
plt.show()
