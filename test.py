from change_class import *
import sys

# Define images to use
img1 = "/home/daniel/floods/l8/filtrados/sample1_before_filtrado"
img2 = "/home/daniel/floods/l8/filtrados/sample1_after_filtrado"

# Create object
t = ChangeImage(img1,img2)
# Load and create imag diff using NDI vectors

t.load()

#t.calibrate('tex.imd')

#combinations = [(6,4),(7,5),(0,7),(1,0),(2,7),(6,5),(2,3)]
t.imageDiff(source='ndi')

# Create matrix of ROS and ALPHAS
t.getMagnitudeMatrix()

nozero = t.ros[np.where(t.ros !=0)]
t.ros[np.where(t.ros == 0)]= -1
t.getAlphaMatrix()

t.fixRos()


umbral = 0.13573749839952443
#t.ros[np.where(t.ros>=0.021913564571751486)] = 1
#t.ros[np.where(t.ros<umbral)] = 0

t.alpha[np.where(t.ros<umbral)] = 0

cos = t.alpha

alphas = np.arccos(t.alpha)
#ros = t.ros[np.where(t.ros >=0.10693342490619757)]

copy = t.ros
copy[np.where(copy<umbral)] = 0

x = t.ros * cos
y = copy * np.sin(alphas)

rows, col = t.ros.shape
print t.ros.shape

cambios = np.vstack((x.flatten(), y.flatten())).T

print cambios.shape

from sklearn.cluster import MeanShift, estimate_bandwidth

ms = MeanShift(bandwidth=0.03)


X = cambios
#print X
ms.fit(X)

print "termino fit"

labels = ms.labels_
print labels.shape


#clus = ms.predict(cambios)
print "termino predecir"
#print "shape"

pa = labels.reshape(rows,col)

print pa.shape
#saveTiff(t.ros, "ros", img1)
#saveTiff(pa, "class", img1)


#print clus.reshape(2,1000)


exit()

cluster_centers = ms.cluster_centers_
print cluster_centers


labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

print n_clusters_

print ms.get_params(deep=True)

import matplotlib.pyplot as plt
from itertools import cycle

plt.figure(1)
plt.clf()

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    my_members = labels == k
    cluster_center = cluster_centers[k]
    plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)
plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()


t.alpha[np.where(t.alpha > 0.775)] = 200
t.alpha[np.where(t.alpha <= 0.775)] = 100
t.alpha[np.where(t.ros < 0.10693342490619757)] = 0

t.alpha= t.alpha/100

gdal_array.SaveArray(t.alpha.astype("int"), "SUELO/clasificada_AGUA_AGUA", "GTiff", gdal.Open(img1))

exit()

#t.getAlphaMatrix()
#print t.ros[120:190,120:190]
#print ""
#print t.ros[300:399,300:399]
#exit()
