from osgeo import gdal, gdal_array
import numpy as np
from utils import *
from scipy.stats import norm
from math import cos, sin , sqrt
import time

from sklearn.cluster import MeanShift, estimate_bandwidth

start_time = time.clock()

def iterf(ros,alpha,d,f):
	res = np.empty([d,f], dtype='object')

	for j in range(d):
		for e in range(f):
			res[j,e] = (ros[j,e]*cos(alpha[j,e]), ros[j,e]*sin(alpha[j,e])) 
	return res


start_time = time.clock()

# Load Images
b1 = gdal.Open("img/recorte1")
array1 = b1.ReadAsArray().astype('float')
b2 = gdal.Open("img/recorte2")
array2 = b2.ReadAsArray().astype('float')


# Get Indexes
ndi1 = getNDIs(array1)
ndi2 = getNDIs(array2)


# Get difference
resta = ndi2 - ndi1
# Get ROS
ros = getMagnitudeMatrix(resta)

#rosFix = (1/(2*sqrt(21))) * ros
alpha = getAlphaMatrix(resta, sqrt(21)*ros)

print np.mean(alpha), alpha.min(), alpha.max()

"""
#Fixin ros
rosFix = (1/(2*sqrt(21))) * ros

# Get ALPHAS
alpha = getAlphaMatrix(resta, sqrt(21)*ros)


# Get (x,y)
res = iterf(ros, alpha,*ros.shape)
sample = res[:10,:]
result=np.array(sample).flatten()

coord = map(np.array, result)
coord = np.array(coord)

print time.clock() - start_time

start_time = time.clock()
#ban = estimate_bandwidth(coord, quantile=0.2, n_samples=500)

ms = MeanShift()
ms.fit(coord)

labels = ms.labels_
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

print("number of estimated clusters : %d" % n_clusters_)

print time.clock() - start_time
# Plot result
"""

"""
import matplotlib.pyplot as plt
from itertools import cycle

plt.figure(1)
plt.clf()

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    my_members = labels == k
    cluster_center = cluster_centers[k]
    plt.plot(coord[my_members, 0], coord[my_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)
plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()
#print time.clock() - start_time
"""
"""
ndv1 = getNDIs(img1)
ndv2 = getNDIs(img2)

# las diferencia de NDI
resta = ndv1- ndv2

ros = getMagnitudeMatrix(resta)
alpha = getAlphaMatrix(resta, sqrt(3)*ros)
"""


"""b1 = gdal.Open("img/recorte_rgb")

array1 = b1.ReadAsArray().astype('float')

print getNDVs(array1).shape"""

"""
b1 = gdal.Open("img/recorte1")
b2 = gdal.Open("img/recorte2")

array1 = b1.ReadAsArray()
array2 = b2.ReadAsArray()

print array1
print array2
p1 = np.array(array1).astype('int')
p2 = np.array(array2).astype('int')
print p1.shape, p2.shape
#print np.mean(stacked), np.std(stacked), stacked.min(), stacked.max()

ros  = getMagnitudeMatrix(stacked)
#print np.mean(ros), np.std(ros), ros.min(), ros.max()

lambdas = getAlphaMatrix(stacked, ros)
lambdas = 0.5 * lambdas
#print np.mean(lambdas), np.std(lambdas), lambdas.min(), lambdas.max()


#unique, count = np.unique(ros,return_counts=True)

#frecuencies = np.asarray((unique, count)).T

teta = vfun(lambdas,0,1)
"""