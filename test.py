from change_class import *
import sys
# Define images to use
img1 = "RECORTES/400_mix"
img2 = "RECORTES/ruido_normal_agua"

# Create object

t = ChangeImage(img1,img2)

# Load and create imag diff using NDI vectors
t.load()
t.imageDiff(source='ndi')

# Create matrix of ROS and ALPHAS
t.getMagnitudeMatrix()

#t.getAlphaMatrix()

t.fixRos()

nozero = t.ros[np.where(t.ros !=0)]

t.ros[np.where(t.ros ==0)]= nozero.min()


"""
t.ros[np.where(t.ros>=0.30420492642543551)] = 1
t.ros[np.where(t.ros<0.30420492642543551)] = 0

saveTiff(t.ros, "otrapuebaagua", img1)
exit()
"""

print "ROS extremos"

print t.ros.min(), t.ros.max()

print "Medio"
print (-t.ros.min() + t.ros.max())/2

#line = sys.stdin.readline()
alpha = raw_input("Define ALPHA: ")
print "Limits with ALPHA = " + alpha.strip()
alpha = float(alpha.strip())

t.getLimits(alpha)

print "limites"
print t.tn, t.tc


# Start iterations to retrieve params
mu_c, sigma_c = getInitialParams(t.ros, t.tc)
mu_nc, sigma_nc = getInitialParams(t.ros, t.tn,lower=True)


pnorig, pcorig  = 1 , 1
mnorig, mcorig  = 1 , 1
snorig, scorig  = 1 , 1
pn ,pc  =  0.95, 0.05
print "PNO-CHANGE = %f, PCHANGE = %f" % (pn ,pc)


for i in range(100):
	if (pcorig == pc and pnorig == pn and mcorig == mu_c and mnorig == mu_nc and scorig == sigma_c and snorig == sigma_nc):
		break
	
	pcorig = pc
	pnorig = pn
	
	mcorig = mu_c
	mnorig = mu_nc
	
	scorig = sigma_c
	snorig = sigma_nc

	if i!=0:
		sigma_c , sigma_nc = sqrt(sigma_c) , sqrt(sigma_nc)

	Pn_Pxn_Px , Pc_Pxc_Px = t.computeRatio(pc, pn, mu_c, sigma_c, mu_nc, sigma_nc)
	mu_c_orig,mu_nc_orig  =  mu_c , mu_nc

	pn, pc =  t.equation1(Pn_Pxn_Px, Pc_Pxc_Px)

	mu_nc, mu_c = t.equation2(Pn_Pxn_Px, Pc_Pxc_Px)
	sigma_nc, sigma_c = t.equation3(Pn_Pxn_Px, Pc_Pxc_Px, mu_nc_orig, mu_c_orig)

	print "Iter %s" % i
	print (pn,pc)
	print (mu_nc, mu_c)
	print (sigma_nc, sigma_c)
	

print "Roots"
print computeRoots(mu_nc,mu_c,sigma_nc,sigma_c,pn,pc)

bins = [0,0.1,0.2,0.3,0.4,0.5,0.6]
histogram(t.ros,bins)

"""
#cambios = t.alpha[np.where(t.ros >0.30420492642543551)]

t.alpha[np.where(t.alpha >-0.825)] = 2
t.alpha[np.where(t.alpha <=-0.825)] = 1
t.alpha[np.where(t.ros <=0.30420492642543551)] = 0

gdal_array.SaveArray(t.alpha.astype("int"), "clasificada_AGUA_AGUA", "GTiff", gdal.Open(img1))

exit()
"""
#t.getAlphaMatrix()
#print t.ros[120:190,120:190]
#print ""
#print t.ros[300:399,300:399]
"""
exit()
from sklearn.cluster import MeanShift, estimate_bandwidth

ms = MeanShift(bandwidth=0.05)
X = cambios.reshape(-1, 1)
print X
ms.fit(X)
labels = ms.labels_
cluster_centers = ms.cluster_centers_
print cluster_centers

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)


import matplotlib.pyplot as plt
from itertools import cycle

plt.figure(1)
plt.clf()

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    my_members = labels == k
    cluster_center = cluster_centers[k]
    plt.plot(X[my_members, 0], col + '.')
    plt.plot(cluster_center[0],  'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)
plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()

"""