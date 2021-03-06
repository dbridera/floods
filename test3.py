from change_class import *
import sys
# Define images to use
img1 = "/home/daniel/floods/l8/sample1_before"
img2 = "/home/daniel/floods/l8/sample1_after"

# Create object

t = ChangeImage(img1,img2)

# Load and create imag diff using NDI vectors
t.load()

t.imageDiff(source='ndi')

# Create matrix of ROS and ALPHAS
t.getMagnitudeMatrix()
nozero = t.ros[np.where(t.ros !=0)]
t.ros[np.where(t.ros ==0)]= nozero.min()
#t.getAlphaMatrix()

t.fixRos()

"""
t.ros[np.where(t.ros>=0.07007667413584022)] = 1
t.ros[np.where(t.ros<0.07007667413584022)] = None

saveTiff(t.ros, "l8/resultados/sample6", img1)
exit()
"""

print "ROS extremos"
print t.ros.min(), t.ros.max()

print "Medio"
print (-t.ros.min() + t.ros.max())/2

#line = sys.stdin.readline()
alpha = raw_input("Defined ALPHA: ")
print "Limits with ALPHA = " + alpha.strip()
alpha = float(alpha.strip())

t.getLimits(alpha)

print t.tn, t.tc

# Start iterations to retrieve params
mu_c, sigma_c = getInitialParams(t.ros, t.tc)
mu_nc, sigma_nc = getInitialParams(t.ros, t.tn,lower=True)


pnorig, pcorig  = 1 , 1
mnorig, mcorig  = 1 , 1
snorig, scorig  = 1 , 1
pn ,pc  =  0.68359318882172837, 0.3164061117827169
mu_nc = 0.065044686302178578
mu_c = 0.15246536667231769
sigma_nc = 0.00073523422023898382
sigma_c = 0.0042433410739568514

print "PNO-CHANGE = %f, PCHANGE = %f" % (pn ,pc)


for i in range(3000):
	if (pcorig == pc and pnorig == pn and mcorig == mu_c and mnorig == mu_nc and scorig == sigma_c and snorig == sigma_nc):
		break
	
	pcorig = pc
	pnorig = pn
	
	mcorig = mu_c
	mnorig = mu_nc
	
	scorig = sigma_c
	snorig = sigma_nc

	
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
alphas = t.alpha[np.where(t.ros >=0.10693342490619757)]

cos = alphas

alphas = np.arccos(alphas)
ros = t.ros[np.where(t.ros >=0.10693342490619757)]

x = ros * cos
y = ros * np.sin(alphas)

#print x
#print y

cambios = np.vstack((x, y)).T

#print cambios


from sklearn.cluster import MeanShift, estimate_bandwidth

ms = MeanShift(bandwidth=0.03)
X = cambios
#print X
ms.fit(X)
labels = ms.labels_
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
"""

"""
t.alpha[np.where(t.alpha > 0.775)] = 200
t.alpha[np.where(t.alpha <= 0.775)] = 100
t.alpha[np.where(t.ros < 0.10693342490619757)] = 0

t.alpha= t.alpha/100

gdal_array.SaveArray(t.alpha.astype("int"), "SUELO/clasificada_AGUA_AGUA", "GTiff", gdal.Open(img1))

exit()
"""
"""
#t.getAlphaMatrix()
#print t.ros[120:190,120:190]
#print ""
#print t.ros[300:399,300:399]
"""
exit()
