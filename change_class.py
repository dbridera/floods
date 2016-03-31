from osgeo import gdal, gdal_array
import numpy as np
from utils import *
from scipy.stats import norm
from math import cos, sin , sqrt, factorial
import time


class ChangeImage(object):

	def __init__(self, imgBef, imgAft):
		super(ChangeImage, self).__init__()
		self.imgBef = imgBef
		self.imgAft = imgAft
		self.diff = None
		self.ros = None
		self.alpha = None
		self.coef = None
		self.bands , self.x, self.y = None , None, None
		self.tn , self.tc = None, None

	def load(self):

		self.imgBef = gdal.Open(self.imgBef).ReadAsArray().astype('float')
		self.imgAft = gdal.Open(self.imgAft).ReadAsArray().astype('float')

		self.bands , self.x, self.y = self.imgAft.shape
		self.coef = sqrt(factorial(self.bands)/ (2* factorial(self.bands - 2)))
	def getNDIs(self, matrix):

		to_combine = combinations(range(self.bands),2)

		aux = []

		for e in to_combine:
			aux.append(e)

		res = np.empty([len(aux), self.x , self.y])

		for i, (b1, b2) in enumerate(aux):

			img1 = matrix[b1,:,:]
			img2 = matrix[b2,:,:]
			#with np.errstate(invalid='ignore',divide='ignore'):
			res[i,:,:]= (img1 - img2) / (img1 + img2)

		#res[~np.isfinite(res)]= 0

		return res

	def imageDiff(self, source='origin'):
		"""
		Compute the image difference.
		The result type depends on parameteers type.
		"""
		if source =='origin':
			self.diff = self.imgAft - self.imgBef
		elif source =='ndi':
			self.diff = getNDIs(self.imgAft) - getNDIs(self.imgBef)
			self.bands , self.x, self.y = self.diff.shape
			self.coef = sqrt(self.bands)


	def getMagnitudeMatrix(self):
		pre = self.diff**2
		self.ros = np.sqrt(np.sum(pre, axis=0))

	def fixRos(self):
		self.ros = (1/(2*self.coef)) * self.ros

	def getAlphaMatrix(self):

		numerator = np.sum(self.diff, axis=0)
		#coef = sqrt(factorial(self.bands)/ (2* factorial(self.bands - 2)))
		#with np.errstate(invalid='ignore',divide='ignore'):
		self.alpha = numerator/ (self.coef * self.ros)

	def getLimits(self, alpha):
		middle = (self.ros.max() - self.ros.min())/2
		self.tn, self.tc = middle - alpha, middle + alpha

	def getInitialParams(self,numpyArray, t, lower=False):
		if lower:
			values = numpyArray[np.where(numpyArray < t)]
		else:
			values = numpyArray[np.where(numpyArray > t)]

		return np.mean(values), np.std(values)

	def pdf(self, matrix, mu, sigma):

		def fun(a):
			return norm(mu,sigma).pdf(a)

		vfun = np.vectorize(fun)
		res = vfun(matrix)
		return res

	def initial(self,alpha):
		
		self.getLimits(alpha)
		m_n, sigma_n = getInitialParams(self.ros,self.tn, lower=True)
		m_c, sigma_c = getInitialParams(self.ros,self.tc)

		return m_n, sigma_n, m_c, sigma_c

	def computeRatio(self,Pc, Pnc, mu_c, sigma_c, mu_nc, sigma_nc):
		Pxn = pdf(self.ros, mu_nc, sigma_nc)
		Pxc = pdf(self.ros, mu_c, sigma_c)

		Pn_Pxn =Pnc * Pxn
		Pc_Pxc = Pc* Pxc
		#print Pn_Pxn , Pc_Pxc

		Px = Pn_Pxn + Pc_Pxc
		ratio1 = Pn_Pxn/Px
		ratio2 = Pc_Pxc/Px
			
		"""
		with np.errstate(invalid='ignore',divide='ignore'):
			ratio1 = Pn_Pxn/Px
			ratio2 = Pc_Pxc/Px
			ratio1[~np.isfinite(ratio1)] = 0
			ratio2[~np.isfinite(ratio2)] = 0
		"""
		return  ratio1, ratio2

	def equation1(self,ratioPn, ratioPc):
		# esto me viene, para no calcularlo de nuevo!!!
		#newPn , newPc = computeRatio(Pc, Pnc, mu_c, sigma_c, mu_nc, sigma_nc)
		denominator = (self.x * self.y)
		#denominator = 6262
		return np.sum(ratioPn)/denominator, np.sum(ratioPc)/denominator

	def equation2(self,ratioPn, ratioPc):
		num_n = np.sum(ratioPn * self.ros)
		num_c = np.sum(ratioPc * self.ros)

		return num_n/np.sum(ratioPn), num_c/np.sum(ratioPc)

	def equation3(self,ratioPn, ratioPc, mu_nc,  mu_c):
		num_n = np.sum(ratioPn * ((self.ros - mu_nc)**2))
		num_c = np.sum(ratioPc * ((self.ros - mu_c)**2))
		
		
		return num_n/np.sum(ratioPn), num_c/np.sum(ratioPc)

start_time = time.clock()

img1 = "RECORTES/400_mix"
img2 = "RECORTES/ruido_normal_cultivos_agua_agua"
t = ChangeImage(img1,img2)

t.load()
t.imageDiff(source='ndi')
t.getMagnitudeMatrix()
t.getAlphaMatrix()

t.fixRos()

print t.alpha.min(), t.alpha.max()


#cambios = t.alpha[np.where(t.ros >0.30420492642543551)]

t.alpha[np.where(t.alpha >-0.825)] = 2
t.alpha[np.where(t.alpha <=-0.825)] = 1
t.alpha[np.where(t.ros <=0.30420492642543551)] = 0

gdal_array.SaveArray(t.alpha.astype("int"), "clasificada_AGUA_AGUA", "GTiff", gdal.Open(img1))

exit()





#t.getAlphaMatrix()
#print t.ros[120:190,120:190]
#print ""
#print t.ros[300:399,300:399]

"""
print "ROS extremos"

p = t.ros[np.where(t.ros !=0)]

#print p.min()
t.ros[np.where(t.ros ==0)]= p.min()

print t.ros.min(), t.ros.max()


#print t.ros[np.where(t.ros >0.6)].shape

print "Medio"
print (-t.ros.min() + t.ros.max())/2

t.getLimits(0.2)

print "limites"
"""

"""
import matplotlib.pyplot as plt

plt.hist(t.ros, bins=[0,0.1,0.2,0.3,0.4,0.5,0.6])

plt.show()
exit()
"""
"""
t.ros[np.where(t.ros >0.30420492642543551)] = 1	
t.ros[np.where(t.ros <= 0.30420492642543551)] = 0

gdal_array.SaveArray(t.ros.astype("int"), "ruido_normal_SUELO_AGUA", "GTiff", gdal.Open(img1))

exit()"""

"""
mu_c, sigma_c = getInitialParams(t.ros, t.tc)
mu_nc, sigma_nc = getInitialParams(t.ros, t.tn,lower=True)


pnorig, pcorig  = 1 , 1
mnorig, mcorig  = 1 , 1
snorig, scorig  = 1 , 1

pn ,pc  =  0.75, 0.25

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

	print (pn,pc)
	print (mu_nc, mu_c)
	print (sigma_nc, sigma_c)
	print ""

print "Roots"
print computeRoots(mu_nc,mu_c,sigma_nc,sigma_c,pn,pc)
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

"""
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