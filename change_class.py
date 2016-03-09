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
			with np.errstate(invalid='ignore',divide='ignore'):
				res[i,:,:]= (img1 - img2) / (img1 + img2)

		res[~np.isfinite(res)]= 0

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
			self.coef = sqrt(factorial(self.bands)/ (2* factorial(self.bands - 2)))

	def getMagnitudeMatrix(self):
		pre = self.diff**2
		self.ros = np.sqrt(np.sum(pre, axis=0))

	def fixRos(self):
		self.ros = (1/(2*self.coef)) * self.ros

	def getAlphaMatrix(self):

		numerator = np.sum(self.diff, axis=0)
		#coef = sqrt(factorial(self.bands)/ (2* factorial(self.bands - 2)))
		with np.errstate(invalid='ignore',divide='ignore'):
			self.alpha = numerator/ (self.coef * self.ros)
			self.alpha[~np.isfinite(self.alpha)] = 0

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
		Px = Pn_Pxn + Pc_Pxc

		return Pn_Pxn/Px , Pc_Pxc/Px

	def equation1(self,ratioPn, ratioPc):
		# esto me viene, para no calcularlo de nuevo!!!
		#newPn , newPc = computeRatio(Pc, Pnc, mu_c, sigma_c, mu_nc, sigma_nc)
		denominator = (self.x * self.y)
		return np.sum(ratioPn)/denominator, np.sum(ratioPc)/denominator

	def equation2(self,ratioPn, ratioPc,Pc, Pnc, mu_c, sigma_c, mu_nc, sigma_nc):
		num_n = np.sum(ratioPn * ros)
		num_c = np.sum(ratioPc * ros)

		return num_n/np.sum(ratioPn), num_c/np.sum(ratioPc)

	def equation3(self,ratioPn, ratioPc,Pc, Pnc, mu_c, sigma_c, mu_nc, sigma_nc):
		num_n = np.sum(ratioPn * ((ros - mu_n)**2))
		num_c = np.sum(ratioPc * ((ros - mu_c)**2))

		return num_n/np.sum(ratioPn), num_c/np.sum(ratioPc)


t = ChangeImage("img/recorte1","img/recorte2")
t.load()
t.imageDiff(source='ndi')
t.getMagnitudeMatrix()
t.getAlphaMatrix()
t.fixRos()
t.getLimits(0.03)

print "Max ro \t\tMin ro \t\tMedia ro \t\tPromedio ro"
print t.ros.max(), t.ros.min(), (t.ros.max()-t.ros.min())/2, np.mean(t.ros)
print "Threshold tn , Threshold tc"
print t.tn,t.tc

mu_c, sigma_c = getInitialParams(t.ros, t.tc)
mu_nc, sigma_nc = getInitialParams(t.ros, t.tn,lower=True)

print "Mu c , sigma c"
print mu_c, sigma_c
print "Mu nc , sigma nc"
print mu_nc, sigma_nc
print "\n"
print "Compute ratio"

Pn_Pxn_Px , Pc_Pxc_Px = t.computeRatio(0.3, 0.7, mu_c, sigma_c, mu_nc, sigma_nc)

print Pn_Pxn_Px , Pc_Pxc_Px

"""
t.imageDiff(source='ndi')

t.getMagnitudeMatrix()
t.getAlphaMatrix()
print np.mean(t.alpha), t.alpha.min(), t.alpha.max()
"""