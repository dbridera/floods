from osgeo import gdal, gdal_array
import numpy as np
from utils import *
from scipy.stats import norm
from math import cos, sin , sqrt, factorial
import time
import matplotlib.pyplot as plt


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
			self.diff = self.getNDIs(self.imgAft) - self.getNDIs(self.imgBef)
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
			values = numpyArray[np.where(numpyArray <= t)]
		else:
			values = numpyArray[np.where(numpyArray >= t)]

		return np.mean(values), np.std(values)

	def pdf(self, matrix, mu, sigma):

		def fun(a):
			return norm(mu,sigma).pdf(a)

		vfun = np.vectorize(fun)
		res = vfun(matrix)
		return res

	def initial(self,alpha):
		
		self.getLimits(alpha)
		m_n, sigma_n = self.getInitialParams(self.ros,self.tn, lower=True)
		m_c, sigma_c = self.getInitialParams(self.ros,self.tc)

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

