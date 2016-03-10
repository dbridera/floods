import numpy as np
from itertools import product 
from math import sqrt,log, pow
from itertools import combinations
from scipy.stats import norm

def getSCVS(numpyArray, posX, posY):
	"""
	Get a SCV from a image.
	"""
	return numpyArray[:,posX,posY]

def imageDiff(ImgNumpyPre,imgNumpyPost):
	"""
	Compute the image difference.
	The result type depends on parameteers type.
	"""
	return imgNumpyPost - ImgNumpyPre

def getMagnitude(imgDiff, sizeX , sizeY):
		result = np.empty([sizeX,sizeY])
		for pos in product(range(sizeX),range(sizeY)):
			posX , posY = pos[0], pos[1]
			result[posX][posY] = sqrt(np.sum((getSCVS(imgDiff, posX , posY))**2))
		return result

def getMagnitudeMatrix(imgDiff):
		pre = imgDiff**2
		return np.sqrt(np.sum(pre, axis=0))

def getAlphas(imgDiff, sizeX , sizeY):
		result = np.empty([sizeX,sizeY])
		for pos in product(range(sizeX),range(sizeY)):
			posX , posY = pos[0], pos[1]
			numeral= sqrt(np.sum((getSCVS(imgDiff, posX , posY))**2))
			denominator = np.sum(getSCVS(imgDiff, posX , posY))
			result[posX][posY] = numeral/denominator
		return result

def getAlphaMatrix(imgDiff, denominator):
		
		numerator = np.sum(imgDiff, axis=0)
		with np.errstate(invalid='ignore',divide='ignore'):
			j = numerator/denominator
			j[~np.isfinite(j)] = 0
			return j


def getInitialParams(numpyArray, t, lower=False):
	if lower:
		values = numpyArray[np.where(numpyArray <= t)]
	else:
		values = numpyArray[np.where(numpyArray >= t)]

	return np.mean(values), np.std(values)


def getNDIs(matrix):
	bands, x , y = matrix.shape
	to_combine = combinations(range(bands),2)

	aux = []

	for e in to_combine:
		aux.append(e)

	res = np.empty([len(aux), x , y])

	for i, (b1, b2) in enumerate(aux):

		img1 = matrix[b1,:,:]
		img2 = matrix[b2,:,:]
		with np.errstate(invalid='ignore',divide='ignore'):
			res[i,:,:]= (img1 - img2) / (img1 + img2)

	res[~np.isfinite(res)]= 0

	return res

def pdf(matrix, mu, sigma):

	def fun(a):
		return norm(mu,sigma).pdf(a)

	vfun = np.vectorize(fun)

	res = vfun(matrix)

	return res

def computeRoots(mu_n, mu_c, var_n, var_c, Pn, Pc):
	a = var_n - var_c
	b = 2 ( (var_c*mu_n) - (var_n*mu_c)
	
	c1 = var_n * pow(mu_c,2)
	c2 = -var_c*(mu_n**2)
	c3 = -2*(var_c*var_n)* log( (sqrt(var_n)* Pc) / (sqrt(var_c)* Pn) )

	c = c1 + c2 + c3

	expr = (b**2) - (4*a*c)  
	denominator = 2*a

	if expr>0:
		return (-b - sqrt(expr))/denominator , (-b + sqrt(expr))/denominator
	else:
		print "No hay raices reales"

print computeRoots(0.012015065332291968, 0.032281068167829857, 1.347091345656299e-05, 0.00017052328317002674, 0.95092060861307237, 0.049079391386927629)

def test():
	""" Matrix examples	"""
	a = np.array([ [ [6,7,8] , [9,11,15] ] , [ [11,12,13] , [14,15,16] ] ])
	b = np.array([ [ [1,2,3] , [4,5,6] ] , [ [3,2,1] , [6,5,4] ] ])
	c = np.array([ [ [5,5,5] , [5,6,9] ] , [ [8,10,12] , [8,10,12] ] ])
	d = np.sqrt(np.array([ [89,125,169] , [89,136,225] ]))
	
	dif = imageDiff(b,a)
	print "Difference image"
	print dif
	print "Check"
	print np.array_equal(dif, c)
	ros = getMagnitudeMatrix(dif)
	print "Ros matrix"
	print ros
	print np.array_equal(ros,d)
"""if __name__ == "__main__":
	a = np.array([ [ [6,7,8] , [9,11,15] ] , [ [11,12,13] , [14,15,16] ] ])
	b = np.array([ [ [1,2,3] , [4,5,6] ] , [ [3,2,1] , [6,5,4] ] ])
	print getNDIs(a).shape
	print getNDIs(b).shape
	
"""