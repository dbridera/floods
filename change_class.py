from osgeo import gdal, gdal_array
import numpy as np
from utils import *
from scipy.stats import norm
from math import cos, sin, sqrt, factorial, pi, radians
import time
import matplotlib.pyplot as plt

from calibration.parser import BAND_MAP, getMetadata, earthSunDistance


class ChangeImage(object):

    """
        Class for compute the changes between two images

    """

    def __init__(self, imgBef, imgAft):

        """
            Create an ChangeImage object

        """

        super(ChangeImage, self).__init__()
        self.imgBef = imgBef
        self.imgAft = imgAft
        self.diff = None
        self.ros = None
        self.alpha = None
        self.coef = None
        self.bands, self.x, self.y = None, None, None
        self.tn, self.tc = None, None

    def load(self):

        """
            Load the two initial images as numpy arrays.

        """

        self.imgBef = gdal.Open(self.imgBef).ReadAsArray().astype('float')
        self.imgAft = gdal.Open(self.imgAft).ReadAsArray().astype('float')

        nozero = self.imgBef[np.where(self.imgBef != 0)]
        self.imgBef[np.where(self.imgBef == 0)] = nozero.min()

        nozero = self.imgAft[np.where(self.imgAft != 0)]
        self.imgAft[np.where(self.imgAft == 0)] = nozero.min()

        self.bands, self.x, self.y = self.imgAft.shape
        self.coef = sqrt(
            factorial(self.bands) / (2 * factorial(self.bands - 2)))

    def getNDIs(self, matrix):

        """
            Function to get all indexes for each
            band combination of input matrix.

        """

        to_combine = combinations(range(self.bands), 2)

        aux = []

        for e in to_combine:
            aux.append(e)

        res = np.empty([len(aux), self.x, self.y])

        for i, (b1, b2) in enumerate(aux):

            img1 = matrix[b1, :, :]
            img2 = matrix[b2, :, :]
            res[i, :, :] = (img1 - img2) / (img1 + img2)

        return res

    def selectedNDI(self, matrix, combinatios):

        """
            Create indexes using only the specified
            combinations

        """

        res = np.empty([len(combinatios), self.x, self.y])

        for i, (b1, b2) in enumerate(combinatios):

            img1 = matrix[b1, :, :]
            img2 = matrix[b2, :, :]
            res[i, :, :] = (img1 - img2) / (img1 + img2)

        return res

    def imageDiff(self, source='origin', combinations=None):

        """
            Compute the image difference.
            The result type depends on parameteers type.

        """

        if source == 'origin':
            self.diff = self.imgAft - self.imgBef
        
        elif source == 'ndi':
            self.diff = self.getNDIs(self.imgAft) - self.getNDIs(self.imgBef)
            self.bands, self.x, self.y = self.diff.shape
            self.coef = sqrt(self.bands)
        
        elif source == 'combinations':
            self.diff = self.selectedNDI(self.imgAft, combinations) - \
                self.selectedNDI(self.imgBef, combinations)
            self.bands, self.x, self.y = self.diff.shape
            self.coef = sqrt(len(combinations))

    def calibrate(self, metaFile):

        """
            Function to change a WorldView 2 image
            from DN to TOA reflectance.

        """

        data = getMetadata(metaFile)
        date = data['date']

        sunDist = earthSunDistance(date)
        tita_s = radians(90 - data['sunEl'])

        for b_number, b_name in BAND_MAP:

            band_digital = self.imgBef[int(b_number), :, :]
            absFactor = data[b_name]['absCalFactor']
            effBanw = data[b_name]['effectiveBandwidth']
            self.imgBef[int(b_number), :, :] = (
                float(absFactor) * band_digital) / float(effBanw)

        for b_number, b_name in BAND_MAP:

            band_radiance = self.imgBef[b_number, :, :]
            esun = data[b_name]['solarirr']

            self.imgBef[b_number, :, :] = (
                pi * band_radiance * (sunDist ** 2)) / (esun * cos(2 * tita_s))

        for b_number, b_name in BAND_MAP:

            band_digital = self.imgAft[int(b_number), :, :]
            absFactor = data[b_name]['absCalFactor']
            effBanw = data[b_name]['effectiveBandwidth']
            self.imgAft[int(b_number), :, :] = (
                float(absFactor) * band_digital) / float(effBanw)

        for b_number, b_name in BAND_MAP:

            band_radiance = self.imgAft[b_number, :, :]
            esun = data[b_name]['solarirr']

            self.imgAft[b_number, :, :] = (
                pi * band_radiance * (sunDist ** 2)) / (esun * cos(2 * tita_s))

    def getMagnitudeMatrix(self):

        """
            Get all RO vectors for a matrix

        """

        pre = self.diff ** 2
        self.ros = np.sqrt(np.sum(pre, axis=0))

    def fixRos(self):

        """
            Normalize RO vectors

        """

        self.ros = (1 / (2 * self.coef)) * self.ros

    def getAlphaMatrix(self):

        """
            Get all alpha vectors for a matrix

        """

        numerator = np.sum(self.diff, axis=0)
        self.alpha = numerator / (self.coef * self.ros)

    def getLimits(self, middle, alpha):

        """
            Get initial limits for classes

        """

        #middle = (self.ros.max() - self.ros.min()) / 2
        self.tn, self.tc = middle - alpha, middle + alpha

    def getInitialParams(self, numpyArray, t, lower=False):

        """
            Get initial params for iterations

        """

        if lower:
            values = numpyArray[np.where(numpyArray <= t)]
        else:
            values = numpyArray[np.where(numpyArray >= t)]

        return np.mean(values), np.std(values)

    def pdf(self, matrix, mu, sigma):

        """
            Get pdf for each element in matrix

        """

        def fun(a):
            return norm(mu, sigma).pdf(a)

        vfun = np.vectorize(fun)
        res = vfun(matrix)
        return res

    def initial(self, alpha):

        """
            Get initial limits for each image.

        """

        self.getLimits(alpha)
        m_n, sigma_n = self.getInitialParams(self.ros, self.tn, lower=True)
        m_c, sigma_c = self.getInitialParams(self.ros, self.tc)

        return m_n, sigma_n, m_c, sigma_c

    def computeRatio(self, Pc, Pnc, mu_c, sigma_c, mu_nc, sigma_nc):

        """
            Perform ratio useful for equations

        """

        Pxn = self.pdf(self.ros, mu_nc, sigma_nc)
        Pxc = self.pdf(self.ros, mu_c, sigma_c)

        Pn_Pxn = Pnc * Pxn
        Pc_Pxc = Pc * Pxc

        Px = Pn_Pxn + Pc_Pxc
        ratio1 = Pn_Pxn / Px
        ratio2 = Pc_Pxc / Px

        """
        with np.errstate(invalid='ignore',divide='ignore'):
            ratio1 = Pn_Pxn/Px
            ratio2 = Pc_Pxc/Px
            ratio1[~np.isfinite(ratio1)] = 0
            ratio2[~np.isfinite(ratio2)] = 0
        """
        return ratio1, ratio2

    def equation1(self, ratioPn, ratioPc):

        """
            Perform equation 1

        """

        # esto me viene, para no calcularlo de nuevo!!!
        #newPn , newPc = computeRatio(Pc, Pnc, mu_c, sigma_c, mu_nc, sigma_nc)
        denominator = (self.x * self.y)
        #denominator = 6262
        return np.sum(ratioPn) / denominator, np.sum(ratioPc) / denominator

    def equation2(self, ratioPn, ratioPc):

        """
            Perform equation 1

        """

        num_n = np.sum(ratioPn * self.ros)
        num_c = np.sum(ratioPc * self.ros)

        return num_n / np.sum(ratioPn), num_c / np.sum(ratioPc)

    def equation3(self, ratioPn, ratioPc, mu_nc,  mu_c):

        """
            Perform equation 3

        """

        num_n = np.sum(ratioPn * ((self.ros - mu_nc) ** 2))
        num_c = np.sum(ratioPc * ((self.ros - mu_c) ** 2))

        return num_n / np.sum(ratioPn), num_c / np.sum(ratioPc)

    def computeRatio_aux(self,ros, Pc, Pnc, mu_c, sigma_c, mu_nc, sigma_nc):

        """
            Perform ratio, useful for equations

        """

        Pxn = self.pdf(ros, mu_nc, sigma_nc)
        Pxc = self.pdf(ros, mu_c, sigma_c)

        Pn_Pxn = Pnc * Pxn
        Pc_Pxc = Pc * Pxc

        Px = Pn_Pxn + Pc_Pxc
        ratio1 = Pn_Pxn / Px
        ratio2 = Pc_Pxc / Px

        """
        with np.errstate(invalid='ignore',divide='ignore'):
            ratio1 = Pn_Pxn/Px
            ratio2 = Pc_Pxc/Px
            ratio1[~np.isfinite(ratio1)] = 0
            ratio2[~np.isfinite(ratio2)] = 0
        """
        return ratio1, ratio2

    def equation1_aux(self, denominator, ratioPn, ratioPc):

            """
                Perform equation 1

            """

            # esto me viene, para no calcularlo de nuevo!!!
            #newPn , newPc = computeRatio(Pc, Pnc, mu_c, sigma_c, mu_nc, sigma_nc)
            #denominator = (self.x * self.y)
            #denominator = 6262
            return np.sum(ratioPn) / denominator, np.sum(ratioPc) / denominator

    def equation2_aux(self, ros, ratioPn, ratioPc):

        """
            Perform equation 1

        """

        num_n = np.sum(ratioPn * ros)
        num_c = np.sum(ratioPc * ros)

        return num_n / np.sum(ratioPn), num_c / np.sum(ratioPc)

    def equation3_aux(self, ros, ratioPn, ratioPc, mu_nc,  mu_c):

        """
            Perform equation 3

        """

        num_n = np.sum(ratioPn * ((ros - mu_nc) ** 2))
        num_c = np.sum(ratioPc * ((ros - mu_c) ** 2))

        return num_n / np.sum(ratioPn), num_c / np.sum(ratioPc)
