import re
from dateutil import parser
from astropy.time import Time
import time
from math import cos, radians, pi

BAND_MAP = [
(0, 'BAND_C'),
(1, 'BAND_B'),
(2, 'BAND_G'),
(3, 'BAND_Y'),
(4, 'BAND_R'),
(5, 'BAND_RE'),
(6, 'BAND_N'),
(7, 'BAND_N2')
]

SOLAR_SPECTRAL_IRRADIANCE = {
'BAND_C' : 1758.2229,
'BAND_B' : 1974.2416,
'BAND_G' : 1856.4104,
'BAND_Y' : 1738.4791,
'BAND_R' : 1559.4555,
'BAND_RE' : 1342.0695,
'BAND_N' : 1069.7302,
'BAND_N2' :  861.2866
}

def getMetadata(file):
	data = {}

	with open (file, "r") as myfile:
		lines = myfile.read()


	pattern = 'BEGIN_GROUP = (BAND_[A-Z0-9]+)[^_]+_GROUP = BAND_[A-Z0-9]+'

	date = re.search('firstLineTime =\s*(.+);', lines).group(1)
	sunEl = re.search('meanSunEl =\s*(.+);', lines).group(1)

	data['date'] = date
	data['sunEl'] = float(sunEl)

	res = re.finditer(pattern, lines)


	for e in res:
		#new[e.group(1)] = e.group(0)
		text = e.group(0)
		factor = re.search('absCalFactor =\s*(.+);', text).group(1)
		ban = re.search('effectiveBandwidth =\s*(.+);', text).group(1)

		data[e.group(1)] = {'absCalFactor' : factor, 'effectiveBandwidth' : ban ,
							'solarirr': SOLAR_SPECTRAL_IRRADIANCE[e.group(1)]}



	return data


def earthSunDistance(date):
	t = Time(date)
	# Julian day
	
	jd = t.jd
	d = jd - 2451545.0
	
	g = radians( 357.529 + (0.98560028 * d))
	
	distance = 1.00014 - 0.01671  * cos(g) - 0.00014 * cos(2* g)

	return distance

def calibrate(metaFile, img):

	# TODO: check image shape (q-.-)q

	data = getMetadata(metaFile)

	date = data['date']

	sunDist = earthSunDistance(date)

	tita_s = radians(90 - data['sunEl'])

	for b_name, b_number in BAND_MAP:

		band_digital = img[b_number,:,:]
		absFactor = data[b_name]['absCalFactor']
		effBanw = data[b_name]['effectiveBandwidth']
		img[b_number,:,:] = (absFactor * band_digital)/ effBanw 

	for b_name, b_number in BAND_MAP:

		band_radiance = img[b_number,:,:]
		esun = data[b_name]['solarirr']
		
		img[b_number,:,:] = (pi * band_radiance * (sunDist**2) )/ (esun * cos(2*tita_s)) 


if __name__ == '__main__':
	print earthSunDistance('2009-10-08T18:51:00.000000Z')

