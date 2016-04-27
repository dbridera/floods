import re
from dateutil import parser
from astropy.time import Time
import time
from math import cos, radians

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
	data['sunEl'] = sunEl

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
	
	print jd
	
	d = jd - 2451545.0
	print d
	
	g = radians( 357.529 + (0.98560028 * d))
	
	distance = 1.00014 - 0.01671  * cos(g) - 0.00014 * cos(2* g)

	return distance
	
if __name__ == '__main__':
	print earthSunDistance('2009-10-08T18:51:00.000000Z')
