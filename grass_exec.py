#!/usr/bin/env python
import math 
import os
import sys
import glob
import subprocess
import random
 
# path to the GRASS GIS launch script
# MS Windows
grass7bin_win = r'C:\OSGeo4W\bin\grass70svn.bat'
# uncomment when using standalone WinGRASS installer
# grass7bin_win = r'C:\Program Files (x86)\GRASS GIS 7.0.0beta3\grass70.bat'
# Linux
grass7bin_lin = 'grass70'
# Mac OS X
# this is TODO
grass7bin_mac = '/Applications/GRASS/GRASS-7.0.app/'
 
# DATA
# define GRASS DATABASE
# add your path to grassdata (GRASS GIS database) directory
gisdb = os.path.join(os.path.expanduser("~"), "gisbase")
#gisdb = "/home/daniel/grassdata"

# the following path is the default path on MS Windows
# gisdb = os.path.join(os.path.expanduser("~"), "Documents/grassdata")
 
# specify (existing) location and mapset
location = "test"
mapset   = "PERMANENT"
 
 
########### SOFTWARE
if sys.platform.startswith('linux'):
    # we assume that the GRASS GIS start script is available and in the PATH
    # query GRASS 7 itself for its GISBASE
    grass7bin = grass7bin_lin
elif sys.platform.startswith('win'):
    grass7bin = grass7bin_win
else:
    raise OSError('Platform not configured.')
                                                                                                            
# query GRASS 7 itself for its GISBASE
startcmd = [grass7bin, '--config', 'path']
 
gisbase = '/usr/lib/grass70'
 
# Set GISBASE environment variable
os.environ['GISBASE'] = gisbase
# the following not needed with trunk
os.environ['PATH'] += os.pathsep + os.path.join(gisbase, 'extrabin')
# add path to GRASS addons
home = os.path.expanduser("~")
os.environ['PATH'] += os.pathsep + os.path.join(home, '.grass7', 'addons', 'scripts')
 
# define GRASS-Python environment
gpydir = os.path.join(gisbase, "etc", "python")
sys.path.append(gpydir)
 
########### DATA
# Set GISDBASE environment variable
os.environ['GISDBASE'] = gisdb
 
# import GRASS Python bindings (see also pygrass)
import grass.script as grass
import grass.script.setup as gsetup

import atexit

gsetup.init(gisbase, gisdb, location, mapset)


from grass.pygrass.modules.shortcuts import general as g

import math
from utc_to_esd import AcquisitionTime, jd_to_esd

# globals -------------------------------------------------------------------
acq_tim = ''
tmp = ''
tmp_rad = ''
tmp_toar = ''


# constants ----------------------------------------------------------------
"""
Factors for Conversion to Top-of-Atmosphere Spectral Radiance
    (absolute radiometric calibration factors)

Structure of the dictionary:
- Key: Name of band
- Items in tupple(s):
    - 1st: Absolute Calibration Factors
    - 2nd: Spectral Band's Effective Bandwidth, 
    - 3rd: Band-Averaged Solar Spectral Irradiance [W/sq.m./micro-m]

Retrieving values:
    band = <Name of Band>
    K[band][0]  # for Effective Bandwidth
    K[band][1]  # for Esun
    K[band][2]  # for Absolute Conversion Factor
"""

CF_BW_ESUN = {
    'Pan':     (0.28460000, 1580.8140, 0.056783450),
    'Coastal': (0.04730000, 1758.2229, 0.009295654),
    'Blue':    (0.05430000, 1974.2416, 0.012608250),
    'Green':   (0.06300000, 1856.4104, 0.009713071),
    'Yellow':  (0.03740000, 1738.4791, 0.005829815),
    'Red':     (0.05740000, 1559.4555, 0.011036230),
    'RedEdge': (0.03930000, 1342.0695, 0.005188136),
    'NIR1':    (0.09890000, 1069.7302, 0.012243800),
    'NIR2':    (0.09960000,  861.2866, 0.009042234)}

spectral_bands = CF_BW_ESUN.keys()

# string for metadata
source1_rad = source1_toar = '''"Radiometric Use of WorldView-2 Imagery,
                                 Technical Note (2010)", by Todd Updike &
                                 Chris Comp'''

source2_rad = source2_toar = ""


# helper functions ----------------------------------------------------------
def cleanup():
    grass.run_command('g.remove', flags='f', type="rast",
                      pattern='tmp.%s*' % os.getpid(), quiet=True)


def run(cmd, **kwargs):
    """ """
    grass.run_command(cmd, quiet=True, **kwargs)


def main():

    global acq_time, esd

    """1st, get input, output, options and flags"""

    spectral_bands = options['band'].split(',')
    outputsuffix = options['outputsuffix']
    utc = options['utc']
    doy = options['doy']
    sea = options['sea']

    radiance = flags['r']
    if radiance and outputsuffix == 'toar':
        outputsuffix = 'rad'
        g.message("Output-suffix set to %s" % outputsuffix)

    keep_region = flags['k']
    info = flags['i']

    # -----------------------------------------------------------------------
    # Equations
    # -----------------------------------------------------------------------

    if info:
        # conversion to Radiance based on (1)
        msg = "|i Spectral Radiance = K * DN / Effective Bandwidth | " \
              "Reflectance = ( Pi * Radiance * ESD^2 ) / BAND_Esun * cos(SZA)"
        g.message(msg)

    # -----------------------------------------------------------------------
    # List images and their properties
    # -----------------------------------------------------------------------

    mapset = grass.gisenv()['MAPSET']  # Current Mapset?

#    imglst = [pan]
#    imglst.extend(msxlst)  # List of input imagery

    
    # -----------------------------------------------------------------------
    # Temporary Region and Files
    # -----------------------------------------------------------------------

    if not keep_region:
        grass.use_temp_region()  # to safely modify the region
    tmpfile = grass.tempfile()  # Temporary file - replace with os.getpid?
    tmp = "tmp." + grass.basename(tmpfile)  # use its basename

    # -----------------------------------------------------------------------
    # Global Metadata: Earth-Sun distance, Sun Zenith Angle
    # -----------------------------------------------------------------------

    # Earth-Sun distance
    if doy:
        g.message("|! Using Day of Year to calculate Earth-Sun distance.")
        esd = jd_to_esd(int(doy))

    elif (not doy) and utc:
        acq_utc = AcquisitionTime(utc)  # will hold esd (earth-sun distance)
        esd = acq_utc.esd

    else:
        grass.fatal(_("Either the UTC string or "
                      "the Day-of-Year (doy) are required!"))

    sza = 90 - float(sea)  # Sun Zenith Angle based on Sun Elevation Angle

    # -----------------------------------------------------------------------
    # Loop processing over all bands
    # -----------------------------------------------------------------------
    for band in spectral_bands:

        global tmp_rad

        # -------------------------------------------------------------------
        # Match bands region if... ?
        # -------------------------------------------------------------------

        if not keep_region:
            run('g.region', rast=band)   # ## FixMe?
            msg = "\n|! Region matching the %s spectral band" % band
            g.message(msg)

        elif keep_region:
            msg = "|! Operating on current region"
            g.message(msg)

        # -------------------------------------------------------------------
        # Band dependent metadata for Spectral Radiance
        # -------------------------------------------------------------------

        g.message("\n|* Processing the %s band" % band, flags='i')

        # Why is this necessary?  Any function to remove the mapsets name?
        if '@' in band:
            band = (band.split('@')[0])

        # get absolute calibration factor
        acf = float(CF_BW_ESUN[band][2])
        acf_msg = "K=" + str(acf)

        # effective bandwidth
        bw = float(CF_BW_ESUN[band][0])

        # -------------------------------------------------------------------
        # Converting to Spectral Radiance
        # -------------------------------------------------------------------

        msg = "\n|> Converting to Spectral Radiance " \
              "| Conversion Factor %s, Bandwidth=%.3f" % (acf_msg, bw)
        g.message(msg)

        # convert
        tmp_rad = "%s.Radiance" % tmp  # Temporary Map
        rad = "%s = %f * %s / %f" \
            % (tmp_rad, acf, band, bw)  # Attention: 32-bit calculations requ.
        grass.mapcalc(rad, overwrite=True)

        # strings for metadata
        history_rad = rad
        history_rad += "Conversion Factor=%f; Effective Bandwidth=%.3f" \
            % (acf, bw)
        title_rad = ""
        description_rad = "Top-of-Atmosphere %s band spectral Radiance " \
                          "[W/m^2/sr/um]" % band
        units_rad = "W / sq.m. / um / ster"

        if not radiance:

            # ---------------------------------------------------------------
            # Converting to Top-of-Atmosphere Reflectance
            # ---------------------------------------------------------------

            global tmp_toar

            msg = "\n|> Converting to Top-of-Atmosphere Reflectance"
            g.message(msg)

            esun = float(CF_BW_ESUN[band][1])
            msg = "   %s band mean solar exoatmospheric irradiance=%.2f" \
                % (band, esun)
            g.message(msg)

            # convert
            tmp_toar = "%s.Reflectance" % tmp  # Spectral Reflectance
            toar = "%s = %f * %s * %f^2 / %f * cos(%f)" \
                % (tmp_toar, math.pi, tmp_rad, esd, esun, sza)
            grass.mapcalc(toar, overwrite=True)

            # report range? Using a flag and skip actual conversion?
            # todo?

            # strings for metadata
            title_toar = "%s band (Top of Atmosphere Reflectance)" % band
            description_toar = "Top of Atmosphere %s band spectral Reflectance" \
                % band
            units_toar = "Unitless planetary reflectance"
            history_toar = "K=%f; Bandwidth=%.1f; ESD=%f; Esun=%.2f; SZA=%.1f" \
                % (acf, bw, esd, esun, sza)

        if tmp_toar:

            # history entry
            run("r.support", map=tmp_toar, title=title_toar,
                units=units_toar, description=description_toar,
                source1=source1_toar, source2=source2_toar,
                history=history_toar)

            # add suffix to basename & rename end product
            toar_name = ("%s.%s" % (band.split('@')[0], outputsuffix))
            run("g.rename", rast=(tmp_toar, toar_name))

        elif tmp_rad:

            # history entry
            run("r.support", map=tmp_rad,
                title=title_rad, units=units_rad, description=description_rad,
                source1=source1_rad, source2=source2_rad, history=history_rad)

            # add suffix to basename & rename end product
            rad_name = ("%s.%s" % (band.split('@')[0], outputsuffix))
            run("g.rename", rast=(tmp_rad, rad_name))

    # visualising-related information
    if not keep_region:
        grass.del_temp_region()  # restoring previous region settings
    g.message("\n|! Region's resolution restored!")
    g.message("\n>>> Hint: rebalancing colors "
              "(i.colors.enhance) may improve appearance of RGB composites!",
              flags='i')

if __name__ == "__main__":
    options, flags = grass.parser()
    atexit.register(cleanup)
    sys.exit(main())
