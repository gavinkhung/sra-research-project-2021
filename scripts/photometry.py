from astropy.io import fits
from matplotlib.colors import LogNorm
from matplotlib import pyplot as plt
from photutils import DAOStarFinder
from photutils import CircularAperture
from photutils.aperture import aperture_photometry
from astropy.time import Time
from astropy.table import Table
from glob import glob
import numpy as np


def display_fits(path):
    im, err_im, _, _ = load_image(path)
    plt.figure(figsize=(8, 8))
    display_image = np.copy(im)
    min_clip = 30
    display_image[display_image < min_clip] = min_clip + 1  # will remove the 'static' of white dots
    plt.imshow(np.rot90(display_image, k=-1), norm=LogNorm(vmin=min_clip, vmax=5000), cmap='Greys_r')
    plt.xlim((500, 1600))  # zoom in.
    plt.ylim((1900, 1200))
    plt.show()


def load_image(path):
    hdu_list = fits.open(path)
    mid_im_time = Time(hdu_list['sci'].header['date-obs'])
    return hdu_list['SCI'].data, hdu_list['ERR'].data, hdu_list['SCI'].header['EXPTIME'], mid_im_time


def locate_stars(im, err_im, min_size=10, min_sn=10):
    sn_im = im / err_im
    daofind = DAOStarFinder(fwhm=min_size, threshold=min_sn, exclude_border=True)
    return daofind(sn_im)


def restrict_sources(sources, x_limits=(1200, 1900), y_limits=(600, 1600)):
    cell = [['xcentroid', x_limits], ['ycentroid', y_limits]]
    x_bool, y_bool = [np.logical_and(sources[opt[0]] > min(opt[1]), sources[opt[0]] < max(opt[1])) for opt in cell]
    return sources[np.logical_and(x_bool, y_bool)]


def magnitude(anchored_flux, exp_time):
    return -2.5 * np.log10(anchored_flux / exp_time)


def inst_mag(anchored_flux, anchored_flux_error, exp_time):
    ft = anchored_flux / exp_time
    star_mag = magnitude(anchored_flux, exp_time)
    var_ft = anchored_flux_error ** 2 / exp_time ** 2
    var_inst_mag = var_ft * (2.5 / ft / np.log(10)) ** 2
    return star_mag, np.sqrt(var_inst_mag)


def aperture_science(sources, im, err_im, exp_time, aperture_r=30., anchor_stars_n=9):
    star_pos = [(s['xcentroid'], s['ycentroid']) for s in sources]
    aperture = CircularAperture(star_pos, r=aperture_r)
    phot_table = aperture_photometry(im, aperture, error=err_im)
    ap = 'aperture_sum'
    phot_table.sort(ap, reverse=True)
    anchored_flux = phot_table[ap][0] - np.average(phot_table[ap][1:anchor_stars_n + 1])
    anchored_flux_error = np.sqrt(
        phot_table[ap][0] ** 2 + np.sum(phot_table[ap][1:anchor_stars_n + 1] ** 2)) / anchor_stars_n ** 2
    star_mag, star_mag_error = inst_mag(anchored_flux, anchored_flux_error, exp_time)
    return anchored_flux, anchored_flux_error, star_mag, star_mag_error


def jd_to_hour(time_jd):
    return (time_jd-int(time_jd))*24%(0.3207204*24)


def process_im(path):
    im, err_im, exp_time, mid_time = load_image(path)
    sources = restrict_sources(locate_stars(im, err_im))
    flux, flux_error, star_mag, star_mag_error = aperture_science(sources, im, err_im, exp_time)
    return flux, flux_error, star_mag, star_mag_error, mid_time
