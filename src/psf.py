import numpy as np
from astropy.io import fits
from astropy.nddata import Cutout2D
from scipy.ndimage import center_of_mass, zoom
from scipy.signal import convolve2d
import pypher

def generate_empirical_psf(image, sources, stamp_size=32):
    """
    Generate empirical PSFs for JWST/NIRCam bands using stars identified within the FOV.

    Parameters:
    image (numpy.ndarray): The input image data.
    sources (list): List of detected sources with their properties.
    stamp_size (int): Size of the stamp to extract around each source.

    Returns:
    numpy.ndarray: The generated empirical PSF.
    """
    psf_stamps = []
    for source in sources:
        x, y = source['x'], source['y']
        cutout = Cutout2D(image, (x, y), stamp_size)
        stamp = cutout.data
        com = center_of_mass(stamp)
        shift_x, shift_y = (stamp_size // 2) - com[1], (stamp_size // 2) - com[0]
        stamp = np.roll(stamp, int(shift_x), axis=1)
        stamp = np.roll(stamp, int(shift_y), axis=0)
        stamp /= np.sum(stamp)
        psf_stamps.append(stamp)
    psf_stamps = np.array(psf_stamps)
    empirical_psf = np.mean(psf_stamps, axis=0)
    return empirical_psf

def match_psf(psf, target_psf, regularization_parameter):
    """
    Match PSFs using Pypher and apply the resulting kernels to the images.

    Parameters:
    psf (numpy.ndarray): The PSF to be matched.
    target_psf (numpy.ndarray): The target PSF to match to.
    regularization_parameter (float): Regularization parameter for Pypher.

    Returns:
    numpy.ndarray: The PSF matching kernel.
    """
    kernel = pypher.psf_match(psf, target_psf, regularization_parameter)
    return kernel

def apply_kernel(image, kernel):
    """
    Apply the PSF matching kernel to the image.

    Parameters:
    image (numpy.ndarray): The input image data.
    kernel (numpy.ndarray): The PSF matching kernel.

    Returns:
    numpy.ndarray: The PSF-matched image.
    """
    matched_image = convolve2d(image, kernel, mode='same')
    return matched_image
