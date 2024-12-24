import numpy as np
import sep
from astropy.io import fits
from astropy.wcs import WCS
from photutils import CircularAperture, aperture_photometry

def perform_aperture_photometry(image, sources, photometry_params):
    """
    Perform aperture photometry on PSF-matched images.
    """
    apertures = [CircularAperture((source['x'], source['y']), r=ap/2) for source in sources for ap in photometry_params['apertures']]
    phot_table = aperture_photometry(image, apertures)
    return phot_table

def correct_photometry(photometry_results, correction_params):
    """
    Correct photometry for magnification via strong gravitational lensing and derive photometric uncertainties.
    """
    corrected_results = {}
    for band, phot_table in photometry_results.items():
        # Apply corrections to photometry
        if correction_params['f444w_curve_of_growth']:
            phot_table['flux'] /= correction_params['f444w_curve_of_growth']
        
        # Derive photometric uncertainties
        local_noise = np.sqrt(1 / np.median(phot_table['weight'], axis=1))
        phot_table['flux_err'] = phot_table['flux'] * local_noise
        
        corrected_results[band] = phot_table
    return corrected_results
