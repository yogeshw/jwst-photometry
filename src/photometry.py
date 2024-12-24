import numpy as np
import sep
from astropy.io import fits
from astropy.wcs import WCS
from photutils import CircularAperture, aperture_photometry

def perform_aperture_photometry(image, sources, photometry_params):
    """
    Perform aperture photometry on PSF-matched images.

    Parameters:
    image (numpy.ndarray): The input image data.
    sources (list): List of detected sources with their properties.
    photometry_params (dict): Parameters for aperture photometry.

    Returns:
    phot_table (Table): Aperture photometry results.
    """
    apertures = [CircularAperture((source['x'], source['y']), r=ap/2) for source in sources for ap in photometry_params['apertures']]
    phot_table = aperture_photometry(image, apertures)
    return phot_table

def correct_photometry(photometry_results, correction_params):
    """
    Correct photometry for magnification via strong gravitational lensing and derive photometric uncertainties.

    Parameters:
    photometry_results (dict): Photometry results for each band.
    correction_params (dict): Parameters for photometric corrections.

    Returns:
    corrected_results (dict): Corrected photometry results.
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

def derive_photometric_uncertainties(image, sources, correction_params):
    """
    Derive photometric uncertainties by placing circular apertures in regions outside detected sources.

    Parameters:
    image (numpy.ndarray): The input image data.
    sources (list): List of detected sources with their properties.
    correction_params (dict): Parameters for photometric corrections.

    Returns:
    sources (list): Sources with updated photometric uncertainties.
    """
    bkg = sep.Background(image)
    data_sub = image - bkg
    noise_apertures = []
    for _ in range(correction_params['num_background_apertures']):
        x = np.random.randint(0, image.shape[1])
        y = np.random.randint(0, image.shape[0])
        noise_apertures.append(CircularAperture((x, y), r=correction_params['local_noise_box_size']/2))
    
    noise_phot_table = aperture_photometry(data_sub, noise_apertures)
    noise_fluxes = noise_phot_table['aperture_sum']
    noise_fluxes = noise_fluxes[np.abs(noise_fluxes) < correction_params['outlier_sigma'] * np.std(noise_fluxes)]
    noise_std = np.std(noise_fluxes)
    
    for source in sources:
        source['flux_err'] = noise_std * np.sqrt(1 / np.median(source['weight'], axis=1))
    
    return sources
