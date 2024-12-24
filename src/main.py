import numpy as np
import sep
import yaml
from astropy.io import fits
from astropy.wcs import WCS
from photutils import CircularAperture, aperture_photometry
from psf import generate_empirical_psf, match_psf
from photometry import perform_aperture_photometry, correct_photometry
from detection import detect_sources, generate_segmentation_map, identify_star_candidates
from utils import read_image, save_catalog

def main():
    # Load user inputs from config.yaml
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Read images
    images = {}
    for band in ['F115W', 'F150W', 'F200W', 'F277W', 'F356W', 'F410M', 'F444W']:
        images[band] = read_image(f'data/{band}.fits')

    # Source detection
    detection_params = config['source_detection']
    sources, segmentation_map = detect_sources(images['F444W'], detection_params)

    # PSF homogenization
    target_psf = generate_empirical_psf(images['F444W'], sources)
    for band in images:
        if band != 'F444W':
            psf = generate_empirical_psf(images[band], sources)
            kernel = match_psf(psf, target_psf, config['psf_homogenization']['regularization_parameter'])
            images[band] = apply_kernel(images[band], kernel)

    # Aperture photometry
    photometry_params = config['aperture_photometry']
    photometry_results = {}
    for band in images:
        photometry_results[band] = perform_aperture_photometry(images[band], sources, photometry_params)

    # Photometric corrections
    corrected_photometry = correct_photometry(photometry_results, config['photometric_corrections'])

    # Save the resulting photometry catalog
    save_catalog(corrected_photometry, config['output']['catalog_format'], config['output']['output_directory'])

if __name__ == '__main__':
    main()
