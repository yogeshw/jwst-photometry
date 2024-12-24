import numpy as np
import yaml
from astropy.io import fits

def read_image(file_path):
    """
    Read an image file and return the data.
    """
    with fits.open(file_path) as hdul:
        image_data = hdul[0].data
    return image_data

def save_catalog(catalog, format, output_directory):
    """
    Save the photometry catalog in the specified output format.
    """
    if format == 'fits':
        save_catalog_fits(catalog, output_directory)
    else:
        raise ValueError(f"Unsupported catalog format: {format}")

def save_catalog_fits(catalog, output_directory):
    """
    Save the photometry catalog in FITS format.
    """
    hdu = fits.BinTableHDU.from_columns(catalog)
    hdu.writeto(f"{output_directory}/photometry_catalog.fits", overwrite=True)

def load_config(config_file):
    """
    Load user inputs from a YAML configuration file.
    """
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config
