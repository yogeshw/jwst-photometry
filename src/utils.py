import numpy as np
import yaml
from astropy.io import fits
import sep

def read_image(file_path):
    """
    Read an image file and return the data.

    Parameters:
    file_path (str): Path to the image file.

    Returns:
    numpy.ndarray: Image data.
    """
    with fits.open(file_path) as hdul:
        image_data = hdul[0].data
    return image_data

def save_catalog(catalog, format, output_directory):
    """
    Save the photometry catalog in the specified output format.

    Parameters:
    catalog (Table): Photometry catalog.
    format (str): Output format (e.g., 'fits').
    output_directory (str): Directory to save the catalog.
    """
    if format == 'fits':
        save_catalog_fits(catalog, output_directory)
    else:
        raise ValueError(f"Unsupported catalog format: {format}")

def save_catalog_fits(catalog, output_directory):
    """
    Save the photometry catalog in FITS format.

    Parameters:
    catalog (Table): Photometry catalog.
    output_directory (str): Directory to save the catalog.
    """
    hdu = fits.BinTableHDU.from_columns(catalog)
    hdu.writeto(f"{output_directory}/photometry_catalog.fits", overwrite=True)

def load_config(config_file):
    """
    Load user inputs from a YAML configuration file.

    Parameters:
    config_file (str): Path to the YAML configuration file.

    Returns:
    dict: Configuration parameters.
    """
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def process_image(image_data):
    """
    Process the image data (e.g., background subtraction, normalization).

    Parameters:
    image_data (numpy.ndarray): Input image data.

    Returns:
    numpy.ndarray: Processed image data.
    """
    # Example processing: background subtraction
    background = sep.Background(image_data)
    processed_image = image_data - background.back()
    return processed_image

def save_results(results, output_file):
    """
    Save the results to a specified output file.

    Parameters:
    results (dict): Results to be saved.
    output_file (str): Path to the output file.
    """
    with open(output_file, 'w') as file:
        yaml.dump(results, file)
