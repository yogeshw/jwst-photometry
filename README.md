# JWST Photometry Script

This project provides a Python script to carry out JWST photometry in the following bands: F115W, F150W, F200W, F277W, F356W, F410M, and F444W. The script includes source detection, PSF homogenization, aperture photometry, corrections to photometry accounting for magnification via strong gravitational lensing, identification of star candidates, and a general recommended "use" flag. The software used to produce these catalogs, aperpy, is generally applicable to any JWST/NIRCam data and is freely available.

## Purpose

The purpose of this project is to provide a comprehensive tool for performing photometry on JWST images, ensuring consistent photometric measurements across all bands, thus enabling accurate source colors, redshifts, and physical parameters.

## How to Run the Script

1. Install the necessary dependencies listed in `requirements.txt`.
2. Modify the `config.yaml` file to specify the parameters for source detection, PSF homogenization, aperture photometry, and photometric corrections.
3. Run the main script using the command line:
   ```
   python src/main.py
   ```

## Modifying the YAML File

The `config.yaml` file contains all the user inputs required for the script. Users should modify this file to set the desired parameters for their specific use case. The YAML file includes parameters for:

- Source detection
- PSF homogenization
- Aperture photometry
- Photometric corrections

For detailed instructions on how to install dependencies, run, and configure the script, please refer to the separate LaTex user guide.
