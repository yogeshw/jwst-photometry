import numpy as np
import pytest
from astropy.io import fits
from src.psf import generate_empirical_psf, match_psf, apply_kernel

def test_generate_empirical_psf():
    # Create a mock image and sources
    image = np.random.rand(100, 100)
    sources = [{'x': 50, 'y': 50}, {'x': 60, 'y': 60}]
    
    # Generate empirical PSF
    psf = generate_empirical_psf(image, sources)
    
    # Check the shape of the PSF
    assert psf.shape == (32, 32)
    
    # Check that the PSF is normalized
    assert np.isclose(np.sum(psf), 1.0)

def test_match_psf():
    # Create mock PSFs
    psf = np.random.rand(32, 32)
    target_psf = np.random.rand(32, 32)
    
    # Match PSFs
    kernel = match_psf(psf, target_psf, 0.003)
    
    # Check the shape of the kernel
    assert kernel.shape == (32, 32)
    
    # Check that the kernel is not None
    assert kernel is not None

def test_apply_kernel():
    # Create a mock image and kernel
    image = np.random.rand(100, 100)
    kernel = np.random.rand(32, 32)
    
    # Apply kernel to the image
    matched_image = apply_kernel(image, kernel)
    
    # Check the shape of the matched image
    assert matched_image.shape == (100, 100)
    
    # Check that the matched image is not None
    assert matched_image is not None
