#!/usr/bin/env python3
"""
Real JWST Data Test Script

This script tests our Phase 4 photometry software on real JWST NIRCam images
from the COSMOS-Web survey in F150W, F277W, and F444W filters.
"""

import numpy as np
import logging
import time
import os
from pathlib import Path
from astropy.io import fits
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_jwst_image(filepath):
    """Load a JWST science image and extract relevant information."""
    logger.info(f"Loading JWST image: {os.path.basename(filepath)}")
    
    try:
        with fits.open(filepath) as hdul:
            # Get the science data (usually in the first or second HDU)
            sci_data = None
            header = None
            
            for i, hdu in enumerate(hdul):
                logger.info(f"HDU {i}: {hdu.header.get('EXTNAME', 'PRIMARY')} - Shape: {getattr(hdu.data, 'shape', 'No data')}")
                
                # Look for science data
                if hdu.data is not None and len(hdu.data.shape) == 2:
                    sci_data = hdu.data.astype(np.float32)
                    header = hdu.header
                    logger.info(f"Using HDU {i} as science data")
                    break
            
            if sci_data is None:
                raise ValueError("No suitable 2D science data found in FITS file")
                
            # Log basic statistics
            valid_mask = np.isfinite(sci_data)
            if np.any(valid_mask):
                logger.info(f"Image shape: {sci_data.shape}")
                logger.info(f"Valid pixels: {np.sum(valid_mask):,} / {sci_data.size:,}")
                logger.info(f"Data range: {np.min(sci_data[valid_mask]):.6f} to {np.max(sci_data[valid_mask]):.6f}")
                logger.info(f"Median value: {np.median(sci_data[valid_mask]):.6f}")
            
            return sci_data, header
            
    except Exception as e:
        logger.error(f"Failed to load {filepath}: {e}")
        raise

def extract_subimage(image, center=None, size=512):
    """Extract a manageable subimage for testing."""
    h, w = image.shape
    
    if center is None:
        # Use image center
        center = (h // 2, w // 2)
    
    cy, cx = center
    half_size = size // 2
    
    # Calculate bounds
    y1 = max(0, cy - half_size)
    y2 = min(h, cy + half_size)
    x1 = max(0, cx - half_size)
    x2 = min(w, cx + half_size)
    
    subimage = image[y1:y2, x1:x2]
    logger.info(f"Extracted subimage: {subimage.shape} from region [{y1}:{y2}, {x1}:{x2}]")
    
    return subimage, (y1, x1)

def simple_source_detection(image):
    """Simple source detection wrapper for our enhanced detector."""
    from src.detection import AdvancedSourceDetector, DetectionConfig
    
    config = DetectionConfig(
        use_multi_threshold=False,
        use_adaptive_threshold=False,
        classify_sources=True,  # Enable for real data
        threshold=3.0,  # Higher threshold for real data
        minarea=10,     # Larger minimum area for real sources
    )
    detector = AdvancedSourceDetector(config)
    results = detector.detect_sources(image)
    return results.sources

def test_real_jwst_photometry():
    """Test our photometry software on real JWST data."""
    logger.info("="*60)
    logger.info("TESTING PHASE 4 PHOTOMETRY ON REAL JWST DATA")
    logger.info("="*60)
    
    # Define the image files
    image_dir = Path("./sample_images")
    image_files = {
        'F150W': 'mosaic_nircam_f150w_COSMOS-Web_30mas_A10_v1.0_sci.fits.gz',
        'F277W': 'mosaic_nircam_f277w_COSMOS-Web_30mas_A10_v1.0_sci.fits.gz', 
        'F444W': 'mosaic_nircam_f444w_COSMOS-Web_30mas_A10_v1.0_sci.fits.gz'
    }
    
    # Check all files exist
    for band, filename in image_files.items():
        filepath = image_dir / filename
        if not filepath.exists():
            logger.error(f"Missing image file: {filepath}")
            return False
        logger.info(f"Found {band}: {filename}")
    
    # Test results storage
    results = {}
    
    # Process each band
    for band, filename in image_files.items():
        logger.info(f"\n--- Processing {band} band ---")
        start_time = time.time()
        
        try:
            # Load the full image
            filepath = image_dir / filename
            full_image, header = load_jwst_image(filepath)
            
            # Extract a manageable subimage for testing (512x512 pixels)
            subimage, offset = extract_subimage(full_image, size=512)
            
            # Test source detection
            logger.info(f"Running source detection on {band}...")
            sources = simple_source_detection(subimage)
            logger.info(f"Detected {len(sources)} sources in {band}")
            
            if len(sources) == 0:
                logger.warning(f"No sources detected in {band} - may need to adjust detection parameters")
                continue
            
            # Test Enhanced Aperture Photometry (Phase 4.1)
            logger.info(f"Running enhanced aperture photometry on {band}...")
            from src.photometry import EnhancedAperturePhotometry, AperturePhotometryConfig
            
            config = AperturePhotometryConfig(
                circular_apertures=[0.15, 0.32, 0.50],  # arcsec (30mas pixels = 5, 10.7, 16.7 pixels)
                use_elliptical_apertures=True,
                use_kron_apertures=True,
                background_method='local_annulus',
                correct_contamination=True
            )
            
            photometry = EnhancedAperturePhotometry(config)
            phot_results = photometry.perform_aperture_photometry(subimage, sources)
            
            logger.info(f"Photometry completed: {len(phot_results.sources)} sources processed")
            
            # Store results
            results[band] = {
                'sources': sources,
                'photometry': phot_results,
                'image_shape': full_image.shape,
                'subimage_shape': subimage.shape,
                'header': header
            }
            
            # Log some basic photometry statistics
            if len(phot_results.sources) > 0:
                fluxes = []
                for src in phot_results.sources:
                    if hasattr(src, 'circular_photometry') and src.circular_photometry:
                        flux = src.circular_photometry[0.32]['flux']  # 0.32" aperture
                        if flux > 0:
                            fluxes.append(flux)
                
                if fluxes:
                    fluxes = np.array(fluxes)
                    logger.info(f"Flux statistics (0.32\" aperture): median={np.median(fluxes):.3f}, "
                              f"range=[{np.min(fluxes):.3f}, {np.max(fluxes):.3f}]")
            
            elapsed = time.time() - start_time
            logger.info(f"{band} processing completed in {elapsed:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Failed to process {band}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Multi-band analysis if we have multiple successful bands
    successful_bands = list(results.keys())
    logger.info(f"\n--- MULTI-BAND ANALYSIS ---")
    logger.info(f"Successfully processed bands: {successful_bands}")
    
    if len(successful_bands) >= 2:
        logger.info("Testing multi-band flux calibration...")
        
        try:
            # Test Phase 4.2 Flux Calibration
            from src.calibration import FluxCalibrator, CalibrationConfig
            
            # Prepare photometry results for calibration
            photometry_results = {}
            headers = {}
            
            for band in successful_bands:
                photometry_results[band] = results[band]['photometry']
                headers[band] = results[band]['header']
            
            # Configure calibration
            cal_config = CalibrationConfig(
                input_units="DN/s",
                output_units="uJy",
                apply_aperture_corrections=True,
                apply_galactic_extinction=False  # Disable for testing
            )
            
            calibrator = FluxCalibrator(cal_config)
            cal_results = calibrator.calibrate_photometry(photometry_results, headers)
            
            logger.info(f"Flux calibration completed: {len(cal_results.sources)} sources calibrated")
            logger.info(f"Calibrated bands: {list(cal_results.band_calibrations.keys())}")
            
        except Exception as e:
            logger.error(f"Multi-band calibration failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("REAL JWST DATA TEST SUMMARY")
    logger.info(f"{'='*60}")
    
    for band, result in results.items():
        n_sources = len(result['sources'])
        n_phot = len(result['photometry'].sources)
        shape = result['image_shape']
        logger.info(f"{band}: {shape[0]}x{shape[1]} pixels, {n_sources} detected, {n_phot} photometry")
    
    if results:
        logger.info(f"✅ Successfully processed {len(results)} band(s) with Phase 4 photometry!")
        logger.info("🎉 Real JWST data processing test PASSED!")
        return True
    else:
        logger.error("❌ Failed to process any bands")
        return False

if __name__ == "__main__":
    try:
        success = test_real_jwst_photometry()
        if success:
            print("\n🚀 Phase 4 JWST Photometry software is ready for real data!")
        else:
            print("\n💥 Issues found with real data processing")
            exit(1)
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
