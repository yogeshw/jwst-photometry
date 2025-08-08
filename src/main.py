"""
Enhanced JWST NIRCam Photometry Pipeline

This is the main entry point for the enhanced JWST photometry pipeline,
featuring robust configuration management, comprehensive error handling,
and advanced photometry capabilities.
"""

import sys
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import traceback
import time

import numpy as np
from astropy.table import Table
from astropy.io import fits

# Import enhanced pipeline modules
from config_manager import ConfigManager, load_config
from parallel_processing import ParallelProcessor
from cache import CacheManager, CacheConfig, stable_hash
from jwst_data_handler import JWSTDataHandler
from utils import (
    setup_logging, timing_context, memory_monitor,
    DataValidationError, ConfigurationError, PhotometryError,
    check_dependencies, get_memory_usage
)

# Import existing modules (to be enhanced)
from detection import detect_sources, generate_segmentation_map, identify_star_candidates
from psf import generate_empirical_psf, match_psf, apply_kernel

# Import enhanced Phase 4 modules
from photometry import EnhancedAperturePhotometry, AperturePhotometryConfig, extract_photometry_table
from calibration import FluxCalibrator, CalibrationConfig
from uncertainties import ComprehensiveErrorEstimator, ErrorEstimationConfig


class JWSTPhotometryPipeline:
    """
    Enhanced JWST NIRCam photometry pipeline with robust error handling,
    comprehensive logging, and modular design.
    """
    
    def __init__(self, config_path: Optional[str] = None, log_level: str = 'INFO'):
        """
        Initialize the photometry pipeline.
        
        Parameters:
        -----------
        config_path : str, optional
            Path to configuration file. If None, uses default config.
        log_level : str, default='INFO'
            Logging level for the pipeline
        """
        # Set up logging
        self.logger = setup_logging(
            level=log_level,
            log_file=f'jwst_photometry_{self._get_timestamp()}.log',
            enable_colors=True
        )
        
        self.logger.info("="*80)
        self.logger.info("JWST NIRCam Photometry Pipeline - Enhanced Version")
        self.logger.info("="*80)
        
        # Check dependencies
        self._check_dependencies()
        
        # Initialize configuration manager
        try:
            if config_path:
                self.config_manager = load_config(config_path)
            else:
                self.config_manager = ConfigManager()
                self.logger.info("Using default configuration")
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            raise ConfigurationError(f"Configuration loading failed: {e}")
        
    # Initialize data handler
        self.data_handler = JWSTDataHandler()
        
        # Initialize storage for pipeline data
        self.images = {}
        self.metadata = {}
        self.weights = {}
        self.dq_arrays = {}
        self.error_arrays = {}
        self.background_maps = {}
        
        # Results storage
        self.sources = None
        self.segmentation_map = None
        self.star_candidates = None
        self.photometry_results = {}
        self.final_catalog = None

        # Caching (Phase 7: advanced caching strategies)
        try:
            output_dir = Path(self.config_manager.get_output_config().output_directory)
        except Exception:
            output_dir = Path('.')
        cache_base = output_dir / '.cache'
        self.cache_detection = CacheManager(CacheConfig(cache_base, namespace='detection', enabled=True))
        self.cache_photometry = CacheManager(CacheConfig(cache_base, namespace='photometry', enabled=True))
        
        self.logger.info(f"Pipeline initialized - Memory usage: {get_memory_usage():.1f} MB")
    
    def _get_timestamp(self) -> str:
        """Get current timestamp string."""
        import time
        return time.strftime("%Y%m%d_%H%M%S")
    
    def _check_dependencies(self) -> None:
        """Check if all required dependencies are available."""
        self.logger.info("Checking dependencies...")
        dependencies = check_dependencies()
        
        missing_required = [pkg for pkg, available in dependencies.items() 
                          if not available and pkg in ['numpy', 'astropy', 'sep', 'pypher']]
        
        if missing_required:
            raise ImportError(f"Missing required packages: {missing_required}")
        
        self.logger.info("All required dependencies available")
    
    def load_images(self) -> None:
        """
        Load all JWST images specified in the configuration.
        
        Raises:
        -------
        DataValidationError
            If image loading or validation fails
        """
        self.logger.info("Loading JWST images...")
        
        with timing_context("Image loading", self.logger):
            with memory_monitor("Image loading", self.logger):
                # Get filter configurations
                filter_configs = self.config_manager.config['filters']
                
                if not filter_configs:
                    raise DataValidationError("No filters configured")
                
                # Prepare file paths for batch loading
                image_paths = {}
                weight_paths = {}
                
                for filter_name, filter_config in filter_configs.items():
                    image_paths[filter_name] = filter_config.image_path
                    if filter_config.weight_path:
                        weight_paths[filter_name] = filter_config.weight_path
                
                # Load science images
                self.logger.info(f"Loading {len(image_paths)} science images")
                try:
                    images_data = self.data_handler.load_multiple_images(
                        image_paths, 
                        extension='SCI',
                        load_dq=True,
                        load_err=True
                    )
                    
                    # Organize loaded data
                    for filter_name, (data, metadata, dq_array, err_array) in images_data.items():
                        self.images[filter_name] = data
                        self.metadata[filter_name] = metadata
                        self.dq_arrays[filter_name] = dq_array
                        self.error_arrays[filter_name] = err_array
                        
                        # Update configuration with extracted metadata
                        self.config_manager.update_filter_metadata(
                            filter_name, {
                                'EXPTIME': metadata.exposure_time,
                                'GAIN': metadata.gain,
                                'PHOTZP': metadata.zero_point,
                                'DETECTOR': metadata.detector
                            }
                        )
                    
                    # Load weight images if specified
                    if weight_paths:
                        self.logger.info(f"Loading {len(weight_paths)} weight images")
                        weight_data = self.data_handler.load_multiple_images(
                            weight_paths,
                            extension=0,  # Weight images typically in primary extension
                            load_dq=False,
                            load_err=False
                        )
                        
                        for filter_name, (weight_data, _, _, _) in weight_data.items():
                            self.weights[filter_name] = weight_data
                    
                    # Validate image consistency
                    if not self.data_handler.validate_image_consistency(images_data):
                        self.logger.warning("Image consistency validation failed")
                    
                    self.logger.info(f"Successfully loaded {len(self.images)} images")
                    
                except Exception as e:
                    self.logger.error(f"Image loading failed: {e}")
                    raise DataValidationError(f"Failed to load images: {e}")
    
    def process_images(self) -> None:
        """
        Process loaded images (background subtraction, unit conversion, etc.).
        """
        self.logger.info("Processing images...")
        
        with timing_context("Image processing", self.logger):
            for filter_name, image_data in self.images.items():
                self.logger.debug(f"Processing {filter_name} image")
                
                try:
                    # Convert to microjanskys if needed
                    metadata = self.metadata[filter_name]
                    if metadata.flux_conversion_factor:
                        converted_image = self.data_handler.convert_to_microjanskys(
                            image_data, metadata
                        )
                        self.images[filter_name] = converted_image
                        self.logger.debug(f"Converted {filter_name} to µJy")
                    
                    # Background subtraction will be handled by the background module
                    # For now, keep the original process_image function
                    from utils import process_image
                    processed_image, background_map = process_image(
                        self.images[filter_name],
                        subtract_background=True
                    )
                    
                    self.images[filter_name] = processed_image
                    if background_map is not None:
                        self.background_maps[filter_name] = background_map
                    
                except Exception as e:
                    self.logger.error(f"Failed to process {filter_name}: {e}")
                    raise PhotometryError(f"Image processing failed for {filter_name}: {e}")
        
        self.logger.info("Image processing completed")
    
    def create_detection_image(self) -> np.ndarray:
        """
        Create a combined detection image from multiple bands.
        
        Returns:
        --------
        numpy.ndarray
            Combined detection image
        """
        self.logger.info("Creating detection image...")
        
        # Get detection configuration
        detection_config = self.config_manager.get_detection_config()
        
        # Use specified bands or default combination
        if hasattr(detection_config, 'detection_bands') and detection_config.detection_bands:
            detection_bands = detection_config.detection_bands
        else:
            # Default: use red bands for better S/N
            available_bands = list(self.images.keys())
            red_bands = ['F277W', 'F356W', 'F444W']
            detection_bands = [b for b in red_bands if b in available_bands]
            
            if not detection_bands:
                detection_bands = available_bands[:3]  # Use first 3 available
        
        self.logger.info(f"Using bands for detection: {detection_bands}")
        
        # Cache key uses input image identities and bands selected
        try:
            # Prefer file paths to avoid hashing large arrays
            filter_configs = self.config_manager.config['filters']
            band_paths = {b: (filter_configs[b].image_path, filter_configs[b].weight_path)
                          for b in detection_bands if b in filter_configs}
            cache_key = f"detimg|{stable_hash(band_paths)}"
            cached = self.cache_detection.get(cache_key)
            if cached is not None:
                self.logger.info("Loaded detection image from cache")
                return cached
        except Exception:
            cache_key = None

        # Create weighted combination
        combined_image = None
        total_weight = 0
        
        for band in detection_bands:
            if band not in self.images:
                self.logger.warning(f"Detection band {band} not available")
                continue
            
            image = self.images[band]
            weight = self.weights.get(band, np.ones_like(image))
            
            if combined_image is None:
                combined_image = image * weight
                total_weight = weight
            else:
                combined_image += image * weight
                total_weight += weight
        
        if combined_image is None:
            raise DataValidationError("No images available for detection")
        
        # Normalize by total weight
        detection_image = combined_image / np.maximum(total_weight, 1e-10)
        
        self.logger.info(f"Created detection image with shape {detection_image.shape}")

        # Store in cache
        if cache_key:
            try:
                self.cache_detection.set(cache_key, detection_image)
            except Exception:
                pass
        return detection_image
    
    def run_source_detection(self) -> None:
        """
        Perform source detection on the combined detection image.
        """
        self.logger.info("Running source detection...")
        
        with timing_context("Source detection", self.logger):
            # Create detection image
            detection_image = self.create_detection_image()
            
            # Get detection parameters
            detection_config = self.config_manager.get_detection_config()
            detection_params = {
                'kernel': detection_config.kernel_fwhm,
                'minarea': detection_config.minarea,
                'thresh': detection_config.thresh,
                'deblend_nthresh': detection_config.deblend_nthresh,
                'deblend_cont': detection_config.deblend_cont,
                'clean': detection_config.clean
            }
            
            try:
                # Detect sources
                self.sources, background = detect_sources(detection_image, detection_params)
                
                self.logger.info(f"Detected {len(self.sources)} sources")
                
                # Generate segmentation map
                self.segmentation_map = generate_segmentation_map(detection_image, self.sources)
                
                # Identify star candidates
                self.star_candidates = identify_star_candidates(self.sources)
                self.logger.info(f"Identified {len(self.star_candidates)} star candidates")
                
            except Exception as e:
                self.logger.error(f"Source detection failed: {e}")
                raise PhotometryError(f"Source detection failed: {e}")
    
    def run_psf_homogenization(self) -> None:
        """
        Perform PSF homogenization across all bands.
        """
        self.logger.info("Running PSF homogenization...")
        
        with timing_context("PSF homogenization", self.logger):
            try:
                # Get PSF configuration
                psf_config = self.config_manager.get_psf_config()
                target_band = psf_config.target_band
                
                if target_band not in self.images:
                    available_bands = list(self.images.keys())
                    target_band = available_bands[0]  # Use first available as fallback
                    self.logger.warning(f"Target PSF band not available, using {target_band}")
                
                # Generate empirical PSF for target band
                self.logger.info(f"Generating target PSF from {target_band}")
                target_psf = generate_empirical_psf(
                    self.images[target_band], 
                    self.star_candidates
                )
                
                # Match PSFs for other bands
                for band in self.images:
                    if band != target_band:
                        self.logger.debug(f"Matching PSF for {band} to {target_band}")
                        
                        # Generate PSF for current band
                        band_psf = generate_empirical_psf(
                            self.images[band], 
                            self.star_candidates
                        )
                        
                        # Create matching kernel
                        kernel = match_psf(
                            band_psf, 
                            target_psf, 
                            psf_config.regularization_parameter
                        )
                        
                        # Apply kernel to image
                        self.images[band] = apply_kernel(self.images[band], kernel)
                
                self.logger.info("PSF homogenization completed")
                
            except Exception as e:
                self.logger.error(f"PSF homogenization failed: {e}")
                raise PhotometryError(f"PSF homogenization failed: {e}")
    
    def run_aperture_photometry(self) -> None:
        """
        Perform enhanced aperture photometry on all bands.
        """
        self.logger.info("Running enhanced aperture photometry...")
        
        with timing_context("Enhanced aperture photometry", self.logger):
            try:
                # Get photometry configuration
                photometry_config = self.config_manager.get_photometry_config()
                
                # Create enhanced aperture photometry configuration
                aperture_config = AperturePhotometryConfig(
                    circular_apertures=photometry_config.aperture_radii,
                    use_elliptical_apertures=True,
                    use_kron_apertures=True,
                    use_adaptive_apertures=True,
                    background_method="local_annulus",
                    correct_contamination=True,
                    apply_aperture_corrections=True,
                    estimate_uncertainties=True,
                    flag_contaminated=True,
                    create_diagnostic_plots=True
                )
                
                # Initialize enhanced aperture photometry processor
                aperture_processor = EnhancedAperturePhotometry(aperture_config)

                # Prepare common inputs
                sources_array = self._convert_sources_to_array(self.sources)
                output_config = self.config_manager.get_output_config()
                output_dir = Path(output_config.output_directory)

                # Parallel per-band processing (Phase 7 integration)
                processor = ParallelProcessor(max_workers=None, use_processes=False)

                def _process_band(band_name: str):
                    # Cache key built from image file path and relevant config
                    try:
                        fcfg = self.config_manager.config['filters'].get(band_name)
                        key_meta = {
                            'band': band_name,
                            'image_path': getattr(fcfg, 'image_path', None),
                            'weights': getattr(fcfg, 'weight_path', None),
                            'apertures': aperture_config.circular_apertures,
                            'kron': aperture_config.use_kron_apertures,
                        }
                        cache_key = f"phot|{stable_hash(key_meta)}"
                        cached = self.cache_photometry.get(cache_key)
                        if cached is not None:
                            return band_name, cached
                    except Exception:
                        cache_key = None

                    image = self.images[band_name]
                    background_map = self.background_maps.get(band_name)
                    rms_map = self.error_arrays.get(band_name)
                    wcs = self.metadata[band_name].wcs if hasattr(self.metadata[band_name], 'wcs') else None

                    photometry_results = aperture_processor.perform_aperture_photometry(
                        image=image,
                        sources=sources_array,
                        background_map=background_map,
                        rms_map=rms_map,
                        segmentation_map=self.segmentation_map,
                        psf_model=None,
                        wcs=wcs
                    )

                    # Diagnostics
                    if aperture_config.create_diagnostic_plots:
                        plot_path = output_dir / f'aperture_photometry_diagnostics_{band_name}.png'
                        aperture_processor.plot_photometry_diagnostics(photometry_results, str(plot_path))

                    # Cache
                    if cache_key:
                        try:
                            self.cache_photometry.set(cache_key, photometry_results)
                        except Exception:
                            pass

                    return band_name, photometry_results

                bands = list(self.images.keys())
                # Execute in parallel threads
                results = processor.process_chunks_parallel(
                    chunks=[{'band': b} for b in bands],
                    processing_function=lambda chunk: _process_band(chunk['band'])
                )

                # Collect results preserving band mapping
                for item in results:
                    if item is None:
                        continue
                    band_name, phot_res = item
                    self.photometry_results[band_name] = phot_res
                
                self.logger.info(f"Enhanced aperture photometry completed for {len(self.photometry_results)} bands")
                
            except Exception as e:
                self.logger.error(f"Enhanced aperture photometry failed: {e}")
                raise PhotometryError(f"Enhanced aperture photometry failed: {e}")
    
    def _convert_sources_to_array(self, sources):
        """Convert sources to structured array format for enhanced photometry."""
        n_sources = len(sources)
        
        # Create structured array
        dtype = [('x', 'f4'), ('y', 'f4'), ('a', 'f4'), ('b', 'f4'), ('theta', 'f4')]
        sources_array = np.zeros(n_sources, dtype=dtype)
        
        for i, source in enumerate(sources):
            sources_array[i]['x'] = source.get('x', 0)
            sources_array[i]['y'] = source.get('y', 0)
            sources_array[i]['a'] = source.get('a', 2.0)
            sources_array[i]['b'] = source.get('b', 2.0)
            sources_array[i]['theta'] = source.get('theta', 0.0)
        
        return sources_array
    
    def apply_photometric_corrections(self) -> None:
        """
        Apply comprehensive flux calibration and uncertainty estimation.
        """
        self.logger.info("Applying comprehensive flux calibration...")
        
        with timing_context("Flux calibration and uncertainty estimation", self.logger):
            try:
                # Phase 4.2: Flux Calibration
                self.logger.info("Performing flux calibration...")
                
                # Create calibration configuration
                calibration_config = CalibrationConfig(
                    input_units="DN/s",
                    output_units="uJy",
                    use_in_flight_zeropoints=True,
                    apply_aperture_corrections=True,
                    apply_galactic_extinction=True,
                    check_color_consistency=True,
                    save_calibration_diagnostics=True,
                    create_calibration_plots=True
                )
                
                # Initialize flux calibrator
                flux_calibrator = FluxCalibrator(calibration_config)
                
                # Prepare headers dictionary
                headers = {}
                for band in self.images.keys():
                    # Convert metadata to FITS header format
                    header = fits.Header()
                    metadata = self.metadata[band]
                    
                    if hasattr(metadata, 'zero_point') and metadata.zero_point:
                        header['PHOTZPT'] = metadata.zero_point
                    if hasattr(metadata, 'gain') and metadata.gain:
                        header['GAIN'] = metadata.gain
                    if hasattr(metadata, 'exposure_time') and metadata.exposure_time:
                        header['EXPTIME'] = metadata.exposure_time
                    
                    headers[band] = header
                
                # Perform flux calibration
                calibration_results = flux_calibrator.calibrate_photometry(
                    photometry_results=self.photometry_results,
                    headers=headers
                )
                
                # Phase 4.3: Comprehensive Error Estimation
                self.logger.info("Performing comprehensive uncertainty estimation...")
                
                # Create error estimation configuration
                error_config = ErrorEstimationConfig(
                    include_poisson_noise=True,
                    include_readnoise=True,
                    include_dark_current=True,
                    include_crowding_uncertainty=True,
                    model_correlated_noise=True,
                    estimate_cross_band_correlation=True,
                    use_monte_carlo_propagation=True,
                    flag_high_uncertainty_sources=True,
                    save_uncertainty_maps=True,
                    generate_uncertainty_diagnostics=True
                )
                
                # Initialize error estimator
                error_estimator = ComprehensiveErrorEstimator(error_config)
                
                # Perform uncertainty estimation
                uncertainty_results = error_estimator.estimate_uncertainties(
                    images=self.images,
                    photometry_results=self.photometry_results,
                    calibration_results=calibration_results,
                    psf_models=None,  # Will be enhanced in future phases
                    background_maps=self.background_maps,
                    rms_maps=self.error_arrays
                )
                
                # Store results
                self.calibration_results = calibration_results
                self.uncertainty_results = uncertainty_results
                
                # Create diagnostic plots
                output_config = self.config_manager.get_output_config()
                output_dir = Path(output_config.output_directory)
                
                # Calibration diagnostics
                cal_plot_path = output_dir / 'flux_calibration_diagnostics.png'
                flux_calibrator.plot_calibration_diagnostics(
                    calibration_results, str(cal_plot_path)
                )
                
                # Uncertainty diagnostics
                unc_plot_path = output_dir / 'uncertainty_diagnostics.png'
                error_estimator.plot_uncertainty_diagnostics(
                    uncertainty_results, str(unc_plot_path)
                )
                
                self.logger.info("Comprehensive calibration and uncertainty estimation completed")
                
            except Exception as e:
                self.logger.error(f"Calibration and uncertainty estimation failed: {e}")
                raise PhotometryError(f"Calibration and uncertainty estimation failed: {e}")
    
    def create_final_catalog(self) -> None:
        """
        Create the final comprehensive photometry catalog.
        """
        self.logger.info("Creating final comprehensive photometry catalog...")
        
        with timing_context("Catalog creation", self.logger):
            try:
                # Use enhanced catalog creation from calibration results
                if hasattr(self, 'calibration_results') and self.calibration_results:
                    # Export calibrated catalog
                    output_config = self.config_manager.get_output_config()
                    output_dir = Path(output_config.output_directory)
                    
                    # Create comprehensive catalog using flux calibrator
                    flux_calibrator = FluxCalibrator()
                    catalog_path = output_dir / 'comprehensive_catalog.fits'
                    
                    flux_calibrator.export_calibrated_catalog(
                        self.calibration_results,
                        str(catalog_path),
                        format='fits'
                    )
                    
                    # Load the exported catalog as final catalog
                    self.final_catalog = Table.read(catalog_path)
                    
                    # Add additional metadata
                    self.final_catalog.meta['ORIGIN'] = 'JWST-Photometry Pipeline Enhanced v2.0'
                    self.final_catalog.meta['PIPELINE_PHASE'] = 'Phase 4 - Advanced Photometry Complete'
                    self.final_catalog.meta['NSOURCES'] = len(self.final_catalog)
                    self.final_catalog.meta['NFILTERS'] = len(self.calibration_results.band_calibrations)
                    self.final_catalog.meta['CALIBRATION_QUALITY'] = self.calibration_results.overall_quality
                    self.final_catalog.meta['UNCERTAINTY_QUALITY'] = self.uncertainty_results.overall_uncertainty_quality
                    
                    # Add processing statistics
                    total_measurements = sum(
                        len(source.calibrated_fluxes) for source in self.calibration_results.sources
                    )
                    self.final_catalog.meta['TOTAL_MEASUREMENTS'] = total_measurements
                    
                else:
                    # Fallback to basic catalog creation
                    self.logger.warning("Using fallback catalog creation - enhanced results not available")
                    self._create_basic_catalog()
                
                self.logger.info(f"Created comprehensive catalog with {len(self.final_catalog)} sources")
                self.logger.info(f"Catalog contains {len(self.final_catalog.colnames)} columns")
                
            except Exception as e:
                self.logger.error(f"Comprehensive catalog creation failed: {e}")
                # Fallback to basic catalog
                self.logger.warning("Falling back to basic catalog creation")
                self._create_basic_catalog()
    
    def _create_basic_catalog(self) -> None:
        """Create basic catalog as fallback."""
        n_sources = len(self.sources)
        catalog_data = {}
        
        # Basic source properties
        catalog_data['id'] = np.arange(1, n_sources + 1)
        catalog_data['x'] = self.sources['x']
        catalog_data['y'] = self.sources['y']
        
        # Add RA/Dec if WCS is available
        first_band = list(self.metadata.keys())[0]
        if hasattr(self.metadata[first_band], 'wcs') and self.metadata[first_band].wcs is not None:
            try:
                wcs = self.metadata[first_band].wcs
                ra, dec = wcs.pixel_to_world_values(self.sources['x'], self.sources['y'])
                catalog_data['ra'] = ra
                catalog_data['dec'] = dec
            except Exception:
                self.logger.warning("Failed to compute RA/Dec coordinates")
        
        # Add basic photometry for each band
        for band, photometry in self.photometry_results.items():
            if hasattr(photometry, 'sources'):
                # Extract photometry from enhanced results
                phot_table = extract_photometry_table(photometry)
                
                # Add largest aperture measurements
                largest_aperture = max(photometry.config.circular_apertures)
                flux_col = f'flux_r{largest_aperture}'
                fluxerr_col = f'flux_err_r{largest_aperture}'
                
                if flux_col in phot_table.colnames:
                    catalog_data[f'flux_{band}'] = phot_table[flux_col]
                    catalog_data[f'flux_err_{band}'] = phot_table[fluxerr_col]
                else:
                    catalog_data[f'flux_{band}'] = np.zeros(n_sources)
                    catalog_data[f'flux_err_{band}'] = np.zeros(n_sources)
            else:
                # Legacy format
                catalog_data[f'flux_{band}'] = photometry.get('flux', np.zeros(n_sources))
                catalog_data[f'flux_err_{band}'] = photometry.get('flux_err', np.zeros(n_sources))
        
        # Create table
        self.final_catalog = Table(catalog_data)
        
        # Add basic metadata
        self.final_catalog.meta['ORIGIN'] = 'JWST-Photometry Pipeline Enhanced (Basic Mode)'
        self.final_catalog.meta['NSOURCES'] = n_sources
        self.final_catalog.meta['NFILTERS'] = len(self.photometry_results)
    
    def save_outputs(self) -> None:
        """
        Save all pipeline outputs.
        """
        self.logger.info("Saving pipeline outputs...")
        
        with timing_context("Output saving", self.logger):
            try:
                # Get output configuration
                output_config = self.config_manager.get_output_config()
                
                # Ensure output directory exists
                from utils import validate_output_directory, save_catalog
                output_dir = validate_output_directory(output_config.output_directory)
                
                # Save final catalog
                if self.final_catalog is not None:
                    catalog_path = save_catalog(
                        self.final_catalog,
                        output_config.catalog_format,
                        output_dir,
                        metadata={
                            'pipeline_version': 'enhanced',
                            'processing_date': self._get_timestamp()
                        }
                    )
                    self.logger.info(f"Saved catalog: {catalog_path}")
                
                # Save intermediate products if requested
                if output_config.save_intermediate_products:
                    self._save_intermediate_products(output_dir)
                
                self.logger.info("All outputs saved successfully")
                
            except Exception as e:
                self.logger.error(f"Output saving failed: {e}")
                raise PhotometryError(f"Output saving failed: {e}")
    
    def _save_intermediate_products(self, output_dir: Path) -> None:
        """Save intermediate data products."""
        from astropy.io import fits
        
        # Save segmentation map
        if self.segmentation_map is not None:
            seg_path = output_dir / 'segmentation_map.fits'
            fits.writeto(seg_path, self.segmentation_map, overwrite=True)
            self.logger.debug(f"Saved segmentation map: {seg_path}")
        
        # Save background maps
        for band, bg_map in self.background_maps.items():
            bg_path = output_dir / f'background_{band}.fits'
            fits.writeto(bg_path, bg_map, overwrite=True)
            self.logger.debug(f"Saved background map: {bg_path}")
    
    def run_full_pipeline(self) -> Table:
        """
        Run the complete photometry pipeline.
        
        Returns:
        --------
        Table
            Final photometry catalog
            
        Raises:
        -------
        PhotometryError
            If any pipeline step fails
        """
        self.logger.info("Starting full photometry pipeline...")
        
        pipeline_start_time = time.time()
        initial_memory = get_memory_usage()
        
        try:
            # Execute pipeline steps
            self.load_images()
            self.process_images()
            self.run_source_detection()
            self.run_psf_homogenization()
            self.run_aperture_photometry()
            self.apply_photometric_corrections()
            self.create_final_catalog()
            self.save_outputs()
            
            # Pipeline completion
            pipeline_duration = time.time() - pipeline_start_time
            final_memory = get_memory_usage()
            memory_delta = final_memory - initial_memory
            
            self.logger.info("="*80)
            self.logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.info(f"Total runtime: {pipeline_duration:.1f} seconds")
            self.logger.info(f"Memory usage: {final_memory:.1f} MB (Δ{memory_delta:+.1f} MB)")
            self.logger.info(f"Sources detected: {len(self.final_catalog) if self.final_catalog else 0}")
            self.logger.info(f"Bands processed: {len(self.photometry_results)}")
            self.logger.info("="*80)
            
            return self.final_catalog
            
        except Exception as e:
            self.logger.error("="*80)
            self.logger.error("PIPELINE FAILED")
            self.logger.error(f"Error: {e}")
            self.logger.error("="*80)
            self.logger.debug("Full traceback:", exc_info=True)
            raise


def main():
    """
    Main function to run the enhanced JWST photometry pipeline.
    """
    import argparse
    import time
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Enhanced JWST NIRCam Photometry Pipeline')
    parser.add_argument('--config', '-c', type=str, default='config.yaml',
                       help='Path to configuration file (default: config.yaml)')
    parser.add_argument('--log-level', '-l', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                       help='Logging level (default: INFO)')
    parser.add_argument('--output-dir', '-o', type=str, default=None,
                       help='Override output directory from config')
    
    args = parser.parse_args()
    
    try:
        # Initialize and run pipeline
        pipeline = JWSTPhotometryPipeline(
            config_path=args.config,
            log_level=args.log_level
        )
        
        # Override output directory if specified
        if args.output_dir:
            pipeline.config_manager.config['output'].output_directory = args.output_dir
        
        # Run the pipeline
        catalog = pipeline.run_full_pipeline()
        
        print(f"\nPipeline completed successfully!")
        print(f"Final catalog contains {len(catalog)} sources")
        
        return 0
        
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
        return 1
    except Exception as e:
        print(f"\nPipeline failed: {e}")
        return 1


if __name__ == '__main__':
    import time
    sys.exit(main())
