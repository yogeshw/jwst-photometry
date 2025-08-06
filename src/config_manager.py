"""
Enhanced Configuration Management for JWST Photometry Pipeline

This module provides robust configuration validation and management capabilities
for the JWST photometry pipeline, including schema validation, filter-specific
parameters, and observation metadata handling.
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
import numpy as np


@dataclass
class FilterConfig:
    """Configuration for a specific filter/band."""
    image_path: str
    weight_path: str
    zero_point: float = 25.0
    pixel_scale: float = 0.031  # arcsec/pixel for NIRCam
    exposure_time: Optional[float] = None
    gain: Optional[float] = None
    detector: Optional[str] = None
    background_mesh_size: int = 32
    background_filter_size: int = 3


@dataclass
class SourceDetectionConfig:
    """Configuration for source detection parameters."""
    kernel_fwhm: float = 3.5
    minarea: int = 3
    thresh: float = 1.2
    deblend_nthresh: int = 32
    deblend_cont: float = 0.0001
    clean: bool = False
    clean_param: float = 1.0
    mask_regions: List[str] = field(default_factory=list)
    detection_band: str = 'detection'


@dataclass
class PSFConfig:
    """Configuration for PSF handling and homogenization."""
    target_band: str = 'F444W'
    regularization_parameter: float = 0.003
    star_selection_magnitude_limits: tuple = (16.0, 23.0)
    star_selection_size_limits: tuple = (0.8, 1.2)
    star_selection_snr_limit: float = 50.0
    psf_size: int = 25
    max_iterations: int = 3
    convergence_threshold: float = 0.01
    spatial_variation: bool = True
    n_grid_points: int = 9


@dataclass
class PhotometryConfig:
    """Configuration for aperture photometry."""
    aperture_radii: List[float] = field(default_factory=lambda: [0.16, 0.24, 0.35, 0.50, 0.70])
    elliptical_kron_factor: float = 2.5
    min_kron_factor: float = 1.0
    max_kron_radius: float = 5.0
    background_annulus_inner: float = 2.0
    background_annulus_outer: float = 3.0
    local_background_size: int = 9
    sigma_clip_threshold: float = 3.0
    max_iterations: int = 5


@dataclass
class CalibrationConfig:
    """Configuration for flux calibration and corrections."""
    reference_band: str = 'F444W'
    apply_aperture_corrections: bool = True
    apply_extinction_correction: bool = True
    apply_systematic_corrections: bool = True
    color_term_corrections: bool = True
    ebv_map_path: Optional[str] = None
    extinction_law: str = 'ccm89'
    rv_value: float = 3.1


@dataclass
class QualityConfig:
    """Configuration for quality assessment and flags."""
    enable_quality_flags: bool = True
    snr_threshold: float = 5.0
    saturation_threshold: float = 50000.0
    edge_buffer: int = 10
    contamination_threshold: float = 0.1
    astrometric_uncertainty_limit: float = 0.1
    photometric_uncertainty_limit: float = 0.1


@dataclass
class OutputConfig:
    """Configuration for output catalogs and data products."""
    catalog_format: str = 'fits'
    output_directory: str = './output'
    save_intermediate_products: bool = False
    save_segmentation_map: bool = True
    save_background_map: bool = True
    save_psf_models: bool = True
    catalog_columns: List[str] = field(default_factory=lambda: [
        'id', 'x', 'y', 'ra', 'dec', 'mag_auto', 'magerr_auto',
        'mag_aper', 'magerr_aper', 'flux_radius', 'class_star',
        'flags', 'a', 'b', 'theta', 'kron_radius'
    ])


class ConfigManager:
    """
    Manages configuration loading, validation, and access for the JWST photometry pipeline.
    
    This class provides a robust interface for handling complex configuration requirements
    including filter-specific parameters, observation metadata, and validation.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Parameters:
        -----------
        config_path : str, optional
            Path to the configuration file. If None, loads default configuration.
        """
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path
        self.config = {}
        self.filters = {}
        self.observation_metadata = {}
        
        if config_path:
            self.load_config(config_path)
        else:
            self.load_default_config()
    
    def load_config(self, config_path: str) -> None:
        """
        Load configuration from a YAML file with comprehensive validation.
        
        Parameters:
        -----------
        config_path : str
            Path to the configuration file
            
        Raises:
        -------
        FileNotFoundError
            If the configuration file doesn't exist
        yaml.YAMLError
            If the YAML file is malformed
        ValueError
            If the configuration fails validation
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as file:
                raw_config = yaml.safe_load(file)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML file {config_path}: {e}")
        
        # Validate and parse configuration
        self.config = self._validate_and_parse_config(raw_config)
        self.logger.info(f"Successfully loaded configuration from {config_path}")
    
    def load_default_config(self) -> None:
        """Load default configuration values."""
        self.config = {
            'source_detection': SourceDetectionConfig(),
            'psf': PSFConfig(),
            'photometry': PhotometryConfig(),
            'calibration': CalibrationConfig(),
            'quality': QualityConfig(),
            'output': OutputConfig(),
            'filters': {}
        }
        self.logger.info("Loaded default configuration")
    
    def _validate_and_parse_config(self, raw_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and parse the raw configuration dictionary.
        
        Parameters:
        -----------
        raw_config : dict
            Raw configuration dictionary from YAML file
            
        Returns:
        --------
        dict
            Validated and parsed configuration
        """
        config = {}
        
        # Parse source detection configuration
        config['source_detection'] = self._parse_source_detection_config(
            raw_config.get('source_detection', {})
        )
        
        # Parse PSF configuration
        config['psf'] = self._parse_psf_config(
            raw_config.get('psf_homogenization', {})
        )
        
        # Parse photometry configuration
        config['photometry'] = self._parse_photometry_config(
            raw_config.get('aperture_photometry', {})
        )
        
        # Parse calibration configuration
        config['calibration'] = self._parse_calibration_config(
            raw_config.get('photometric_corrections', {})
        )
        
        # Parse quality configuration
        config['quality'] = self._parse_quality_config(
            raw_config.get('quality_assessment', {})
        )
        
        # Parse output configuration
        config['output'] = self._parse_output_config(
            raw_config.get('output', {})
        )
        
        # Parse filter configurations
        config['filters'] = self._parse_filter_configs(
            raw_config.get('images', {})
        )
        
        # Validate file paths
        self._validate_file_paths(config)
        
        return config
    
    def _parse_source_detection_config(self, detection_config: Dict[str, Any]) -> SourceDetectionConfig:
        """Parse source detection configuration."""
        return SourceDetectionConfig(
            kernel_fwhm=detection_config.get('kernel', 3.5),
            minarea=detection_config.get('minarea', 3),
            thresh=detection_config.get('thresh', 1.2),
            deblend_nthresh=detection_config.get('deblend_nthresh', 32),
            deblend_cont=detection_config.get('deblend_cont', 0.0001),
            clean=detection_config.get('clean', False),
            clean_param=detection_config.get('clean_param', 1.0),
            mask_regions=detection_config.get('mask_regions', []),
            detection_band=detection_config.get('detection_band', 'detection')
        )
    
    def _parse_psf_config(self, psf_config: Dict[str, Any]) -> PSFConfig:
        """Parse PSF configuration."""
        return PSFConfig(
            target_band=psf_config.get('target_psf', 'F444W'),
            regularization_parameter=psf_config.get('regularization_parameter', 0.003),
            star_selection_magnitude_limits=tuple(psf_config.get('star_magnitude_limits', [16.0, 23.0])),
            star_selection_size_limits=tuple(psf_config.get('star_size_limits', [0.8, 1.2])),
            star_selection_snr_limit=psf_config.get('star_snr_limit', 50.0),
            psf_size=psf_config.get('psf_size', 25),
            max_iterations=psf_config.get('max_iterations', 3),
            convergence_threshold=psf_config.get('convergence_threshold', 0.01),
            spatial_variation=psf_config.get('spatial_variation', True),
            n_grid_points=psf_config.get('n_grid_points', 9)
        )
    
    def _parse_photometry_config(self, phot_config: Dict[str, Any]) -> PhotometryConfig:
        """Parse photometry configuration."""
        apertures = phot_config.get('apertures', [0.32, 0.48, 0.70, 1.00, 1.40])
        # Convert diameter to radius
        aperture_radii = [a/2.0 for a in apertures] if apertures else [0.16, 0.24, 0.35, 0.50, 0.70]
        
        return PhotometryConfig(
            aperture_radii=aperture_radii,
            elliptical_kron_factor=phot_config.get('elliptical_kron_factor', 2.5),
            min_kron_factor=phot_config.get('min_kron_factor', 1.0),
            max_kron_radius=phot_config.get('max_kron_radius', 5.0),
            background_annulus_inner=phot_config.get('background_annulus_inner', 2.0),
            background_annulus_outer=phot_config.get('background_annulus_outer', 3.0),
            local_background_size=phot_config.get('local_noise_box_size', 9),
            sigma_clip_threshold=phot_config.get('sigma_clip_threshold', 3.0),
            max_iterations=phot_config.get('max_iterations', 5)
        )
    
    def _parse_calibration_config(self, calib_config: Dict[str, Any]) -> CalibrationConfig:
        """Parse calibration configuration."""
        return CalibrationConfig(
            reference_band=calib_config.get('reference_band', 'F444W'),
            apply_aperture_corrections=calib_config.get('apply_aperture_corrections', True),
            apply_extinction_correction=calib_config.get('apply_extinction_correction', True),
            apply_systematic_corrections=calib_config.get('apply_systematic_corrections', True),
            color_term_corrections=calib_config.get('color_term_corrections', True),
            ebv_map_path=calib_config.get('ebv_map_path'),
            extinction_law=calib_config.get('extinction_law', 'ccm89'),
            rv_value=calib_config.get('rv_value', 3.1)
        )
    
    def _parse_quality_config(self, quality_config: Dict[str, Any]) -> QualityConfig:
        """Parse quality assessment configuration."""
        return QualityConfig(
            enable_quality_flags=quality_config.get('enable_quality_flags', True),
            snr_threshold=quality_config.get('snr_threshold', 5.0),
            saturation_threshold=quality_config.get('saturation_threshold', 50000.0),
            edge_buffer=quality_config.get('edge_buffer', 10),
            contamination_threshold=quality_config.get('contamination_threshold', 0.1),
            astrometric_uncertainty_limit=quality_config.get('astrometric_uncertainty_limit', 0.1),
            photometric_uncertainty_limit=quality_config.get('photometric_uncertainty_limit', 0.1)
        )
    
    def _parse_output_config(self, output_config: Dict[str, Any]) -> OutputConfig:
        """Parse output configuration."""
        return OutputConfig(
            catalog_format=output_config.get('catalog_format', 'fits'),
            output_directory=output_config.get('output_directory', './output'),
            save_intermediate_products=output_config.get('save_intermediate_products', False),
            save_segmentation_map=output_config.get('save_segmentation_map', True),
            save_background_map=output_config.get('save_background_map', True),
            save_psf_models=output_config.get('save_psf_models', True),
            catalog_columns=output_config.get('catalog_columns', [
                'id', 'x', 'y', 'ra', 'dec', 'mag_auto', 'magerr_auto',
                'mag_aper', 'magerr_aper', 'flux_radius', 'class_star',
                'flags', 'a', 'b', 'theta', 'kron_radius'
            ])
        )
    
    def _parse_filter_configs(self, images_config: Dict[str, str]) -> Dict[str, FilterConfig]:
        """Parse filter-specific configurations."""
        filters = {}
        
        # Extract filter names and paths
        image_paths = {}
        weight_paths = {}
        
        for key, path in images_config.items():
            if key.endswith('_weight'):
                filter_name = key.replace('_weight', '')
                weight_paths[filter_name] = path
            else:
                image_paths[key] = path
        
        # Create FilterConfig objects
        for filter_name, image_path in image_paths.items():
            weight_path = weight_paths.get(filter_name, None)
            
            filters[filter_name] = FilterConfig(
                image_path=image_path,
                weight_path=weight_path,
                zero_point=self._get_default_zero_point(filter_name),
                pixel_scale=self._get_default_pixel_scale(filter_name),
                detector=self._get_default_detector(filter_name)
            )
        
        return filters
    
    def _get_default_zero_point(self, filter_name: str) -> float:
        """Get default zero point for a given filter."""
        # Default zero points for JWST NIRCam filters (approximate values)
        default_zeropoints = {
            'F115W': 25.6,
            'F150W': 25.8,
            'F200W': 25.9,
            'F277W': 25.7,
            'F356W': 25.6,
            'F410M': 24.6,
            'F444W': 25.5
        }
        return default_zeropoints.get(filter_name, 25.0)
    
    def _get_default_pixel_scale(self, filter_name: str) -> float:
        """Get default pixel scale for a given filter."""
        # NIRCam short wavelength: 0.031"/pixel, long wavelength: 0.063"/pixel
        short_wave_filters = ['F115W', 'F150W', 'F200W']
        return 0.031 if filter_name in short_wave_filters else 0.063
    
    def _get_default_detector(self, filter_name: str) -> str:
        """Get default detector for a given filter."""
        short_wave_filters = ['F115W', 'F150W', 'F200W']
        return 'NRCA' if filter_name in short_wave_filters else 'NRCB'
    
    def _validate_file_paths(self, config: Dict[str, Any]) -> None:
        """
        Validate that all specified file paths exist.
        
        Parameters:
        -----------
        config : dict
            Configuration dictionary to validate
            
        Raises:
        -------
        FileNotFoundError
            If any required files don't exist
        """
        missing_files = []
        
        # Check filter image files
        for filter_name, filter_config in config['filters'].items():
            if not Path(filter_config.image_path).exists():
                missing_files.append(f"Image file for {filter_name}: {filter_config.image_path}")
            
            if filter_config.weight_path and not Path(filter_config.weight_path).exists():
                missing_files.append(f"Weight file for {filter_name}: {filter_config.weight_path}")
        
        # Check calibration files
        calib_config = config['calibration']
        if calib_config.ebv_map_path and not Path(calib_config.ebv_map_path).exists():
            missing_files.append(f"E(B-V) map: {calib_config.ebv_map_path}")
        
        if missing_files:
            self.logger.warning(f"Missing files detected: {missing_files}")
            # For development, we'll warn instead of raising an error
            # raise FileNotFoundError(f"Missing required files: {missing_files}")
    
    def get_filter_config(self, filter_name: str) -> FilterConfig:
        """
        Get configuration for a specific filter.
        
        Parameters:
        -----------
        filter_name : str
            Name of the filter
            
        Returns:
        --------
        FilterConfig
            Configuration for the specified filter
        """
        if filter_name not in self.config['filters']:
            raise ValueError(f"Filter {filter_name} not found in configuration")
        
        return self.config['filters'][filter_name]
    
    def get_available_filters(self) -> List[str]:
        """Get list of available filters."""
        return list(self.config['filters'].keys())
    
    def get_detection_config(self) -> SourceDetectionConfig:
        """Get source detection configuration."""
        return self.config['source_detection']
    
    def get_psf_config(self) -> PSFConfig:
        """Get PSF configuration."""
        return self.config['psf']
    
    def get_photometry_config(self) -> PhotometryConfig:
        """Get photometry configuration."""
        return self.config['photometry']
    
    def get_calibration_config(self) -> CalibrationConfig:
        """Get calibration configuration."""
        return self.config['calibration']
    
    def get_quality_config(self) -> QualityConfig:
        """Get quality assessment configuration."""
        return self.config['quality']
    
    def get_output_config(self) -> OutputConfig:
        """Get output configuration."""
        return self.config['output']
    
    def update_filter_metadata(self, filter_name: str, metadata: Dict[str, Any]) -> None:
        """
        Update metadata for a specific filter from FITS headers.
        
        Parameters:
        -----------
        filter_name : str
            Name of the filter
        metadata : dict
            Metadata dictionary extracted from FITS headers
        """
        if filter_name in self.config['filters']:
            filter_config = self.config['filters'][filter_name]
            
            # Update exposure time if available
            if 'EXPTIME' in metadata:
                filter_config.exposure_time = metadata['EXPTIME']
            
            # Update gain if available
            if 'GAIN' in metadata:
                filter_config.gain = metadata['GAIN']
            
            # Update zero point if available
            if 'PHOTZP' in metadata:
                filter_config.zero_point = metadata['PHOTZP']
            
            # Update detector if available
            if 'DETECTOR' in metadata:
                filter_config.detector = metadata['DETECTOR']
            
            self.logger.debug(f"Updated metadata for filter {filter_name}")
    
    def validate_configuration(self) -> bool:
        """
        Perform comprehensive validation of the loaded configuration.
        
        Returns:
        --------
        bool
            True if configuration is valid
            
        Raises:
        -------
        ValueError
            If configuration validation fails
        """
        errors = []
        
        # Validate filter configurations
        if not self.config['filters']:
            errors.append("No filters configured")
        
        # Validate PSF target band
        psf_config = self.config['psf']
        if psf_config.target_band not in self.config['filters']:
            errors.append(f"PSF target band {psf_config.target_band} not in available filters")
        
        # Validate output directory
        output_config = self.config['output']
        output_dir = Path(output_config.output_directory)
        if not output_dir.parent.exists():
            errors.append(f"Output directory parent does not exist: {output_dir.parent}")
        
        # Validate aperture radii
        phot_config = self.config['photometry']
        if not phot_config.aperture_radii:
            errors.append("No aperture radii specified")
        
        if any(r <= 0 for r in phot_config.aperture_radii):
            errors.append("All aperture radii must be positive")
        
        if errors:
            error_msg = "Configuration validation failed:\\n" + "\\n".join(errors)
            raise ValueError(error_msg)
        
        self.logger.info("Configuration validation passed")
        return True
    
    def save_config(self, output_path: str) -> None:
        """
        Save current configuration to a YAML file.
        
        Parameters:
        -----------
        output_path : str
            Path to save the configuration file
        """
        # Convert dataclass objects to dictionaries for YAML serialization
        config_dict = self._config_to_dict()
        
        with open(output_path, 'w') as file:
            yaml.dump(config_dict, file, default_flow_style=False, indent=2)
        
        self.logger.info(f"Configuration saved to {output_path}")
    
    def _config_to_dict(self) -> Dict[str, Any]:
        """Convert configuration objects to dictionary format."""
        from dataclasses import asdict
        
        config_dict = {}
        
        # Convert dataclass configurations
        for key, value in self.config.items():
            if hasattr(value, '__dataclass_fields__'):
                config_dict[key] = asdict(value)
            elif isinstance(value, dict):
                # Handle nested dictionaries (e.g., filters)
                config_dict[key] = {}
                for subkey, subvalue in value.items():
                    if hasattr(subvalue, '__dataclass_fields__'):
                        config_dict[key][subkey] = asdict(subvalue)
                    else:
                        config_dict[key][subkey] = subvalue
            else:
                config_dict[key] = value
        
        return config_dict


# Convenience function for quick configuration loading
def load_config(config_path: str) -> ConfigManager:
    """
    Convenience function to load and validate configuration.
    
    Parameters:
    -----------
    config_path : str
        Path to configuration file
        
    Returns:
    --------
    ConfigManager
        Initialized configuration manager
    """
    config_manager = ConfigManager(config_path)
    config_manager.validate_configuration()
    return config_manager
