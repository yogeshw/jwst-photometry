"""
Enhanced Aperture Photometry Module for JWST Photometry

This module implements sophisticated aperture photometry capabilities including
multiple aperture sizes, elliptical apertures, Kron apertures, local background
estimation, and contamination correction.
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple, List, Union
from dataclasses import dataclass, field
from pathlib import Path

import sep
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
from scipy import ndimage, optimize
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

# Import our modules with fallbacks
try:
    from .segmentation import AdvancedSegmentationProcessor, SegmentationResults
except ImportError:
    print("Warning: segmentation module not available")
    # Create dummy classes
    class AdvancedSegmentationProcessor:
        def __init__(self, *args, **kwargs): pass
    class SegmentationResults:
        def __init__(self, *args, **kwargs): pass

try:
    from .utils import setup_logger, memory_monitor, validate_array
except ImportError:
    def setup_logger(name): return logging.getLogger(name)
    def memory_monitor(func): return func
    def validate_array(arr): return arr is not None


@dataclass
class AperturePhotometryConfig:
    """Configuration parameters for aperture photometry."""
    
    # Aperture sizes and types
    circular_apertures: List[float] = field(default_factory=lambda: [1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0])  # in pixels
    use_elliptical_apertures: bool = True
    use_kron_apertures: bool = True
    use_adaptive_apertures: bool = True
    
    # Elliptical aperture parameters
    elliptical_scale_factors: List[float] = field(default_factory=lambda: [1.0, 2.0, 3.0, 5.0])
    max_elliptical_ratio: float = 5.0  # Maximum a/b ratio
    
    # Kron aperture parameters
    kron_factor: float = 2.5
    min_kron_radius: float = 1.0
    max_kron_radius: float = 15.0
    kron_minimum_radius: float = 1.75  # In units of semi-major axis
    
    # Background estimation
    background_method: str = "local_annulus"  # "local_annulus", "global", "segmentation"
    annulus_inner_factor: float = 1.5  # Inner radius = aperture_radius * factor
    annulus_outer_factor: float = 2.5  # Outer radius = aperture_radius * factor
    background_sigma_clip: float = 3.0
    background_iterations: int = 3
    
    # Contamination correction
    correct_contamination: bool = True
    contamination_threshold: float = 0.1  # Fraction of aperture area
    neighbor_search_radius: float = 10.0  # Pixels
    
    # Aperture corrections
    apply_aperture_corrections: bool = True
    aperture_correction_radius: float = 10.0  # Reference radius for total flux
    curve_of_growth_method: str = "empirical"  # "empirical", "psf_based"
    
    # Error estimation
    estimate_uncertainties: bool = True
    uncertainty_method: str = "local_background"  # "local_background", "poisson", "combined"
    n_background_samples: int = 100
    
    # Quality control
    flag_contaminated: bool = True
    flag_edge_sources: bool = True
    edge_buffer: int = 10
    flag_saturated: bool = True
    saturation_threshold: float = 50000.0  # ADU
    
    # Processing options
    parallel_processing: bool = False
    n_processes: int = 4
    save_aperture_masks: bool = False
    create_diagnostic_plots: bool = True


@dataclass
class AperturePhotometrySource:
    """Container for aperture photometry source data."""
    
    id: int
    x: float
    y: float
    ra: Optional[float] = None
    dec: Optional[float] = None
    
    # Source properties
    a: float = 2.0  # Semi-major axis
    b: float = 2.0  # Semi-minor axis
    theta: float = 0.0  # Position angle
    
    # Photometry results
    circular_fluxes: Dict[float, float] = field(default_factory=dict)
    circular_flux_errors: Dict[float, float] = field(default_factory=dict)
    circular_magnitudes: Dict[float, float] = field(default_factory=dict)
    circular_magnitude_errors: Dict[float, float] = field(default_factory=dict)
    
    elliptical_fluxes: Dict[float, float] = field(default_factory=dict)
    elliptical_flux_errors: Dict[float, float] = field(default_factory=dict)
    
    kron_flux: Optional[float] = None
    kron_flux_error: Optional[float] = None
    kron_magnitude: Optional[float] = None
    kron_magnitude_error: Optional[float] = None
    kron_radius: Optional[float] = None
    
    adaptive_flux: Optional[float] = None
    adaptive_flux_error: Optional[float] = None
    adaptive_radius: Optional[float] = None
    
    # Background measurements
    local_background: Optional[float] = None
    background_rms: Optional[float] = None
    background_area: Optional[float] = None
    
    # Aperture corrections
    aperture_corrections: Dict[float, float] = field(default_factory=dict)
    total_flux: Optional[float] = None
    total_magnitude: Optional[float] = None
    
    # Quality flags
    flags: List[str] = field(default_factory=list)
    contamination_fraction: float = 0.0
    
    # Diagnostic information
    aperture_masks: Dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class AperturePhotometryResults:
    """Container for aperture photometry results."""
    
    sources: List[AperturePhotometrySource]
    config: AperturePhotometryConfig
    
    # Global statistics
    statistics: Dict[str, Any] = field(default_factory=dict)
    
    # Aperture correction curves
    aperture_corrections: Dict[float, float] = field(default_factory=dict)
    curve_of_growth: Optional[np.ndarray] = None
    
    # Processing information
    processing_time: float = 0.0
    n_sources_processed: int = 0
    n_sources_successful: int = 0
    
    # Diagnostic information
    diagnostics: Dict[str, Any] = field(default_factory=dict)


class EnhancedAperturePhotometry:
    """
    Enhanced aperture photometry processor for JWST observations.
    
    This class provides sophisticated aperture photometry capabilities including:
    - Multiple circular aperture sizes
    - Elliptical apertures based on source morphology
    - Kron apertures for extended sources
    - Adaptive apertures
    - Local background estimation
    - Contamination correction
    - Aperture corrections
    - Comprehensive error estimation
    """
    
    def __init__(self, config: Optional[AperturePhotometryConfig] = None):
        """
        Initialize the aperture photometry processor.
        
        Parameters:
        -----------
        config : AperturePhotometryConfig, optional
            Aperture photometry configuration. If None, uses defaults.
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or AperturePhotometryConfig()
    
    def perform_aperture_photometry(self,
                                  image: np.ndarray,
                                  sources: np.ndarray,
                                  background_map: Optional[np.ndarray] = None,
                                  rms_map: Optional[np.ndarray] = None,
                                  segmentation_map: Optional[np.ndarray] = None,
                                  psf_model: Optional[np.ndarray] = None,
                                  wcs: Optional[WCS] = None) -> AperturePhotometryResults:
        """
        Perform comprehensive aperture photometry.
        
        Parameters:
        -----------
        image : numpy.ndarray
            Input image data
        sources : numpy.ndarray
            Detected sources catalog
        background_map : numpy.ndarray, optional
            Background map
        rms_map : numpy.ndarray, optional
            RMS/noise map
        segmentation_map : numpy.ndarray, optional
            Source segmentation map
        psf_model : numpy.ndarray, optional
            PSF model for aperture corrections
        wcs : astropy.wcs.WCS, optional
            World coordinate system
            
        Returns:
        --------
        AperturePhotometryResults
            Complete aperture photometry results
        """
        self.logger.info("Starting enhanced aperture photometry")
        
        import time
        start_time = time.time()
        
        # Validate inputs
        self._validate_inputs(image, sources, background_map, rms_map)
        
        # Prepare photometry sources
        phot_sources = self._prepare_sources(sources, wcs)
        
        # Perform photometry
        phot_sources = self._perform_photometry(
            phot_sources, image, background_map, rms_map, segmentation_map
        )
        
        # Apply aperture corrections if requested
        if self.config.apply_aperture_corrections:
            aperture_corrections = self._compute_aperture_corrections(
                phot_sources, image, psf_model, background_map
            )
            phot_sources = self._apply_aperture_corrections(phot_sources, aperture_corrections)
        else:
            aperture_corrections = {}
        
        # Compute curve of growth if PSF model available
        curve_of_growth = None
        if psf_model is not None:
            curve_of_growth = self._compute_curve_of_growth(psf_model)
        
        # Compute statistics
        statistics = self._compute_statistics(phot_sources)
        
        # Create diagnostics
        diagnostics = self._create_diagnostics(phot_sources, image)
        
        processing_time = time.time() - start_time
        
        results = AperturePhotometryResults(
            sources=phot_sources,
            config=self.config,
            statistics=statistics,
            aperture_corrections=aperture_corrections,
            curve_of_growth=curve_of_growth,
            processing_time=processing_time,
            n_sources_processed=len(phot_sources),
            n_sources_successful=len([s for s in phot_sources if s.circular_fluxes]),
            diagnostics=diagnostics
        )
        
        self.logger.info(f"Aperture photometry completed in {processing_time:.2f} seconds")
        self.logger.info(f"Successfully processed {results.n_sources_successful}/{results.n_sources_processed} sources")
        
        return results
    
    def _validate_inputs(self, image: np.ndarray, sources: np.ndarray,
                        background_map: Optional[np.ndarray],
                        rms_map: Optional[np.ndarray]) -> None:
        """Validate input data."""
        validate_array(image, name="image")
        
        if len(sources) == 0:
            raise ValueError("No sources provided")
        
        if background_map is not None:
            validate_array(background_map, name="background_map")
            if background_map.shape != image.shape:
                raise ValueError("Background map shape doesn't match image shape")
        
        if rms_map is not None:
            validate_array(rms_map, name="rms_map")
            if rms_map.shape != image.shape:
                raise ValueError("RMS map shape doesn't match image shape")
    
    def _prepare_sources(self, sources: np.ndarray, wcs: Optional[WCS]) -> List[AperturePhotometrySource]:
        """
        Prepare source list for aperture photometry.
        
        Parameters:
        -----------
        sources : numpy.ndarray
            Source catalog (can be structured array or array of Source objects)
        wcs : astropy.wcs.WCS, optional
            World coordinate system
            
        Returns:
        --------
        list
            List of aperture photometry sources
        """
        phot_sources = []
        
        for i, source in enumerate(sources):
            try:
                # Handle both numpy structured arrays and Source dataclass objects
                if hasattr(source, 'x') and hasattr(source, 'y'):
                    # Source dataclass object
                    x, y = source.x, source.y
                    a = getattr(source, 'a', 2.0)
                    b = getattr(source, 'b', 2.0)
                    theta = getattr(source, 'theta', 0.0)
                elif isinstance(source, np.void):
                    # Numpy structured array record
                    x, y = source['x'], source['y']
                    a = source['a'] if 'a' in source.dtype.names else 2.0
                    b = source['b'] if 'b' in source.dtype.names else 2.0
                    theta = source['theta'] if 'theta' in source.dtype.names else 0.0
                else:
                    # Try accessing as structured array
                    x, y = source['x'], source['y']
                    a = source['a'] if 'a' in source.dtype.names else 2.0
                    b = source['b'] if 'b' in source.dtype.names else 2.0
                    theta = source['theta'] if 'theta' in source.dtype.names else 0.0
                
                phot_source = AperturePhotometrySource(
                    id=i,
                    x=float(x),
                    y=float(y),
                    a=float(a),
                    b=float(b),
                    theta=float(theta)
                )
                
                # Convert to sky coordinates if WCS available
                if wcs is not None:
                    try:
                        sky_coord = wcs.pixel_to_world(phot_source.x, phot_source.y)
                        phot_source.ra = sky_coord.ra.degree
                        phot_source.dec = sky_coord.dec.degree
                    except Exception:
                        pass
                
                phot_sources.append(phot_source)
                
            except Exception as e:
                self.logger.warning(f"Could not process source {i}: {e}")
                continue
        
        return phot_sources
    
    def _perform_photometry(self,
                           sources: List[AperturePhotometrySource],
                           image: np.ndarray,
                           background_map: Optional[np.ndarray],
                           rms_map: Optional[np.ndarray],
                           segmentation_map: Optional[np.ndarray]) -> List[AperturePhotometrySource]:
        """
        Perform photometry for all sources.
        
        Parameters:
        -----------
        sources : list
            List of aperture photometry sources
        image : numpy.ndarray
            Input image
        background_map : numpy.ndarray, optional
            Background map
        rms_map : numpy.ndarray, optional
            RMS map
        segmentation_map : numpy.ndarray, optional
            Segmentation map
            
        Returns:
        --------
        list
            Updated sources with photometry results
        """
        self.logger.debug("Performing photometry measurements")
        
        for source in sources:
            try:
                # Quality checks
                self._perform_quality_checks(source, image)
                
                # Estimate local background
                if self.config.background_method == "local_annulus":
                    self._estimate_local_background(source, image, background_map, rms_map)
                elif background_map is not None:
                    source.local_background = background_map[int(source.y), int(source.x)]
                    if rms_map is not None:
                        source.background_rms = rms_map[int(source.y), int(source.x)]
                
                # Circular aperture photometry
                self._perform_circular_photometry(source, image, rms_map, segmentation_map)
                
                # Elliptical aperture photometry
                if self.config.use_elliptical_apertures:
                    self._perform_elliptical_photometry(source, image, rms_map, segmentation_map)
                
                # Kron aperture photometry
                if self.config.use_kron_apertures:
                    self._perform_kron_photometry(source, image, rms_map, segmentation_map)
                
                # Adaptive aperture photometry
                if self.config.use_adaptive_apertures:
                    self._perform_adaptive_photometry(source, image, rms_map, segmentation_map)
                
                # Check for contamination
                if self.config.correct_contamination and segmentation_map is not None:
                    self._check_contamination(source, segmentation_map)
                
            except Exception as e:
                self.logger.debug(f"Photometry failed for source {source.id}: {e}")
                source.flags.append("photometry_failed")
        
        return sources
    
    def _perform_quality_checks(self, source: AperturePhotometrySource, image: np.ndarray) -> None:
        """Perform quality checks on source."""
        # Check if source is near edge
        if self.config.flag_edge_sources:
            buffer = self.config.edge_buffer
            if (source.x < buffer or source.x >= image.shape[1] - buffer or
                source.y < buffer or source.y >= image.shape[0] - buffer):
                source.flags.append("near_edge")
        
        # Check for saturation
        if self.config.flag_saturated:
            try:
                peak_value = image[int(source.y), int(source.x)]
                if peak_value > self.config.saturation_threshold:
                    source.flags.append("saturated")
            except IndexError:
                source.flags.append("outside_image")
    
    def _estimate_local_background(self,
                                 source: AperturePhotometrySource,
                                 image: np.ndarray,
                                 background_map: Optional[np.ndarray],
                                 rms_map: Optional[np.ndarray]) -> None:
        """
        Estimate local background using annulus.
        
        Parameters:
        -----------
        source : AperturePhotometrySource
            Source for background estimation
        image : numpy.ndarray
            Input image
        background_map : numpy.ndarray, optional
            Global background map
        rms_map : numpy.ndarray, optional
            RMS map
        """
        try:
            # Use largest circular aperture for reference
            max_aperture = max(self.config.circular_apertures)
            inner_radius = max_aperture * self.config.annulus_inner_factor
            outer_radius = max_aperture * self.config.annulus_outer_factor
            
            # Create annular mask
            y, x = np.ogrid[:image.shape[0], :image.shape[1]]
            distance = np.sqrt((x - source.x)**2 + (y - source.y)**2)
            
            annulus_mask = (distance >= inner_radius) & (distance <= outer_radius)
            
            if np.sum(annulus_mask) < 10:  # Not enough pixels
                if background_map is not None:
                    source.local_background = background_map[int(source.y), int(source.x)]
                else:
                    source.local_background = 0.0
                return
            
            # Extract background pixels
            background_pixels = image[annulus_mask]
            
            # Apply sigma clipping
            for _ in range(self.config.background_iterations):
                mean_bg = np.mean(background_pixels)
                std_bg = np.std(background_pixels)
                
                good_pixels = np.abs(background_pixels - mean_bg) < self.config.background_sigma_clip * std_bg
                background_pixels = background_pixels[good_pixels]
                
                if len(background_pixels) < 5:
                    break
            
            if len(background_pixels) > 0:
                source.local_background = np.median(background_pixels)
                source.background_rms = np.std(background_pixels)
                source.background_area = len(background_pixels)
            else:
                source.local_background = 0.0
                source.background_rms = 1.0
                source.flags.append("background_estimation_failed")
            
        except Exception:
            source.local_background = 0.0
            source.background_rms = 1.0
            source.flags.append("background_estimation_failed")
    
    def _perform_circular_photometry(self,
                                   source: AperturePhotometrySource,
                                   image: np.ndarray,
                                   rms_map: Optional[np.ndarray],
                                   segmentation_map: Optional[np.ndarray]) -> None:
        """
        Perform circular aperture photometry.
        
        Parameters:
        -----------
        source : AperturePhotometrySource
            Source for photometry
        image : numpy.ndarray
            Input image
        rms_map : numpy.ndarray, optional
            RMS map
        segmentation_map : numpy.ndarray, optional
            Segmentation map
        """
        for radius in self.config.circular_apertures:
            try:
                # Use SEP for circular aperture photometry
                flux, fluxerr, flag = sep.sum_circle(
                    image,
                    [source.x], [source.y], [radius],
                    err=rms_map,
                    segmap=segmentation_map,
                    gain=1.0,
                    subpix=5
                )
                
                # Store results
                source.circular_fluxes[radius] = flux[0]
                source.circular_flux_errors[radius] = fluxerr[0] if fluxerr is not None else np.sqrt(abs(flux[0]))
                
                # Convert to magnitude
                if flux[0] > 0:
                    source.circular_magnitudes[radius] = -2.5 * np.log10(flux[0]) + 25.0  # Arbitrary zeropoint
                    if source.circular_flux_errors[radius] > 0:
                        source.circular_magnitude_errors[radius] = 1.0857 * source.circular_flux_errors[radius] / flux[0]
                    else:
                        source.circular_magnitude_errors[radius] = 99.0
                else:
                    source.circular_magnitudes[radius] = 99.0
                    source.circular_magnitude_errors[radius] = 99.0
                    source.flags.append(f"negative_flux_r{radius}")
                
                # Check flags
                if flag[0] != 0:
                    source.flags.append(f"sep_flag_r{radius}_{flag[0]}")
                
            except Exception as e:
                self.logger.debug(f"Circular photometry failed for source {source.id}, radius {radius}: {e}")
                source.flags.append(f"circular_photometry_failed_r{radius}")
    
    def _perform_elliptical_photometry(self,
                                     source: AperturePhotometrySource,
                                     image: np.ndarray,
                                     rms_map: Optional[np.ndarray],
                                     segmentation_map: Optional[np.ndarray]) -> None:
        """
        Perform elliptical aperture photometry.
        
        Parameters:
        -----------
        source : AperturePhotometrySource
            Source for photometry
        image : numpy.ndarray
            Input image
        rms_map : numpy.ndarray, optional
            RMS map
        segmentation_map : numpy.ndarray, optional
            Segmentation map
        """
        for scale_factor in self.config.elliptical_scale_factors:
            try:
                # Calculate elliptical aperture parameters
                a_aper = scale_factor * source.a
                b_aper = scale_factor * source.b
                
                # Check ellipticity constraint
                if a_aper / b_aper > self.config.max_elliptical_ratio:
                    continue
                
                # Use SEP for elliptical aperture photometry
                flux, fluxerr, flag = sep.sum_ellipse(
                    image,
                    [source.x], [source.y],
                    [a_aper], [b_aper], [source.theta],
                    err=rms_map,
                    segmap=segmentation_map,
                    gain=1.0,
                    subpix=5
                )
                
                # Store results
                source.elliptical_fluxes[scale_factor] = flux[0]
                source.elliptical_flux_errors[scale_factor] = fluxerr[0] if fluxerr is not None else np.sqrt(abs(flux[0]))
                
                # Check flags
                if flag[0] != 0:
                    source.flags.append(f"elliptical_sep_flag_s{scale_factor}_{flag[0]}")
                
            except Exception as e:
                self.logger.debug(f"Elliptical photometry failed for source {source.id}, scale {scale_factor}: {e}")
                source.flags.append(f"elliptical_photometry_failed_s{scale_factor}")
    
    def _perform_kron_photometry(self,
                               source: AperturePhotometrySource,
                               image: np.ndarray,
                               rms_map: Optional[np.ndarray],
                               segmentation_map: Optional[np.ndarray]) -> None:
        """
        Perform Kron aperture photometry.
        
        Parameters:
        -----------
        source : AperturePhotometrySource
            Source for photometry
        image : numpy.ndarray
            Input image
        rms_map : numpy.ndarray, optional
            RMS map
        segmentation_map : numpy.ndarray, optional
            Segmentation map
        """
        try:
            # Calculate Kron radius
            kron_radius = self._calculate_kron_radius(source, image)
            
            # Apply constraints
            kron_radius = np.clip(kron_radius, self.config.min_kron_radius, self.config.max_kron_radius)
            
            # Ensure minimum radius
            min_radius = self.config.kron_minimum_radius * max(source.a, source.b)
            kron_radius = max(kron_radius, min_radius)
            
            source.kron_radius = kron_radius
            
            # Perform Kron aperture photometry
            flux, fluxerr, flag = sep.sum_ellipse(
                image,
                [source.x], [source.y],
                [kron_radius * source.a / np.sqrt(source.a * source.b)],
                [kron_radius * source.b / np.sqrt(source.a * source.b)],
                [source.theta],
                err=rms_map,
                segmap=segmentation_map,
                gain=1.0,
                subpix=5
            )
            
            # Store results
            source.kron_flux = flux[0]
            source.kron_flux_error = fluxerr[0] if fluxerr is not None else np.sqrt(abs(flux[0]))
            
            # Convert to magnitude
            if flux[0] > 0:
                source.kron_magnitude = -2.5 * np.log10(flux[0]) + 25.0
                if source.kron_flux_error > 0:
                    source.kron_magnitude_error = 1.0857 * source.kron_flux_error / flux[0]
                else:
                    source.kron_magnitude_error = 99.0
            else:
                source.kron_magnitude = 99.0
                source.kron_magnitude_error = 99.0
                source.flags.append("negative_kron_flux")
            
            # Check flags
            if flag[0] != 0:
                source.flags.append(f"kron_sep_flag_{flag[0]}")
            
        except Exception as e:
            self.logger.debug(f"Kron photometry failed for source {source.id}: {e}")
            source.flags.append("kron_photometry_failed")
    
    def _calculate_kron_radius(self, source: AperturePhotometrySource, image: np.ndarray) -> float:
        """
        Calculate Kron radius for a source.
        
        Parameters:
        -----------
        source : AperturePhotometrySource
            Source for Kron radius calculation
        image : numpy.ndarray
            Input image
            
        Returns:
        --------
        float
            Kron radius in pixels
        """
        try:
            # Use series of elliptical apertures to measure flux growth
            max_radius = min(3 * max(source.a, source.b), 20.0)
            radii = np.linspace(0.5, max_radius, 15)
            
            fluxes = []
            mean_radii = []
            
            for radius in radii:
                try:
                    flux, _, _ = sep.sum_ellipse(
                        image, [source.x], [source.y],
                        [radius * source.a / np.sqrt(source.a * source.b)],
                        [radius * source.b / np.sqrt(source.a * source.b)],
                        [source.theta],
                        r=radius,
                        subpix=5
                    )
                    
                    if flux[0] > 0:
                        fluxes.append(flux[0])
                        mean_radii.append(radius)
                
                except:
                    continue
            
            if len(fluxes) < 3:
                return self.config.kron_factor * np.sqrt(source.a * source.b)
            
            fluxes = np.array(fluxes)
            mean_radii = np.array(mean_radii)
            
            # Calculate Kron radius using differential fluxes
            if len(fluxes) > 1:
                diff_fluxes = np.diff(fluxes)
                diff_radii = mean_radii[1:]
                
                if len(diff_fluxes) > 0 and np.sum(diff_fluxes) > 0:
                    kron_radius = np.sum(diff_radii * diff_fluxes) / np.sum(diff_fluxes)
                    return self.config.kron_factor * kron_radius
            
            # Fallback to geometric mean
            return self.config.kron_factor * np.sqrt(source.a * source.b)
            
        except:
            return self.config.kron_factor * np.sqrt(source.a * source.b)
    
    def _perform_adaptive_photometry(self,
                                   source: AperturePhotometrySource,
                                   image: np.ndarray,
                                   rms_map: Optional[np.ndarray],
                                   segmentation_map: Optional[np.ndarray]) -> None:
        """
        Perform adaptive aperture photometry.
        
        Parameters:
        -----------
        source : AperturePhotometrySource
            Source for photometry
        image : numpy.ndarray
            Input image
        rms_map : numpy.ndarray, optional
            RMS map
        segmentation_map : numpy.ndarray, optional
            Segmentation map
        """
        try:
            # Calculate adaptive radius based on flux distribution
            adaptive_radius = self._calculate_adaptive_radius(source, image)
            source.adaptive_radius = adaptive_radius
            
            # Perform adaptive aperture photometry
            flux, fluxerr, flag = sep.sum_circle(
                image,
                [source.x], [source.y], [adaptive_radius],
                err=rms_map,
                segmap=segmentation_map,
                gain=1.0,
                subpix=5
            )
            
            # Store results
            source.adaptive_flux = flux[0]
            source.adaptive_flux_error = fluxerr[0] if fluxerr is not None else np.sqrt(abs(flux[0]))
            
            # Check flags
            if flag[0] != 0:
                source.flags.append(f"adaptive_sep_flag_{flag[0]}")
            
        except Exception as e:
            self.logger.debug(f"Adaptive photometry failed for source {source.id}: {e}")
            source.flags.append("adaptive_photometry_failed")
    
    def _calculate_adaptive_radius(self, source: AperturePhotometrySource, image: np.ndarray) -> float:
        """Calculate adaptive radius based on source properties."""
        try:
            # Use flux-weighted second moment
            stamp_size = int(10 * max(source.a, source.b))
            stamp_size = min(stamp_size, 50)  # Limit stamp size
            
            # Extract stamp
            x_int, y_int = int(source.x), int(source.y)
            half_size = stamp_size // 2
            
            x_min = max(0, x_int - half_size)
            x_max = min(image.shape[1], x_int + half_size + 1)
            y_min = max(0, y_int - half_size)
            y_max = min(image.shape[0], y_int + half_size + 1)
            
            stamp = image[y_min:y_max, x_min:x_max]
            
            if stamp.size == 0:
                return 3.0  # Default radius
            
            # Calculate flux-weighted center and second moments
            y_coords, x_coords = np.mgrid[:stamp.shape[0], :stamp.shape[1]]
            
            # Adjust coordinates to image coordinates
            x_coords = x_coords + x_min - source.x
            y_coords = y_coords + y_min - source.y
            
            # Use positive flux values
            weights = np.maximum(stamp, 0)
            total_weight = np.sum(weights)
            
            if total_weight <= 0:
                return 3.0
            
            # Calculate second moments
            m_xx = np.sum(x_coords**2 * weights) / total_weight
            m_yy = np.sum(y_coords**2 * weights) / total_weight
            
            # Adaptive radius based on second moment
            adaptive_radius = 2.5 * np.sqrt((m_xx + m_yy) / 2.0)
            
            # Apply reasonable limits
            adaptive_radius = np.clip(adaptive_radius, 1.0, 15.0)
            
            return adaptive_radius
            
        except:
            return 3.0  # Default radius
    
    def _check_contamination(self, source: AperturePhotometrySource, segmentation_map: np.ndarray) -> None:
        """
        Check for contamination from nearby sources.
        
        Parameters:
        -----------
        source : AperturePhotometrySource
            Source to check
        segmentation_map : numpy.ndarray
            Segmentation map
        """
        try:
            # Use largest circular aperture for contamination check
            max_aperture = max(self.config.circular_apertures)
            
            # Create circular mask
            y, x = np.ogrid[:segmentation_map.shape[0], :segmentation_map.shape[1]]
            distance = np.sqrt((x - source.x)**2 + (y - source.y)**2)
            aperture_mask = distance <= max_aperture
            
            # Check segmentation within aperture
            seg_values = segmentation_map[aperture_mask]
            source_id = source.id + 1  # Assuming 1-based segmentation IDs
            
            # Calculate contamination fraction
            total_pixels = np.sum(aperture_mask)
            contaminated_pixels = np.sum((seg_values != 0) & (seg_values != source_id))
            
            source.contamination_fraction = contaminated_pixels / total_pixels if total_pixels > 0 else 0.0
            
            # Flag if contamination is significant
            if source.contamination_fraction > self.config.contamination_threshold:
                source.flags.append("contaminated")
            
        except Exception:
            source.contamination_fraction = 0.0
            source.flags.append("contamination_check_failed")
    
    def _compute_aperture_corrections(self,
                                    sources: List[AperturePhotometrySource],
                                    image: np.ndarray,
                                    psf_model: Optional[np.ndarray],
                                    background_map: Optional[np.ndarray]) -> Dict[float, float]:
        """
        Compute aperture corrections.
        
        Parameters:
        -----------
        sources : list
            List of sources
        image : numpy.ndarray
            Input image
        psf_model : numpy.ndarray, optional
            PSF model
        background_map : numpy.ndarray, optional
            Background map
            
        Returns:
        --------
        dict
            Aperture corrections for each radius
        """
        aperture_corrections = {}
        
        if psf_model is None:
            # Use empirical approach with bright, isolated stars
            isolated_stars = self._select_aperture_correction_stars(sources)
            
            if len(isolated_stars) < 5:
                self.logger.warning("Not enough isolated stars for aperture corrections")
                # Return unit corrections
                for radius in self.config.circular_apertures:
                    aperture_corrections[radius] = 1.0
                return aperture_corrections
            
            # Compute corrections using isolated stars
            aperture_corrections = self._compute_empirical_corrections(isolated_stars, image)
            
        else:
            # Use PSF model to compute theoretical corrections
            aperture_corrections = self._compute_psf_based_corrections(psf_model)
        
        return aperture_corrections
    
    def _select_aperture_correction_stars(self, sources: List[AperturePhotometrySource]) -> List[AperturePhotometrySource]:
        """Select bright, isolated stars for aperture corrections."""
        # Select sources based on criteria
        candidates = []
        
        for source in sources:
            # Check if source has valid photometry
            if not source.circular_fluxes:
                continue
            
            # Check brightness (use largest aperture)
            max_aperture = max(self.config.circular_apertures)
            if max_aperture not in source.circular_fluxes:
                continue
            
            flux = source.circular_fluxes[max_aperture]
            if flux <= 0:
                continue
            
            # Check for flags that would disqualify
            disqualifying_flags = ["saturated", "near_edge", "contaminated"]
            if any(flag in source.flags for flag in disqualifying_flags):
                continue
            
            # Check isolation (simplified check)
            if source.contamination_fraction > 0.05:
                continue
            
            candidates.append(source)
        
        # Sort by brightness and take top candidates
        if candidates:
            max_aperture = max(self.config.circular_apertures)
            candidates.sort(key=lambda s: s.circular_fluxes[max_aperture], reverse=True)
            return candidates[:20]  # Take top 20 candidates
        
        return []
    
    def _compute_empirical_corrections(self,
                                     stars: List[AperturePhotometrySource],
                                     image: np.ndarray) -> Dict[float, float]:
        """Compute empirical aperture corrections using isolated stars."""
        aperture_corrections = {}
        
        # Reference radius for "total" flux
        ref_radius = self.config.aperture_correction_radius
        
        # Measure growth curves for selected stars
        growth_curves = []
        
        for star in stars:
            if not star.circular_fluxes:
                continue
            
            # Create growth curve for this star
            radii = sorted(star.circular_fluxes.keys())
            fluxes = [star.circular_fluxes[r] for r in radii]
            
            # Extrapolate to reference radius if needed
            if ref_radius not in star.circular_fluxes:
                # Use largest measured aperture as proxy
                ref_flux = max(fluxes)
            else:
                ref_flux = star.circular_fluxes[ref_radius]
            
            if ref_flux > 0:
                growth_curve = {r: flux / ref_flux for r, flux in zip(radii, fluxes)}
                growth_curves.append(growth_curve)
        
        # Compute median corrections
        for radius in self.config.circular_apertures:
            corrections = []
            for curve in growth_curves:
                if radius in curve:
                    corrections.append(1.0 / curve[radius])  # Correction factor
            
            if corrections:
                aperture_corrections[radius] = np.median(corrections)
            else:
                aperture_corrections[radius] = 1.0
        
        return aperture_corrections
    
    def _compute_psf_based_corrections(self, psf_model: np.ndarray) -> Dict[float, float]:
        """Compute aperture corrections using PSF model."""
        aperture_corrections = {}
        
        # Normalize PSF
        psf = psf_model / np.sum(psf_model)
        
        # PSF center
        center_y, center_x = (psf.shape[0] - 1) / 2, (psf.shape[1] - 1) / 2
        
        # Total flux (sum of entire PSF)
        total_flux = np.sum(psf)
        
        # Compute flux within each aperture
        for radius in self.config.circular_apertures:
            try:
                flux, _, _ = sep.sum_circle(
                    psf, [center_x], [center_y], [radius],
                    subpix=5
                )
                
                if flux[0] > 0:
                    aperture_corrections[radius] = total_flux / flux[0]
                else:
                    aperture_corrections[radius] = 1.0
                
            except:
                aperture_corrections[radius] = 1.0
        
        return aperture_corrections
    
    def _apply_aperture_corrections(self,
                                  sources: List[AperturePhotometrySource],
                                  corrections: Dict[float, float]) -> List[AperturePhotometrySource]:
        """Apply aperture corrections to sources."""
        for source in sources:
            for radius in source.circular_fluxes.keys():
                if radius in corrections:
                    correction = corrections[radius]
                    source.aperture_corrections[radius] = correction
                    
                    # Apply correction to largest aperture for "total" flux
                    if radius == max(self.config.circular_apertures):
                        source.total_flux = source.circular_fluxes[radius] * correction
                        if source.total_flux > 0:
                            source.total_magnitude = -2.5 * np.log10(source.total_flux) + 25.0
                        else:
                            source.total_magnitude = 99.0
        
        return sources
    
    def _compute_curve_of_growth(self, psf_model: np.ndarray) -> np.ndarray:
        """Compute curve of growth from PSF model."""
        try:
            # Normalize PSF
            psf = psf_model / np.sum(psf_model)
            center_y, center_x = (psf.shape[0] - 1) / 2, (psf.shape[1] - 1) / 2
            
            # Create radial profile
            max_radius = min(psf.shape) // 2 - 1
            radii = np.arange(0.5, max_radius, 0.5)
            
            curve_of_growth = []
            
            for radius in radii:
                try:
                    flux, _, _ = sep.sum_circle(
                        psf, [center_x], [center_y], [radius],
                        subpix=5
                    )
                    curve_of_growth.append(flux[0])
                except:
                    curve_of_growth.append(0.0)
            
            curve_of_growth = np.array(curve_of_growth)
            
            # Normalize to total flux
            if np.max(curve_of_growth) > 0:
                curve_of_growth = curve_of_growth / np.max(curve_of_growth)
            
            return curve_of_growth
            
        except:
            return np.array([])
    
    def _compute_statistics(self, sources: List[AperturePhotometrySource]) -> Dict[str, Any]:
        """Compute global statistics."""
        stats = {}
        
        # Basic counts
        stats['total_sources'] = len(sources)
        stats['successful_sources'] = len([s for s in sources if s.circular_fluxes])
        
        # Flag statistics
        all_flags = []
        for source in sources:
            all_flags.extend(source.flags)
        
        flag_counts = {}
        for flag in set(all_flags):
            flag_counts[flag] = all_flags.count(flag)
        
        stats['flag_statistics'] = flag_counts
        
        # Photometry statistics for each aperture
        aperture_stats = {}
        for radius in self.config.circular_apertures:
            fluxes = [s.circular_fluxes.get(radius, 0) for s in sources if radius in s.circular_fluxes]
            if fluxes:
                aperture_stats[f'radius_{radius}'] = {
                    'n_measurements': len(fluxes),
                    'flux_median': np.median(fluxes),
                    'flux_std': np.std(fluxes),
                    'positive_detections': len([f for f in fluxes if f > 0])
                }
        
        stats['aperture_statistics'] = aperture_stats
        
        return stats
    
    def _create_diagnostics(self, sources: List[AperturePhotometrySource], image: np.ndarray) -> Dict[str, Any]:
        """Create diagnostic information."""
        diagnostics = {
            'config': self.config,
            'image_shape': image.shape,
            'aperture_radii': self.config.circular_apertures,
            'n_sources_processed': len(sources)
        }
        
        return diagnostics
    
    def plot_photometry_diagnostics(self, results: AperturePhotometryResults,
                                   output_path: Optional[str] = None) -> None:
        """Create diagnostic plots for aperture photometry."""
        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            # Flux vs aperture size
            aperture_radii = self.config.circular_apertures
            median_fluxes = []
            
            for radius in aperture_radii:
                fluxes = [s.circular_fluxes.get(radius, 0) for s in results.sources if radius in s.circular_fluxes]
                if fluxes:
                    median_fluxes.append(np.median([f for f in fluxes if f > 0]))
                else:
                    median_fluxes.append(0)
            
            axes[0, 0].plot(aperture_radii, median_fluxes, 'bo-')
            axes[0, 0].set_xlabel('Aperture Radius (pixels)')
            axes[0, 0].set_ylabel('Median Flux')
            axes[0, 0].set_title('Growth Curve')
            axes[0, 0].set_yscale('log')
            
            # Aperture corrections
            if results.aperture_corrections:
                radii = list(results.aperture_corrections.keys())
                corrections = list(results.aperture_corrections.values())
                axes[0, 1].plot(radii, corrections, 'ro-')
                axes[0, 1].set_xlabel('Aperture Radius (pixels)')
                axes[0, 1].set_ylabel('Aperture Correction')
                axes[0, 1].set_title('Aperture Corrections')
            
            # Flag statistics
            if results.statistics.get('flag_statistics'):
                flags = list(results.statistics['flag_statistics'].keys())
                counts = list(results.statistics['flag_statistics'].values())
                axes[0, 2].bar(range(len(flags)), counts)
                axes[0, 2].set_xticks(range(len(flags)))
                axes[0, 2].set_xticklabels(flags, rotation=45, ha='right')
                axes[0, 2].set_ylabel('Count')
                axes[0, 2].set_title('Quality Flags')
            
            # Magnitude distribution
            largest_aperture = max(self.config.circular_apertures)
            magnitudes = [s.circular_magnitudes.get(largest_aperture, 99) 
                         for s in results.sources if largest_aperture in s.circular_magnitudes]
            valid_magnitudes = [m for m in magnitudes if m < 90]
            
            if valid_magnitudes:
                axes[1, 0].hist(valid_magnitudes, bins=30, alpha=0.7)
                axes[1, 0].set_xlabel('Magnitude')
                axes[1, 0].set_ylabel('Count')
                axes[1, 0].set_title(f'Magnitude Distribution (r={largest_aperture})')
            
            # Contamination statistics
            contamination_fractions = [s.contamination_fraction for s in results.sources]
            if contamination_fractions:
                axes[1, 1].hist(contamination_fractions, bins=30, alpha=0.7)
                axes[1, 1].set_xlabel('Contamination Fraction')
                axes[1, 1].set_ylabel('Count')
                axes[1, 1].set_title('Contamination Distribution')
            
            # Background statistics
            backgrounds = [s.local_background for s in results.sources if s.local_background is not None]
            if backgrounds:
                axes[1, 2].hist(backgrounds, bins=30, alpha=0.7)
                axes[1, 2].set_xlabel('Local Background')
                axes[1, 2].set_ylabel('Count')
                axes[1, 2].set_title('Background Distribution')
            
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                self.logger.info(f"Aperture photometry diagnostics saved to {output_path}")
            else:
                plt.show()
            
            plt.close()
            
        except Exception as e:
            self.logger.warning(f"Failed to create diagnostic plots: {e}")


# Legacy compatibility functions
def perform_aperture_photometry(image, sources, photometry_params):
    """
    Legacy function for aperture photometry.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image data
    sources : list
        List of detected sources
    photometry_params : dict
        Photometry parameters
        
    Returns:
    --------
    Table or None
        Aperture photometry results
    """
    try:
        # Convert legacy parameters
        config = AperturePhotometryConfig()
        
        if 'apertures' in photometry_params:
            config.circular_apertures = photometry_params['apertures']
        
        # Convert sources to structured array if needed
        if isinstance(sources, list):
            dtype = [('x', 'f4'), ('y', 'f4'), ('a', 'f4'), ('b', 'f4'), ('theta', 'f4')]
            sources_array = np.array([(s.get('x', 0), s.get('y', 0), s.get('a', 2), 
                                     s.get('b', 2), s.get('theta', 0))
                                    for s in sources], dtype=dtype)
        else:
            sources_array = sources
        
        # Perform photometry
        processor = EnhancedAperturePhotometry(config)
        results = processor.perform_aperture_photometry(image, sources_array)
        
        # Convert to legacy format (Table)
        return extract_photometry_table(results)
        
    except Exception as e:
        logging.error(f"Legacy aperture photometry failed: {e}")
        return None


def extract_photometry_table(results: AperturePhotometryResults) -> Table:
    """
    Extract photometry table from results.
    
    Parameters:
    -----------
    results : AperturePhotometryResults
        Photometry results
        
    Returns:
    --------
    Table
        Astropy table with photometry results
    """
    table_data = {}
    
    # Basic information
    table_data['id'] = [s.id for s in results.sources]
    table_data['x'] = [s.x for s in results.sources]
    table_data['y'] = [s.y for s in results.sources]
    
    if any(s.ra is not None for s in results.sources):
        table_data['ra'] = [s.ra if s.ra is not None else np.nan for s in results.sources]
        table_data['dec'] = [s.dec if s.dec is not None else np.nan for s in results.sources]
    
    # Add photometry for each aperture
    for radius in results.config.circular_apertures:
        table_data[f'flux_r{radius}'] = [s.circular_fluxes.get(radius, np.nan) for s in results.sources]
        table_data[f'flux_err_r{radius}'] = [s.circular_flux_errors.get(radius, np.nan) for s in results.sources]
        table_data[f'mag_r{radius}'] = [s.circular_magnitudes.get(radius, np.nan) for s in results.sources]
        table_data[f'mag_err_r{radius}'] = [s.circular_magnitude_errors.get(radius, np.nan) for s in results.sources]
    
    # Add Kron photometry
    table_data['kron_flux'] = [s.kron_flux if s.kron_flux is not None else np.nan for s in results.sources]
    table_data['kron_flux_err'] = [s.kron_flux_error if s.kron_flux_error is not None else np.nan for s in results.sources]
    table_data['kron_mag'] = [s.kron_magnitude if s.kron_magnitude is not None else np.nan for s in results.sources]
    table_data['kron_radius'] = [s.kron_radius if s.kron_radius is not None else np.nan for s in results.sources]
    
    # Quality information
    table_data['contamination_fraction'] = [s.contamination_fraction for s in results.sources]
    table_data['local_background'] = [s.local_background if s.local_background is not None else np.nan for s in results.sources]
    table_data['n_flags'] = [len(s.flags) for s in results.sources]
    
    return Table(table_data)


def correct_photometry(photometry_results, correction_params):
    """
    Legacy function for photometry correction.
    
    Parameters:
    -----------
    photometry_results : dict
        Photometry results for each band
    correction_params : dict
        Correction parameters
        
    Returns:
    --------
    dict
        Corrected photometry results
    """
    # This is a simplified legacy compatibility function
    # In practice, corrections would be handled by the calibration module
    return photometry_results


def derive_photometric_uncertainties(image, sources, correction_params):
    """
    Legacy function for uncertainty derivation.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image
    sources : list
        Source list
    correction_params : dict
        Correction parameters
        
    Returns:
    --------
    list
        Sources with uncertainties
    """
    # This is handled internally by the enhanced photometry processor
    return sources
