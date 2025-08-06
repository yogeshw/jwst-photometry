"""
Enhanced Source Detection Module for JWST Photometry

This module implements advanced source detection capabilities using SEP,
including multi-threshold detection, adaptive parameters, deblending optimization,
and source classification.
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple, List, Union
from dataclasses import dataclass, field
import warnings

import sep
from scipy import ndimage
from scipy.ndimage import binary_dilation, label, center_of_mass
from astropy.stats import sigma_clipped_stats
from astropy.convolution import Gaussian2DKernel, convolve
from astropy.table import Table
import matplotlib.pyplot as plt


@dataclass
class Source:
    """Generic source object for compatibility."""
    id: int
    x: float
    y: float
    flux: float
    a: float = 1.0
    b: float = 1.0
    theta: float = 0.0
    peak: float = None
    flag: int = 0
    
    def __post_init__(self):
        if self.peak is None:
            self.peak = self.flux


@dataclass
class DetectionConfig:
    """Configuration parameters for source detection."""
    
    # Basic detection parameters
    threshold: float = 1.5
    minarea: int = 5
    deblend_nthresh: int = 32
    deblend_cont: float = 0.005
    clean: bool = True
    clean_param: float = 1.0
    
    # Multi-threshold detection
    use_multi_threshold: bool = True
    threshold_range: Tuple[float, float] = (1.0, 5.0)
    threshold_steps: int = 5
    
    # Adaptive parameters
    use_adaptive_threshold: bool = True
    local_background_size: int = 50
    adaptive_factor: float = 1.5
    
    # Deblending optimization
    optimize_deblending: bool = True
    deblend_nthresh_range: Tuple[int, int] = (16, 64)
    deblend_cont_range: Tuple[float, float] = (0.001, 0.01)
    
    # Source classification
    classify_sources: bool = True
    star_size_limit: float = 2.0
    elongation_limit: float = 1.3
    flux_radius_limit: float = 3.0
    
    # Quality control
    edge_buffer: int = 10
    min_separation: float = 2.0
    max_ellipticity: float = 0.8
    snr_threshold: float = 5.0
    
    # Spurious source filtering
    filter_spurious: bool = True
    spurious_size_limit: float = 0.5
    spurious_flux_limit: float = 0.0
    neighbor_contamination_limit: float = 0.3


@dataclass
class DetectionResults:
    """Container for detection results."""
    
    sources: np.ndarray
    background: Any  # SEP Background object
    segmentation_map: Optional[np.ndarray] = None
    detection_image: Optional[np.ndarray] = None
    threshold_map: Optional[np.ndarray] = None
    statistics: Dict[str, Any] = field(default_factory=dict)
    quality_flags: Optional[np.ndarray] = None


class AdvancedSourceDetector:
    """
    Advanced source detection using SEP with optimization and classification.
    
    This class provides sophisticated source detection capabilities including:
    - Multi-threshold detection for completeness optimization
    - Adaptive detection parameters based on local noise
    - Deblending optimization for crowded fields
    - Source classification and quality assessment
    - Spurious source filtering
    """
    
    def __init__(self, config: Optional[DetectionConfig] = None):
        """
        Initialize the source detector.
        
        Parameters:
        -----------
        config : DetectionConfig, optional
            Detection configuration. If None, uses defaults.
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or DetectionConfig()
    
    def detect_sources(self, 
                      image: np.ndarray,
                      background_map: Optional[np.ndarray] = None,
                      rms_map: Optional[np.ndarray] = None,
                      mask: Optional[np.ndarray] = None,
                      weight: Optional[np.ndarray] = None) -> DetectionResults:
        """
        Perform comprehensive source detection.
        
        Parameters:
        -----------
        image : numpy.ndarray
            Input image for detection
        background_map : numpy.ndarray, optional
            Pre-computed background map
        rms_map : numpy.ndarray, optional
            Pre-computed RMS map
        mask : numpy.ndarray, optional
            Bad pixel mask
        weight : numpy.ndarray, optional
            Weight map
            
        Returns:
        --------
        DetectionResults
            Comprehensive detection results
        """
        self.logger.info("Starting advanced source detection")
        
        # Validate inputs
        self._validate_inputs(image, background_map, rms_map, mask, weight)
        
        # Prepare detection image
        detection_image, background, threshold_map = self._prepare_detection_image(
            image, background_map, rms_map, mask
        )
        
        # Perform detection with optimization
        if self.config.use_multi_threshold:
            sources = self._multi_threshold_detection(
                detection_image, threshold_map, mask
            )
        else:
            sources = self._single_threshold_detection(
                detection_image, threshold_map, mask
            )
        
        # Ensure sources is a proper array
        if sources is None:
            sources = np.array([])
        elif not hasattr(sources, '__len__'):
            self.logger.warning(f"Unexpected sources type: {type(sources)}")
            sources = np.array([])
        elif not isinstance(sources, np.ndarray):
            sources = np.array(sources)
        
        # Optimize deblending if requested
        if self.config.optimize_deblending:
            sources = self._optimize_deblending(
                detection_image, threshold_map, mask, sources
            )
        
        # Generate segmentation map
        segmentation_map = self._generate_segmentation_map(
            detection_image, sources
        )
        
        # Classify sources
        if self.config.classify_sources:
            sources = self._classify_sources(sources, detection_image)
        
        # Apply quality filtering
        sources, quality_flags = self._apply_quality_filters(
            sources, detection_image, threshold_map
        )
        
        # Filter spurious detections
        if self.config.filter_spurious:
            sources, quality_flags = self._filter_spurious_sources(
                sources, quality_flags, segmentation_map
            )
        
        # Compute detection statistics
        statistics = self._compute_detection_statistics(
            sources, detection_image, background, segmentation_map
        )
        
        # Create results object
        results = DetectionResults(
            sources=sources,
            background=background,
            segmentation_map=segmentation_map,
            detection_image=detection_image,
            threshold_map=threshold_map,
            statistics=statistics,
            quality_flags=quality_flags
        )
        
        self.logger.info(f"Detection completed - {len(sources)} sources found")
        return results
    
    def _validate_inputs(self, image: np.ndarray, 
                        background_map: Optional[np.ndarray],
                        rms_map: Optional[np.ndarray],
                        mask: Optional[np.ndarray],
                        weight: Optional[np.ndarray]) -> None:
        """Validate input arrays."""
        if image.ndim != 2:
            raise ValueError(f"Image must be 2D, got {image.ndim}D")
        
        arrays_to_check = [
            (background_map, "background_map"),
            (rms_map, "rms_map"), 
            (mask, "mask"),
            (weight, "weight")
        ]
        
        for array, name in arrays_to_check:
            if array is not None and array.shape != image.shape:
                raise ValueError(f"{name} shape {array.shape} doesn't match image shape {image.shape}")
    
    def _prepare_detection_image(self, 
                               image: np.ndarray,
                               background_map: Optional[np.ndarray],
                               rms_map: Optional[np.ndarray],
                               mask: Optional[np.ndarray]) -> Tuple[np.ndarray, Any, np.ndarray]:
        """
        Prepare the detection image and threshold map.
        
        Returns:
        --------
        tuple
            Detection image, background object, threshold map
        """
        # Estimate background if not provided
        if background_map is None or rms_map is None:
            self.logger.debug("Computing background for detection")
            
            # Use basic SEP background if no background provided
            sep_mask = mask.astype(np.uint8) if mask is not None else None
            
            try:
                background = sep.Background(image, mask=sep_mask)
                if background_map is None:
                    background_map = background.back()
                if rms_map is None:
                    rms_map = background.rms()
            except Exception as e:
                self.logger.warning(f"SEP background failed: {e}, using fallback")
                background = None
                if background_map is None:
                    background_map = np.zeros_like(image)
                if rms_map is None:
                    rms_map = np.ones_like(image) * np.nanstd(image)
        else:
            # Create dummy background object for compatibility
            background = type('MockBackground', (), {
                'back': lambda: background_map,
                'rms': lambda: rms_map,
                'globalrms': np.median(rms_map)
            })()
        
        # Create detection image (background subtracted)
        detection_image = image - background_map
        
        # Create threshold map
        if self.config.use_adaptive_threshold:
            threshold_map = self._create_adaptive_threshold_map(
                detection_image, rms_map, mask
            )
        else:
            threshold_map = self.config.threshold * rms_map
        
        return detection_image, background, threshold_map
    
    def _create_adaptive_threshold_map(self, 
                                     detection_image: np.ndarray,
                                     rms_map: np.ndarray,
                                     mask: Optional[np.ndarray]) -> np.ndarray:
        """
        Create adaptive threshold map based on local noise properties.
        
        Parameters:
        -----------
        detection_image : numpy.ndarray
            Background-subtracted image
        rms_map : numpy.ndarray
            RMS map
        mask : numpy.ndarray, optional
            Bad pixel mask
            
        Returns:
        --------
        numpy.ndarray
            Adaptive threshold map
        """
        self.logger.debug("Creating adaptive threshold map")
        
        # Start with basic threshold
        threshold_map = self.config.threshold * rms_map
        
        # Compute local statistics
        box_size = self.config.local_background_size
        
        # Create local standard deviation map
        from scipy.ndimage import uniform_filter
        
        # Local variance using convolution
        local_mean = uniform_filter(detection_image, size=box_size)
        local_var = uniform_filter(detection_image**2, size=box_size) - local_mean**2
        local_std = np.sqrt(np.maximum(local_var, 0))
        
        # Adapt threshold based on local noise
        # Increase threshold in high-noise regions
        noise_factor = local_std / np.median(rms_map)
        adaptive_factor = 1.0 + self.config.adaptive_factor * (noise_factor - 1.0)
        
        # Apply adaptation
        threshold_map *= adaptive_factor
        
        # Ensure minimum threshold
        min_threshold = 0.5 * self.config.threshold * rms_map
        threshold_map = np.maximum(threshold_map, min_threshold)
        
        return threshold_map
    
    def _multi_threshold_detection(self, 
                                 detection_image: np.ndarray,
                                 threshold_map: np.ndarray,
                                 mask: Optional[np.ndarray]) -> np.ndarray:
        """
        Perform multi-threshold detection for completeness optimization.
        
        Parameters:
        -----------
        detection_image : numpy.ndarray
            Detection image
        threshold_map : numpy.ndarray
            Threshold map
        mask : numpy.ndarray, optional
            Bad pixel mask
            
        Returns:
        --------
        numpy.ndarray
            Combined source catalog
        """
        self.logger.debug("Performing multi-threshold detection")
        
        # Generate threshold values
        thresh_min, thresh_max = self.config.threshold_range
        thresholds = np.linspace(thresh_min, thresh_max, self.config.threshold_steps)
        
        all_sources = []
        
        for i, thresh_factor in enumerate(thresholds):
            self.logger.debug(f"  Threshold {i+1}/{len(thresholds)}: {thresh_factor:.2f}")
            
            # Scale threshold map
            if np.isscalar(threshold_map):
                current_threshold = thresh_factor * threshold_map
            else:
                # Use the median threshold for scalar threshold
                current_threshold = thresh_factor * np.median(threshold_map)
            
            try:
                # Detect sources at this threshold
                sources = sep.extract(
                    detection_image,
                    thresh=current_threshold,
                    minarea=self.config.minarea,
                    deblend_nthresh=self.config.deblend_nthresh,
                    deblend_cont=self.config.deblend_cont,
                    clean=self.config.clean,
                    clean_param=self.config.clean_param,
                    mask=mask.astype(np.uint8) if mask is not None else None
                )
                
                # Add threshold information
                if len(sources) > 0:
                    # Add detection threshold as a field
                    sources = Table(sources)
                    sources['detection_threshold'] = thresh_factor
                    all_sources.append(sources)
                
            except Exception as e:
                self.logger.warning(f"Detection failed at threshold {thresh_factor}: {e}")
                continue
        
        if not all_sources:
            self.logger.warning("No sources detected at any threshold")
            return np.array([], dtype=sep.extract(detection_image, 1e10, minarea=1).dtype)
        
        # Combine and deduplicate sources
        combined_sources = self._combine_multi_threshold_catalogs(all_sources)
        
        return combined_sources
    
    def _single_threshold_detection(self, 
                                  detection_image: np.ndarray,
                                  threshold_map: np.ndarray,
                                  mask: Optional[np.ndarray]) -> np.ndarray:
        """
        Perform single-threshold detection.
        
        Parameters:
        -----------
        detection_image : numpy.ndarray
            Detection image
        threshold_map : numpy.ndarray
            Threshold map
        mask : numpy.ndarray, optional
            Bad pixel mask
            
        Returns:
        --------
        numpy.ndarray
            Source catalog
        """
        self.logger.debug("Performing single-threshold detection")
        
        try:
            # Handle threshold properly
            if np.isscalar(threshold_map):
                thresh_value = threshold_map
            else:
                thresh_value = np.median(threshold_map)
            
            sources = sep.extract(
                detection_image,
                thresh=thresh_value,
                minarea=self.config.minarea,
                deblend_nthresh=self.config.deblend_nthresh,
                deblend_cont=self.config.deblend_cont,
                clean=self.config.clean,
                clean_param=self.config.clean_param,
                mask=mask.astype(np.uint8) if mask is not None else None
            )
            
            return sources
            
        except Exception as e:
            self.logger.error(f"Source detection failed: {e}")
            # Return empty catalog with correct dtype
            return np.array([], dtype=[
                ('x', 'f8'), ('y', 'f8'), ('flux', 'f8'), ('a', 'f8'), ('b', 'f8'),
                ('theta', 'f8'), ('cxx', 'f8'), ('cyy', 'f8'), ('cxy', 'f8')
            ])
    
    def _combine_multi_threshold_catalogs(self, source_lists: List[Table]) -> np.ndarray:
        """
        Combine and deduplicate multi-threshold source catalogs.
        
        Parameters:
        -----------
        source_lists : list of Table
            Source catalogs from different thresholds
            
        Returns:
        --------
        numpy.ndarray
            Combined, deduplicated catalog
        """
        # Stack all sources
        all_sources = Table()
        for sources in source_lists:
            if len(all_sources) == 0:
                all_sources = sources.copy()
            else:
                all_sources = Table(np.concatenate([all_sources, sources]))
        
        if len(all_sources) == 0:
            return np.array([], dtype=source_lists[0].dtype)
        
        # Sort by detection threshold (highest first for prioritization)
        all_sources.sort('detection_threshold', reverse=True)
        
        # Deduplicate based on position
        min_sep = self.config.min_separation
        
        unique_sources = []
        used_positions = []
        
        for source in all_sources:
            x, y = source['x'], source['y']
            
            # Check if too close to existing source
            too_close = False
            for used_x, used_y in used_positions:
                separation = np.sqrt((x - used_x)**2 + (y - used_y)**2)
                if separation < min_sep:
                    too_close = True
                    break
            
            if not too_close:
                unique_sources.append(source)
                used_positions.append((x, y))
        
        # Convert back to structured array
        if unique_sources:
            # Convert Source objects back to structured array format
            if hasattr(unique_sources[0], '__dict__'):
                # Convert Source dataclass objects to dictionary
                source_dicts = []
                for source in unique_sources:
                    source_dict = {}
                    for field in ['x', 'y', 'flux', 'a', 'b', 'theta', 'peak', 'flag']:
                        source_dict[field] = getattr(source, field, 0.0)
                    source_dicts.append(source_dict)
                unique_table = Table(source_dicts)
            else:
                unique_table = Table(unique_sources)
            
            # Remove the detection_threshold column for output if it exists
            if 'detection_threshold' in unique_table.colnames:
                unique_table.remove_column('detection_threshold')
            return np.array(unique_table)
        else:
            return np.array([], dtype=source_lists[0].dtype)
    
    def _optimize_deblending(self, 
                           detection_image: np.ndarray,
                           threshold_map: np.ndarray,
                           mask: Optional[np.ndarray],
                           initial_sources: np.ndarray) -> np.ndarray:
        """
        Optimize deblending parameters for crowded fields.
        
        Parameters:
        -----------
        detection_image : numpy.ndarray
            Detection image
        threshold_map : numpy.ndarray
            Threshold map
        mask : numpy.ndarray, optional
            Bad pixel mask
        initial_sources : numpy.ndarray
            Initial source catalog
            
        Returns:
        --------
        numpy.ndarray
            Optimized source catalog
        """
        self.logger.debug("Optimizing deblending parameters")
        
        if len(initial_sources) == 0:
            return initial_sources
        
        # Test different deblending parameters
        nthresh_min, nthresh_max = self.config.deblend_nthresh_range
        cont_min, cont_max = self.config.deblend_cont_range
        
        # Simple grid search (could be made more sophisticated)
        nthresh_values = [16, 32, 64]
        cont_values = [0.001, 0.005, 0.01]
        
        best_sources = initial_sources
        best_score = self._evaluate_deblending_quality(initial_sources, detection_image)
        
        for nthresh in nthresh_values:
            for cont in cont_values:
                try:
                    test_sources = sep.extract(
                        detection_image,
                        thresh=threshold_map,
                        minarea=self.config.minarea,
                        deblend_nthresh=nthresh,
                        deblend_cont=cont,
                        clean=self.config.clean,
                        clean_param=self.config.clean_param,
                        mask=mask.astype(np.uint8) if mask is not None else None
                    )
                    
                    score = self._evaluate_deblending_quality(test_sources, detection_image)
                    
                    if score > best_score:
                        best_sources = test_sources
                        best_score = score
                        self.logger.debug(f"  Improved deblending: nthresh={nthresh}, cont={cont}, score={score:.3f}")
                
                except Exception:
                    continue
        
        return best_sources
    
    def _evaluate_deblending_quality(self, sources: np.ndarray, image: np.ndarray) -> float:
        """
        Evaluate the quality of deblending based on source properties.
        
        Parameters:
        -----------
        sources : numpy.ndarray
            Source catalog
        image : numpy.ndarray
            Detection image
            
        Returns:
        --------
        float
            Quality score (higher is better)
        """
        if len(sources) == 0:
            return 0.0
        
        # Metrics for deblending quality:
        # 1. Reasonable source sizes (not too small/large)
        # 2. Reasonable ellipticities
        # 3. Flux concentration
        
        score = 0.0
        
        # Size distribution score
        sizes = np.sqrt(sources['a'] * sources['b'])
        reasonable_size_fraction = np.sum((sizes > 1.0) & (sizes < 10.0)) / len(sources)
        score += reasonable_size_fraction
        
        # Ellipticity score
        ellipticities = 1 - sources['b'] / sources['a']
        reasonable_ellip_fraction = np.sum(ellipticities < 0.8) / len(sources)
        score += reasonable_ellip_fraction
        
        # Detection significance score (higher flux = better detection)
        if 'flux' in sources.dtype.names:
            flux_score = np.median(np.log10(np.maximum(sources['flux'], 1e-10)))
            score += flux_score / 10.0  # Normalize
        
        return score

    
    def _generate_segmentation_map(self, detection_image: np.ndarray, sources: np.ndarray) -> np.ndarray:
        """
        Generate enhanced segmentation map from detected sources.
        
        Parameters:
        -----------
        detection_image : numpy.ndarray
            Detection image
        sources : numpy.ndarray
            Source catalog
            
        Returns:
        --------
        numpy.ndarray
            Segmentation map with source IDs
        """
        if len(sources) == 0:
            return np.zeros_like(detection_image, dtype=np.int32)
        
        self.logger.debug(f"Generating segmentation map for {len(sources)} sources")
        
        segmentation_map = np.zeros_like(detection_image, dtype=np.int32)
        
        # Create more accurate segmentation using source ellipses
        y_coords, x_coords = np.ogrid[:detection_image.shape[0], :detection_image.shape[1]]
        
        for i, source in enumerate(sources):
            source_id = i + 1
            
            # Source parameters
            x0, y0 = source['x'], source['y']
            a, b = source['a'], source['b']
            theta = source['theta']
            
            # Create elliptical mask
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            
            dx = x_coords - x0
            dy = y_coords - y0
            
            # Rotate coordinates
            x_rot = dx * cos_theta + dy * sin_theta
            y_rot = -dx * sin_theta + dy * cos_theta
            
            # Ellipse equation with margin for better segmentation
            ellipse_mask = (x_rot / (2.5 * a))**2 + (y_rot / (2.5 * b))**2 <= 1
            
            # Only assign to segmentation map if not already assigned to a brighter source
            # (sources are sorted by flux in SEP output)
            new_pixels = ellipse_mask & (segmentation_map == 0)
            segmentation_map[new_pixels] = source_id
        
        return segmentation_map
    
    def _classify_sources(self, sources: np.ndarray, detection_image: np.ndarray) -> np.ndarray:
        """
        Classify sources as stars, galaxies, or artifacts.
        
        Parameters:
        -----------
        sources : numpy.ndarray
            Source catalog
        detection_image : numpy.ndarray
            Detection image
            
        Returns:
        --------
        numpy.ndarray
            Source catalog with classification added
        """
        if len(sources) == 0:
            return sources
        
        self.logger.debug("Classifying sources")
        
        # Calculate classification metrics
        sizes = np.sqrt(sources['a'] * sources['b'])
        elongations = sources['a'] / sources['b']
        
        # Compute flux radius (half-light radius approximation)
        flux_radii = self._compute_flux_radii(sources, detection_image)
        
        # Classification logic
        # 0 = artifact, 1 = star, 2 = galaxy
        classifications = np.zeros(len(sources), dtype=int)
        
        # Star criteria: compact, round, concentrated
        star_mask = (
            (sizes < self.config.star_size_limit) &
            (elongations < self.config.elongation_limit) &
            (flux_radii < self.config.flux_radius_limit)
        )
        classifications[star_mask] = 1
        
        # Galaxy criteria: extended or elongated
        galaxy_mask = (
            (sizes >= self.config.star_size_limit) |
            (elongations >= self.config.elongation_limit) |
            (flux_radii >= self.config.flux_radius_limit)
        ) & ~star_mask
        classifications[galaxy_mask] = 2
        
        # Add classification to source catalog
        # Convert to astropy Table for easier manipulation
        source_table = Table(sources)
        source_table['class'] = classifications
        source_table['size'] = sizes
        source_table['elongation'] = elongations
        source_table['flux_radius'] = flux_radii
        
        # Convert back to structured array
        return np.array(source_table)
    
    def _compute_flux_radii(self, sources: np.ndarray, image: np.ndarray) -> np.ndarray:
        """
        Compute flux radii (half-light radii) for sources.
        
        Parameters:
        -----------
        sources : numpy.ndarray
            Source catalog
        image : numpy.ndarray
            Detection image
            
        Returns:
        --------
        numpy.ndarray
            Flux radii in pixels
        """
        flux_radii = np.zeros(len(sources))
        
        for i, source in enumerate(sources):
            try:
                # Get source position and size
                x, y = source['x'], source['y']
                a, b = max(source['a'], 1.0), max(source['b'], 1.0)
                
                # Create apertures at increasing radii
                max_radius = min(3 * max(a, b), 20.0)  # Reasonable upper limit
                radii = np.linspace(0.5, max_radius, 20)
                
                # Measure flux in apertures
                fluxes = []
                for radius in radii:
                    try:
                        flux, _, _ = sep.sum_circle(
                            image, [x], [y], radius, 
                            mask=None, err=None
                        )
                        fluxes.append(flux[0])
                    except:
                        fluxes.append(0.0)
                
                fluxes = np.array(fluxes)
                
                # Find half-light radius
                if len(fluxes) > 0 and np.max(fluxes) > 0:
                    total_flux = np.max(fluxes)
                    half_flux = total_flux / 2.0
                    
                    # Interpolate to find radius containing half the flux
                    valid_fluxes = fluxes[fluxes > 0]
                    valid_radii = radii[:len(valid_fluxes)]
                    
                    if len(valid_fluxes) > 1:
                        flux_radius = np.interp(half_flux, valid_fluxes, valid_radii)
                        flux_radii[i] = flux_radius
                    else:
                        flux_radii[i] = radii[0]  # Default to minimum radius
                else:
                    flux_radii[i] = np.sqrt(a * b)  # Fall back to geometric mean
                    
            except Exception:
                # Fall back to geometric mean of semi-axes
                flux_radii[i] = np.sqrt(source['a'] * source['b'])
        
        return flux_radii
    
    def _apply_quality_filters(self, 
                             sources: np.ndarray,
                             detection_image: np.ndarray,
                             threshold_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply quality filters to remove problematic sources.
        
        Parameters:
        -----------
        sources : numpy.ndarray
            Source catalog
        detection_image : numpy.ndarray
            Detection image
        threshold_map : numpy.ndarray
            Detection threshold map
            
        Returns:
        --------
        tuple
            Filtered sources and quality flags
        """
        if len(sources) == 0:
            return sources, np.array([])
        
        self.logger.debug("Applying quality filters")
        
        # Initialize quality flags (0 = good, >0 = problematic)
        quality_flags = np.zeros(len(sources), dtype=int)
        
        # Filter 1: Edge sources
        ny, nx = detection_image.shape
        buffer = self.config.edge_buffer
        
        edge_mask = (
            (sources['x'] < buffer) | (sources['x'] > nx - buffer) |
            (sources['y'] < buffer) | (sources['y'] > ny - buffer)
        )
        quality_flags[edge_mask] |= 1  # Edge flag
        
        # Filter 2: Excessive ellipticity
        ellipticities = 1 - sources['b'] / sources['a']
        high_ellip_mask = ellipticities > self.config.max_ellipticity
        quality_flags[high_ellip_mask] |= 2  # High ellipticity flag
        
        # Filter 3: Signal-to-noise ratio
        if 'flux' in sources.dtype.names:
            # Estimate noise from threshold map
            x_int = np.round(sources['x']).astype(int)
            y_int = np.round(sources['y']).astype(int)
            
            # Ensure indices are within bounds
            x_int = np.clip(x_int, 0, nx - 1)
            y_int = np.clip(y_int, 0, ny - 1)
            
            local_noise = threshold_map[y_int, x_int] / self.config.threshold
            snr = sources['flux'] / (local_noise * np.sqrt(np.pi * sources['a'] * sources['b']))
            
            low_snr_mask = snr < self.config.snr_threshold
            quality_flags[low_snr_mask] |= 4  # Low SNR flag
        
        # Filter 4: Unreasonably small sources
        sizes = np.sqrt(sources['a'] * sources['b'])
        tiny_mask = sizes < 0.5  # Smaller than 0.5 pixels
        quality_flags[tiny_mask] |= 8  # Tiny source flag
        
        # Create good source mask
        good_mask = quality_flags == 0
        
        n_filtered = len(sources) - np.sum(good_mask)
        self.logger.info(f"Quality filters removed {n_filtered} sources ({n_filtered/len(sources):.1%})")
        
        return sources[good_mask], quality_flags[good_mask]
    
    def _filter_spurious_sources(self, 
                               sources: np.ndarray,
                               quality_flags: np.ndarray,
                               segmentation_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter spurious sources based on additional criteria.
        
        Parameters:
        -----------
        sources : numpy.ndarray
            Source catalog
        quality_flags : numpy.ndarray
            Quality flags
        segmentation_map : numpy.ndarray
            Segmentation map
            
        Returns:
        --------
        tuple
            Filtered sources and updated quality flags
        """
        if len(sources) == 0:
            return sources, quality_flags
        
        self.logger.debug("Filtering spurious sources")
        
        # Filter very small sources
        sizes = np.sqrt(sources['a'] * sources['b'])
        size_mask = sizes >= self.config.spurious_size_limit
        
        # Filter very faint sources (if flux information available)
        flux_mask = np.ones(len(sources), dtype=bool)
        if 'flux' in sources.dtype.names:
            flux_mask = sources['flux'] > self.config.spurious_flux_limit
        
        # Check for contamination by neighbors
        neighbor_mask = self._check_neighbor_contamination(sources, segmentation_map)
        
        # Combine all filters
        good_mask = size_mask & flux_mask & neighbor_mask
        
        n_spurious = len(sources) - np.sum(good_mask)
        self.logger.info(f"Spurious source filter removed {n_spurious} sources")
        
        return sources[good_mask], quality_flags[good_mask]
    
    def _check_neighbor_contamination(self, 
                                    sources: np.ndarray,
                                    segmentation_map: np.ndarray) -> np.ndarray:
        """
        Check for contamination by neighboring sources.
        
        Parameters:
        -----------
        sources : numpy.ndarray
            Source catalog
        segmentation_map : numpy.ndarray
            Segmentation map
            
        Returns:
        --------
        numpy.ndarray
            Boolean mask of non-contaminated sources
        """
        good_mask = np.ones(len(sources), dtype=bool)
        
        for i, source in enumerate(sources):
            try:
                # Get source region
                x, y = int(source['x']), int(source['y'])
                a, b = max(int(source['a'] * 3), 5), max(int(source['b'] * 3), 5)
                
                # Define region around source
                y1, y2 = max(0, y - b), min(segmentation_map.shape[0], y + b + 1)
                x1, x2 = max(0, x - a), min(segmentation_map.shape[1], x + a + 1)
                
                region = segmentation_map[y1:y2, x1:x2]
                
                # Count different source IDs in region
                unique_ids = np.unique(region)
                unique_ids = unique_ids[unique_ids > 0]  # Exclude background
                
                # If too many sources in region, mark as contaminated
                if len(unique_ids) > 3:  # Source itself + 2 neighbors max
                    good_mask[i] = False
                    
            except Exception:
                # If anything fails, keep the source
                continue
        
        return good_mask
    
    def _compute_detection_statistics(self, 
                                    sources: np.ndarray,
                                    detection_image: np.ndarray,
                                    background: Any,
                                    segmentation_map: np.ndarray) -> Dict[str, Any]:
        """
        Compute comprehensive detection statistics.
        
        Parameters:
        -----------
        sources : numpy.ndarray
            Final source catalog
        detection_image : numpy.ndarray
            Detection image
        background : Background object
            SEP background object
        segmentation_map : numpy.ndarray
            Segmentation map
            
        Returns:
        --------
        dict
            Detection statistics
        """
        stats = {}
        
        # Basic counts
        stats['n_sources'] = len(sources)
        stats['source_density'] = len(sources) / (detection_image.size / 1e6)  # per megapixel
        
        # Background statistics
        if hasattr(background, 'globalrms'):
            stats['global_rms'] = background.globalrms
        else:
            stats['global_rms'] = np.nanstd(detection_image)
        
        # Source property statistics
        if len(sources) > 0:
            if 'flux' in sources.dtype.names:
                stats['flux_range'] = (np.min(sources['flux']), np.max(sources['flux']))
                stats['median_flux'] = np.median(sources['flux'])
            
            sizes = np.sqrt(sources['a'] * sources['b'])
            stats['size_range'] = (np.min(sizes), np.max(sizes))
            stats['median_size'] = np.median(sizes)
            
            ellipticities = 1 - sources['b'] / sources['a']
            stats['ellipticity_range'] = (np.min(ellipticities), np.max(ellipticities))
            stats['median_ellipticity'] = np.median(ellipticities)
            
            # Classification statistics
            if 'class' in sources.dtype.names:
                unique_classes, counts = np.unique(sources['class'], return_counts=True)
                stats['source_classes'] = dict(zip(unique_classes, counts))
        
        # Segmentation statistics
        segmented_pixels = np.sum(segmentation_map > 0)
        stats['segmented_fraction'] = segmented_pixels / segmentation_map.size
        
        return stats
    
    def plot_detection_diagnostics(self, 
                                 results: DetectionResults,
                                 output_path: Optional[str] = None) -> None:
        """
        Create diagnostic plots for source detection.
        
        Parameters:
        -----------
        results : DetectionResults
            Detection results
        output_path : str, optional
            Path to save the plot
        """
        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            # Detection image
            im1 = axes[0, 0].imshow(results.detection_image, origin='lower', cmap='viridis')
            axes[0, 0].set_title(f'Detection Image\\n{len(results.sources)} sources')
            plt.colorbar(im1, ax=axes[0, 0])
            
            # Overlay source positions
            if len(results.sources) > 0:
                axes[0, 0].scatter(results.sources['x'], results.sources['y'], 
                                 c='red', s=20, alpha=0.7, marker='+')
            
            # Threshold map
            im2 = axes[0, 1].imshow(results.threshold_map, origin='lower', cmap='plasma')
            axes[0, 1].set_title('Threshold Map')
            plt.colorbar(im2, ax=axes[0, 1])
            
            # Segmentation map
            im3 = axes[0, 2].imshow(results.segmentation_map, origin='lower', cmap='tab20')
            axes[0, 2].set_title('Segmentation Map')
            
            # Source size distribution
            if len(results.sources) > 0 and 'size' in results.sources.dtype.names:
                axes[1, 0].hist(results.sources['size'], bins=30, alpha=0.7)
                axes[1, 0].set_xlabel('Source Size (pixels)')
                axes[1, 0].set_ylabel('Count')
                axes[1, 0].set_title('Size Distribution')
            
            # Flux distribution
            if len(results.sources) > 0 and 'flux' in results.sources.dtype.names:
                fluxes = results.sources['flux']
                fluxes = fluxes[fluxes > 0]
                if len(fluxes) > 0:
                    axes[1, 1].hist(np.log10(fluxes), bins=30, alpha=0.7)
                    axes[1, 1].set_xlabel('log10(Flux)')
                    axes[1, 1].set_ylabel('Count')
                    axes[1, 1].set_title('Flux Distribution')
            
            # Source classification
            if len(results.sources) > 0 and 'class' in results.sources.dtype.names:
                class_names = ['Artifact', 'Star', 'Galaxy']
                unique_classes, counts = np.unique(results.sources['class'], return_counts=True)
                
                # Create bar plot
                x_pos = np.arange(len(unique_classes))
                axes[1, 2].bar(x_pos, counts, alpha=0.7)
                axes[1, 2].set_xlabel('Source Class')
                axes[1, 2].set_ylabel('Count')
                axes[1, 2].set_title('Source Classification')
                axes[1, 2].set_xticks(x_pos)
                axes[1, 2].set_xticklabels([class_names[c] for c in unique_classes])
            
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                self.logger.info(f"Detection diagnostics saved to {output_path}")
            else:
                plt.show()
            
            plt.close()
            
        except Exception as e:
            self.logger.warning(f"Failed to create diagnostic plots: {e}")


# Legacy function compatibility
def detect_sources(image: Union[np.ndarray, Any], 
                  detection_params: Dict[str, Any]) -> Tuple[np.ndarray, Any]:
    """
    Legacy function for backward compatibility.
    
    Parameters:
    -----------
    image : numpy.ndarray or HDUList
        Input image data
    detection_params : dict
        Detection parameters
        
    Returns:
    --------
    tuple
        Sources and background object
    """
    warnings.warn("detect_sources function is deprecated. Use AdvancedSourceDetector class.", 
                  DeprecationWarning, stacklevel=2)
    
    # Extract image data if HDUList provided
    if hasattr(image, 'data'):
        image_data = image.data
    else:
        image_data = image
    
    # Create basic configuration from legacy parameters
    config = DetectionConfig(
        threshold=detection_params.get('thresh', 1.5),
        minarea=detection_params.get('minarea', 5),
        deblend_nthresh=detection_params.get('deblend_nthresh', 32),
        deblend_cont=detection_params.get('deblend_cont', 0.005),
        clean=detection_params.get('clean', True)
    )
    
    # Use advanced detector
    detector = AdvancedSourceDetector(config)
    results = detector.detect_sources(image_data)
    
    return results.sources, results.background


def generate_segmentation_map(image: np.ndarray, sources: np.ndarray) -> np.ndarray:
    """
    Legacy function for generating segmentation maps.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image
    sources : numpy.ndarray
        Source catalog
        
    Returns:
    --------
    numpy.ndarray
        Segmentation map
    """
    warnings.warn("generate_segmentation_map function is deprecated. Use AdvancedSourceDetector class.", 
                  DeprecationWarning, stacklevel=2)
    
    detector = AdvancedSourceDetector()
    return detector._generate_segmentation_map(image, sources)


def identify_star_candidates(sources: np.ndarray, 
                           mag_limit: float = 24, 
                           size_limit: float = 2.8) -> List[Any]:
    """
    Legacy function for identifying star candidates.
    
    Parameters:
    -----------
    sources : numpy.ndarray
        Source catalog
    mag_limit : float, default=24
        Magnitude limit for stars
    size_limit : float, default=2.8
        Size limit for stars
        
    Returns:
    --------
    list
        Star candidates
    """
    warnings.warn("identify_star_candidates function is deprecated. Use AdvancedSourceDetector classification.", 
                  DeprecationWarning, stacklevel=2)
    
    if len(sources) == 0:
        return []
    
    # Simple star selection based on size
    sizes = np.sqrt(sources['a'] * sources['b'])
    star_mask = sizes < size_limit
    
    # Also check flux if magnitude information available
    if 'mag' in sources.dtype.names:
        star_mask &= sources['mag'] < mag_limit
    
    return sources[star_mask].tolist()


def identify_star_candidates_advanced(sources: np.ndarray, 
                                    f200w_image: Optional[np.ndarray] = None,
                                    f160w_image: Optional[np.ndarray] = None,
                                    mag_limit: float = 25.0,
                                    size_ratio_limits: Tuple[float, float] = (1.5, 1.65),
                                    f160w_mag_limit: float = 23.0) -> List[Any]:
    """
    Legacy function for advanced star candidate identification.
    
    Parameters:
    -----------
    sources : numpy.ndarray
        Source catalog
    f200w_image : numpy.ndarray, optional
        F200W image data
    f160w_image : numpy.ndarray, optional
        F160W image data
    mag_limit : float, default=25.0
        Magnitude limit
    size_ratio_limits : tuple, default=(1.5, 1.65)
        Size ratio limits
    f160w_mag_limit : float, default=23.0
        F160W magnitude limit
        
    Returns:
    --------
    list
        Star candidates
    """
    warnings.warn("identify_star_candidates_advanced function is deprecated. Use AdvancedSourceDetector classification.", 
                  DeprecationWarning, stacklevel=2)
    
    # Use the enhanced classification from AdvancedSourceDetector
    detector = AdvancedSourceDetector()
    
    # Set up configuration for star selection
    detector.config.star_size_limit = size_ratio_limits[1]
    detector.config.elongation_limit = 1.3
    
    # Classify sources (using dummy image if needed)
    if f200w_image is not None:
        classified_sources = detector._classify_sources(sources, f200w_image)
    else:
        # Create dummy image for classification
        dummy_image = np.zeros((100, 100))
        classified_sources = detector._classify_sources(sources, dummy_image)
    
    # Extract star candidates (class == 1)
    if 'class' in classified_sources.dtype.names:
        star_mask = classified_sources['class'] == 1
        return classified_sources[star_mask].tolist()
    else:
        return identify_star_candidates(sources, mag_limit, size_ratio_limits[1])


def run_source_detection(image: np.ndarray, 
                        config: Optional[DetectionConfig] = None,
                        return_background: bool = False) -> List[Source]:
    """
    Convenience function to run source detection with sensible defaults.
    
    This function provides a simple interface to the advanced source detection
    capabilities while maintaining compatibility with the legacy interface.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image for source detection
    config : DetectionConfig, optional
        Detection configuration. If None, uses default parameters
    return_background : bool, default=False
        Whether to return background information
        
    Returns:
    --------
    List[Source]
        List of detected sources
        
    Raises:
    -------
    ValueError
        If image is invalid
    """
    logger = logging.getLogger(__name__)
    
    # Validate input
    if not isinstance(image, np.ndarray):
        raise ValueError("Input must be a numpy array")
    
    if image.ndim != 2:
        raise ValueError(f"Input must be 2D, got {image.ndim}D")
    
    if image.size == 0:
        raise ValueError("Input image is empty")
    
    # Use default config if none provided
    if config is None:
        config = DetectionConfig()
    
    try:
        # Create detector
        detector = AdvancedSourceDetector(config)
        
        # Run detection
        results = detector.detect_sources(image)
        
        # Handle case where sources might be None or not an array
        if hasattr(results, 'sources') and results.sources is not None:
            if hasattr(results.sources, '__len__'):
                logger.info(f"Detected {len(results.sources)} sources")
                num_sources = len(results.sources)
            else:
                logger.info(f"Detected sources (unsized): {results.sources}")
                num_sources = 1 if results.sources is not None else 0
        else:
            logger.info("No sources detected or results incomplete")
            num_sources = 0
        
        # Convert to Source objects for compatibility
        sources = []
        if hasattr(results, 'sources') and results.sources is not None and hasattr(results.sources, '__len__'):
            for source_data in results.sources:
                source = Source(
                    id=source_data.get('id', len(sources)),
                    x=source_data['x'],
                    y=source_data['y'],
                    flux=source_data['flux'],
                    a=source_data.get('a', 1.0),
                    b=source_data.get('b', 1.0),
                    theta=source_data.get('theta', 0.0),
                    peak=source_data.get('peak', source_data['flux']),
                    flag=source_data.get('flag', 0)
                )
                sources.append(source)
        
        if return_background:
            return sources, results.background_info
        else:
            return sources
            
    except Exception as e:
        logger.error(f"Source detection failed: {e}")
        raise
