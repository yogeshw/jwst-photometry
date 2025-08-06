"""
Advanced Segmentation and Masking Module for JWST Photometry

This module implements sophisticated segmentation techniques using SEP,
including watershed segmentation, adaptive apertures, contamination masks,
Kron apertures, and bad pixel handling.
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple, List, Union
from dataclasses import dataclass, field

import sep
from scipy import ndimage
from scipy.ndimage import (
    watershed_ift, binary_dilation, binary_erosion, 
    label, center_of_mass, distance_transform_edt
)
from skimage.segmentation import watershed
from skimage.feature import peak_local_maxima
from skimage.morphology import remove_small_objects
from astropy.table import Table
import matplotlib.pyplot as plt


@dataclass
class SegmentationConfig:
    """Configuration parameters for segmentation and masking."""
    
    # Watershed segmentation
    use_watershed: bool = True
    watershed_threshold: float = 2.0
    min_distance: int = 3
    peak_threshold: float = 0.1
    
    # Adaptive apertures
    use_adaptive_apertures: bool = True
    aperture_scale_factor: float = 2.5
    min_aperture_radius: float = 1.0
    max_aperture_radius: float = 20.0
    
    # Kron apertures
    use_kron_apertures: bool = True
    kron_factor: float = 2.5
    min_kron_radius: float = 1.0
    max_kron_radius: float = 15.0
    kron_minimum_radius: float = 1.75  # Minimum radius in units of A or B
    
    # Contamination masking
    create_contamination_masks: bool = True
    contamination_buffer: float = 1.5
    neighbor_threshold: float = 0.1
    
    # Bad pixel handling
    handle_bad_pixels: bool = True
    interpolate_bad_pixels: bool = True
    max_interpolation_size: int = 5
    
    # Cosmic ray detection
    detect_cosmic_rays: bool = True
    cosmic_ray_threshold: float = 5.0
    cosmic_ray_scale: float = 2.0
    
    # Quality control
    min_segment_size: int = 3
    max_segments_per_source: int = 10
    remove_boundary_segments: bool = True
    boundary_buffer: int = 5


@dataclass
class SegmentationResults:
    """Container for segmentation results."""
    
    segmentation_map: np.ndarray
    source_segments: Dict[int, np.ndarray]
    adaptive_apertures: Dict[int, Dict[str, Any]]
    kron_apertures: Dict[int, Dict[str, Any]]
    contamination_masks: Dict[int, np.ndarray]
    bad_pixel_mask: Optional[np.ndarray] = None
    cosmic_ray_mask: Optional[np.ndarray] = None
    statistics: Dict[str, Any] = field(default_factory=dict)


class AdvancedSegmentationProcessor:
    """
    Advanced segmentation and masking processor for astronomical images.
    
    This class provides sophisticated segmentation capabilities including:
    - Watershed segmentation for complex source morphologies
    - Adaptive apertures based on source properties
    - Kron apertures with proper elliptical fitting
    - Contamination masks for neighboring sources
    - Bad pixel and cosmic ray detection/handling
    """
    
    def __init__(self, config: Optional[SegmentationConfig] = None):
        """
        Initialize the segmentation processor.
        
        Parameters:
        -----------
        config : SegmentationConfig, optional
            Segmentation configuration. If None, uses defaults.
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or SegmentationConfig()
    
    def process_segmentation(self, 
                           image: np.ndarray,
                           sources: np.ndarray,
                           background_map: Optional[np.ndarray] = None,
                           rms_map: Optional[np.ndarray] = None,
                           initial_mask: Optional[np.ndarray] = None) -> SegmentationResults:
        """
        Perform comprehensive segmentation and masking.
        
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
        initial_mask : numpy.ndarray, optional
            Initial bad pixel mask
            
        Returns:
        --------
        SegmentationResults
            Complete segmentation results
        """
        self.logger.info("Starting advanced segmentation processing")
        
        # Validate inputs
        self._validate_inputs(image, sources, background_map, rms_map, initial_mask)
        
        # Prepare working image
        work_image = image.copy()
        if background_map is not None:
            work_image = work_image - background_map
        
        # Initialize bad pixel mask
        bad_pixel_mask = initial_mask.copy() if initial_mask is not None else np.zeros_like(image, dtype=bool)
        
        # Detect and mask cosmic rays
        cosmic_ray_mask = None
        if self.config.detect_cosmic_rays:
            cosmic_ray_mask = self._detect_cosmic_rays(work_image, rms_map)
            bad_pixel_mask |= cosmic_ray_mask
        
        # Interpolate bad pixels if requested
        if self.config.interpolate_bad_pixels and np.any(bad_pixel_mask):
            work_image = self._interpolate_bad_pixels(work_image, bad_pixel_mask)
        
        # Create segmentation map
        if self.config.use_watershed:
            segmentation_map = self._watershed_segmentation(
                work_image, sources, rms_map, bad_pixel_mask
            )
        else:
            segmentation_map = self._basic_segmentation(work_image, sources)
        
        # Extract individual source segments
        source_segments = self._extract_source_segments(segmentation_map, sources)
        
        # Create adaptive apertures
        adaptive_apertures = {}
        if self.config.use_adaptive_apertures:
            adaptive_apertures = self._create_adaptive_apertures(
                sources, work_image, segmentation_map
            )
        
        # Create Kron apertures
        kron_apertures = {}
        if self.config.use_kron_apertures:
            kron_apertures = self._create_kron_apertures(
                sources, work_image, rms_map
            )
        
        # Create contamination masks
        contamination_masks = {}
        if self.config.create_contamination_masks:
            contamination_masks = self._create_contamination_masks(
                sources, segmentation_map
            )
        
        # Compute statistics
        statistics = self._compute_segmentation_statistics(
            segmentation_map, source_segments, bad_pixel_mask, cosmic_ray_mask
        )
        
        # Create results object
        results = SegmentationResults(
            segmentation_map=segmentation_map,
            source_segments=source_segments,
            adaptive_apertures=adaptive_apertures,
            kron_apertures=kron_apertures,
            contamination_masks=contamination_masks,
            bad_pixel_mask=bad_pixel_mask,
            cosmic_ray_mask=cosmic_ray_mask,
            statistics=statistics
        )
        
        self.logger.info(f"Segmentation completed - {len(source_segments)} source segments created")
        return results
    
    def _validate_inputs(self, image: np.ndarray,
                        sources: np.ndarray,
                        background_map: Optional[np.ndarray],
                        rms_map: Optional[np.ndarray],
                        initial_mask: Optional[np.ndarray]) -> None:
        """Validate input arrays."""
        if image.ndim != 2:
            raise ValueError(f"Image must be 2D, got {image.ndim}D")
        
        arrays_to_check = [
            (background_map, "background_map"),
            (rms_map, "rms_map"),
            (initial_mask, "initial_mask")
        ]
        
        for array, name in arrays_to_check:
            if array is not None and array.shape != image.shape:
                raise ValueError(f"{name} shape {array.shape} doesn't match image shape {image.shape}")
    
    def _detect_cosmic_rays(self, image: np.ndarray, rms_map: Optional[np.ndarray]) -> np.ndarray:
        """
        Detect cosmic rays using a combination of techniques.
        
        Parameters:
        -----------
        image : numpy.ndarray
            Input image
        rms_map : numpy.ndarray, optional
            RMS map for normalization
            
        Returns:
        --------
        numpy.ndarray
            Cosmic ray mask
        """
        self.logger.debug("Detecting cosmic rays")
        
        cosmic_ray_mask = np.zeros_like(image, dtype=bool)
        
        try:
            # Method 1: Sharp outliers
            if rms_map is not None:
                # Normalize by local noise
                normalized_image = image / np.maximum(rms_map, np.nanmedian(rms_map) * 0.1)
                threshold = self.config.cosmic_ray_threshold
            else:
                # Use global statistics
                normalized_image = image / np.nanstd(image)
                threshold = self.config.cosmic_ray_threshold
            
            # Find pixels significantly above threshold
            high_pixels = normalized_image > threshold
            
            # Method 2: Sharpness test - cosmic rays are typically very sharp
            from scipy import ndimage
            
            # Laplacian filter to detect sharp features
            laplacian = ndimage.laplace(image)
            laplacian_threshold = self.config.cosmic_ray_scale * np.nanstd(laplacian)
            sharp_pixels = np.abs(laplacian) > laplacian_threshold
            
            # Combine criteria
            cosmic_ray_candidates = high_pixels & sharp_pixels
            
            # Remove large connected regions (likely real sources)
            labeled_regions, n_regions = label(cosmic_ray_candidates)
            
            for region_id in range(1, n_regions + 1):
                region_mask = labeled_regions == region_id
                region_size = np.sum(region_mask)
                
                # Keep only small, isolated regions
                if region_size <= 5:  # Cosmic rays are typically 1-5 pixels
                    cosmic_ray_mask |= region_mask
            
            n_cosmic_rays = np.sum(cosmic_ray_mask)
            if n_cosmic_rays > 0:
                self.logger.info(f"Detected {n_cosmic_rays} cosmic ray pixels")
            
        except Exception as e:
            self.logger.warning(f"Cosmic ray detection failed: {e}")
        
        return cosmic_ray_mask
    
    def _interpolate_bad_pixels(self, image: np.ndarray, bad_pixel_mask: np.ndarray) -> np.ndarray:
        """
        Interpolate over bad pixels using nearby good pixels.
        
        Parameters:
        -----------
        image : numpy.ndarray
            Input image
        bad_pixel_mask : numpy.ndarray
            Bad pixel mask
            
        Returns:
        --------
        numpy.ndarray
            Image with interpolated bad pixels
        """
        if not np.any(bad_pixel_mask):
            return image
        
        self.logger.debug(f"Interpolating {np.sum(bad_pixel_mask)} bad pixels")
        
        interpolated_image = image.copy()
        
        # Find connected regions of bad pixels
        labeled_regions, n_regions = label(bad_pixel_mask)
        
        for region_id in range(1, n_regions + 1):
            region_mask = labeled_regions == region_id
            region_size = np.sum(region_mask)
            
            # Only interpolate small regions
            if region_size <= self.config.max_interpolation_size:
                # Use scipy's interpolation
                try:
                    # Get coordinates of bad pixels
                    bad_coords = np.where(region_mask)
                    
                    # Create distance map to find nearest good pixels
                    distance_map = distance_transform_edt(~region_mask)
                    
                    # For each bad pixel, find nearby good pixels and interpolate
                    for bad_y, bad_x in zip(*bad_coords):
                        # Define local region for interpolation
                        y_min = max(0, bad_y - 3)
                        y_max = min(image.shape[0], bad_y + 4)
                        x_min = max(0, bad_x - 3)
                        x_max = min(image.shape[1], bad_x + 4)
                        
                        # Extract local region
                        local_image = image[y_min:y_max, x_min:x_max]
                        local_mask = bad_pixel_mask[y_min:y_max, x_min:x_max]
                        
                        # Get good pixels in local region
                        good_pixels = local_image[~local_mask]
                        
                        if len(good_pixels) > 0:
                            # Use median of nearby good pixels
                            interpolated_image[bad_y, bad_x] = np.median(good_pixels)
                        else:
                            # Use global median as fallback
                            global_good = image[~bad_pixel_mask]
                            if len(global_good) > 0:
                                interpolated_image[bad_y, bad_x] = np.median(global_good)
                
                except Exception:
                    # If interpolation fails, use local median
                    coords = np.where(region_mask)
                    for y, x in zip(*coords):
                        # Use 5x5 neighborhood
                        y_min, y_max = max(0, y-2), min(image.shape[0], y+3)
                        x_min, x_max = max(0, x-2), min(image.shape[1], x+3)
                        
                        neighborhood = image[y_min:y_max, x_min:x_max]
                        neighborhood_mask = bad_pixel_mask[y_min:y_max, x_min:x_max]
                        
                        good_neighbors = neighborhood[~neighborhood_mask]
                        if len(good_neighbors) > 0:
                            interpolated_image[y, x] = np.median(good_neighbors)
        
        return interpolated_image
    
    def _watershed_segmentation(self, image: np.ndarray,
                              sources: np.ndarray,
                              rms_map: Optional[np.ndarray],
                              mask: np.ndarray) -> np.ndarray:
        """
        Perform watershed segmentation for complex source morphologies.
        
        Parameters:
        -----------
        image : numpy.ndarray
            Input image
        sources : numpy.ndarray
            Source catalog
        rms_map : numpy.ndarray, optional
            RMS map
        mask : numpy.ndarray
            Bad pixel mask
            
        Returns:
        --------
        numpy.ndarray
            Watershed segmentation map
        """
        self.logger.debug("Performing watershed segmentation")
        
        if len(sources) == 0:
            return np.zeros_like(image, dtype=np.int32)
        
        # Create distance map from source centers
        markers = np.zeros_like(image, dtype=np.int32)
        
        # Place markers at source centers
        for i, source in enumerate(sources):
            x, y = int(round(source['x'])), int(round(source['y']))
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                if not mask[y, x]:  # Only place marker if not masked
                    markers[y, x] = i + 1
        
        # Create elevation map (inverted image for watershed)
        if rms_map is not None:
            # Normalize by local noise
            elevation = -image / np.maximum(rms_map, np.nanmedian(rms_map) * 0.1)
        else:
            elevation = -image
        
        # Apply threshold to limit segmentation to significant regions
        if rms_map is not None:
            threshold_map = self.config.watershed_threshold * rms_map
        else:
            threshold_map = self.config.watershed_threshold * np.nanstd(image)
        
        significant_region = image > threshold_map
        
        # Perform watershed
        try:
            # Use skimage watershed
            segmentation = watershed(elevation, markers, mask=~(mask | ~significant_region))
            
            # Clean up segmentation
            segmentation = self._clean_segmentation(segmentation, sources)
            
        except Exception as e:
            self.logger.warning(f"Watershed segmentation failed: {e}, falling back to basic")
            segmentation = self._basic_segmentation(image, sources)
        
        return segmentation.astype(np.int32)
    
    def _basic_segmentation(self, image: np.ndarray, sources: np.ndarray) -> np.ndarray:
        """
        Create basic elliptical segmentation.
        
        Parameters:
        -----------
        image : numpy.ndarray
            Input image
        sources : numpy.ndarray
            Source catalog
            
        Returns:
        --------
        numpy.ndarray
            Basic segmentation map
        """
        segmentation_map = np.zeros_like(image, dtype=np.int32)
        
        if len(sources) == 0:
            return segmentation_map
        
        y_coords, x_coords = np.ogrid[:image.shape[0], :image.shape[1]]
        
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
            
            # Ellipse equation
            ellipse_mask = (x_rot / (2.5 * a))**2 + (y_rot / (2.5 * b))**2 <= 1
            
            # Only assign to pixels not already assigned
            new_pixels = ellipse_mask & (segmentation_map == 0)
            segmentation_map[new_pixels] = source_id
        
        return segmentation_map
    
    def _clean_segmentation(self, segmentation: np.ndarray, sources: np.ndarray) -> np.ndarray:
        """
        Clean up segmentation map.
        
        Parameters:
        -----------
        segmentation : numpy.ndarray
            Raw segmentation map
        sources : numpy.ndarray
            Source catalog
            
        Returns:
        --------
        numpy.ndarray
            Cleaned segmentation map
        """
        cleaned_segmentation = segmentation.copy()
        
        # Remove small segments
        for source_id in range(1, len(sources) + 1):
            source_mask = segmentation == source_id
            if np.sum(source_mask) < self.config.min_segment_size:
                cleaned_segmentation[source_mask] = 0
        
        # Remove boundary segments if requested
        if self.config.remove_boundary_segments:
            buffer = self.config.boundary_buffer
            ny, nx = segmentation.shape
            
            boundary_mask = (
                (np.arange(ny)[:, None] < buffer) |
                (np.arange(ny)[:, None] >= ny - buffer) |
                (np.arange(nx)[None, :] < buffer) |
                (np.arange(nx)[None, :] >= nx - buffer)
            )
            
            cleaned_segmentation[boundary_mask] = 0
        
        return cleaned_segmentation
    
    def _extract_source_segments(self, segmentation_map: np.ndarray, 
                                sources: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Extract individual source segments.
        
        Parameters:
        -----------
        segmentation_map : numpy.ndarray
            Segmentation map
        sources : numpy.ndarray
            Source catalog
            
        Returns:
        --------
        dict
            Dictionary mapping source IDs to segment masks
        """
        source_segments = {}
        
        for i, source in enumerate(sources):
            source_id = i + 1
            segment_mask = segmentation_map == source_id
            
            if np.any(segment_mask):
                source_segments[source_id] = segment_mask
        
        return source_segments
    
    def _create_adaptive_apertures(self, sources: np.ndarray,
                                 image: np.ndarray,
                                 segmentation_map: np.ndarray) -> Dict[int, Dict[str, Any]]:
        """
        Create adaptive apertures based on source morphology.
        
        Parameters:
        -----------
        sources : numpy.ndarray
            Source catalog
        image : numpy.ndarray
            Input image
        segmentation_map : numpy.ndarray
            Segmentation map
            
        Returns:
        --------
        dict
            Dictionary of adaptive aperture parameters
        """
        self.logger.debug("Creating adaptive apertures")
        
        adaptive_apertures = {}
        
        for i, source in enumerate(sources):
            source_id = i + 1
            
            try:
                # Get source segment
                source_mask = segmentation_map == source_id
                
                if not np.any(source_mask):
                    continue
                
                # Calculate adaptive radius based on source flux distribution
                coords = np.where(source_mask)
                if len(coords[0]) == 0:
                    continue
                
                # Get flux-weighted center
                flux_weights = image[coords]
                flux_weights = np.maximum(flux_weights, 0)  # Ensure positive
                
                if np.sum(flux_weights) > 0:
                    center_y = np.average(coords[0], weights=flux_weights)
                    center_x = np.average(coords[1], weights=flux_weights)
                else:
                    center_y = np.mean(coords[0])
                    center_x = np.mean(coords[1])
                
                # Calculate second moments for size estimation
                dy = coords[0] - center_y
                dx = coords[1] - center_x
                
                if len(flux_weights) > 0 and np.sum(flux_weights) > 0:
                    # Flux-weighted second moments
                    m_xx = np.average(dx**2, weights=flux_weights)
                    m_yy = np.average(dy**2, weights=flux_weights)
                    m_xy = np.average(dx * dy, weights=flux_weights)
                else:
                    # Unweighted moments
                    m_xx = np.mean(dx**2)
                    m_yy = np.mean(dy**2)
                    m_xy = np.mean(dx * dy)
                
                # Calculate adaptive radius
                radius_squared = (m_xx + m_yy) / 2.0
                adaptive_radius = self.config.aperture_scale_factor * np.sqrt(radius_squared)
                
                # Apply constraints
                adaptive_radius = np.clip(
                    adaptive_radius,
                    self.config.min_aperture_radius,
                    self.config.max_aperture_radius
                )
                
                adaptive_apertures[source_id] = {
                    'center_x': center_x,
                    'center_y': center_y,
                    'radius': adaptive_radius,
                    'ellipticity': np.sqrt(m_xy**2 + ((m_xx - m_yy)/2)**2) / (m_xx + m_yy),
                    'position_angle': 0.5 * np.arctan2(2 * m_xy, m_xx - m_yy)
                }
                
            except Exception as e:
                self.logger.warning(f"Failed to create adaptive aperture for source {source_id}: {e}")
                continue
        
        return adaptive_apertures
    
    def _create_kron_apertures(self, sources: np.ndarray,
                             image: np.ndarray,
                             rms_map: Optional[np.ndarray]) -> Dict[int, Dict[str, Any]]:
        """
        Create Kron apertures with proper elliptical fitting.
        
        Parameters:
        -----------
        sources : numpy.ndarray
            Source catalog
        image : numpy.ndarray
            Input image
        rms_map : numpy.ndarray, optional
            RMS map
            
        Returns:
        --------
        dict
            Dictionary of Kron aperture parameters
        """
        self.logger.debug("Creating Kron apertures")
        
        kron_apertures = {}
        
        for i, source in enumerate(sources):
            source_id = i + 1
            
            try:
                # Source parameters
                x0, y0 = source['x'], source['y']
                a, b = source['a'], source['b']
                theta = source['theta']
                
                # Calculate Kron radius
                kron_radius = self._calculate_kron_radius(source, image, rms_map)
                
                # Apply constraints
                kron_radius = np.clip(
                    kron_radius,
                    self.config.min_kron_radius,
                    self.config.max_kron_radius
                )
                
                # Ensure minimum radius in units of semi-major axis
                min_radius_pixels = self.config.kron_minimum_radius * max(a, b)
                kron_radius = max(kron_radius, min_radius_pixels)
                
                kron_apertures[source_id] = {
                    'center_x': x0,
                    'center_y': y0,
                    'kron_radius': kron_radius,
                    'semi_major': kron_radius * a / np.sqrt(a * b),
                    'semi_minor': kron_radius * b / np.sqrt(a * b),
                    'position_angle': theta,
                    'ellipticity': 1 - b / a
                }
                
            except Exception as e:
                self.logger.warning(f"Failed to create Kron aperture for source {source_id}: {e}")
                continue
        
        return kron_apertures
    
    def _calculate_kron_radius(self, source: np.ndarray,
                             image: np.ndarray,
                             rms_map: Optional[np.ndarray]) -> float:
        """
        Calculate Kron radius for a source.
        
        Parameters:
        -----------
        source : numpy.ndarray
            Source parameters
        image : numpy.ndarray
            Input image
        rms_map : numpy.ndarray, optional
            RMS map
            
        Returns:
        --------
        float
            Kron radius in pixels
        """
        try:
            # Source parameters
            x0, y0 = source['x'], source['y']
            a, b = source['a'], source['b']
            theta = source['theta']
            
            # Create series of elliptical apertures
            max_radius = min(3 * max(a, b), 20.0)
            radii = np.linspace(0.5, max_radius, 20)
            
            # Measure flux and mean radius for each aperture
            fluxes = []
            mean_radii = []
            
            for radius in radii:
                try:
                    # Create elliptical aperture
                    flux, error, flag = sep.sum_ellipse(
                        image, [x0], [y0], [radius * a / np.sqrt(a * b)], 
                        [radius * b / np.sqrt(a * b)], [theta],
                        r=radius
                    )
                    
                    if flux[0] > 0:
                        fluxes.append(flux[0])
                        mean_radii.append(radius)
                
                except:
                    continue
            
            if len(fluxes) < 3:
                # Not enough points for Kron calculation
                return self.config.kron_factor * np.sqrt(a * b)
            
            fluxes = np.array(fluxes)
            mean_radii = np.array(mean_radii)
            
            # Calculate Kron radius: R_K = Σ(r * I(r)) / Σ(I(r))
            if np.sum(fluxes) > 0:
                # Use differential fluxes
                diff_fluxes = np.diff(fluxes)
                diff_radii = mean_radii[1:]
                
                if len(diff_fluxes) > 0 and np.sum(diff_fluxes) > 0:
                    kron_radius = np.sum(diff_radii * diff_fluxes) / np.sum(diff_fluxes)
                    return self.config.kron_factor * kron_radius
            
            # Fallback to geometric mean
            return self.config.kron_factor * np.sqrt(a * b)
            
        except Exception:
            # Ultimate fallback
            return self.config.kron_factor * np.sqrt(source['a'] * source['b'])
    
    def _create_contamination_masks(self, sources: np.ndarray,
                                  segmentation_map: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Create contamination masks for neighboring sources.
        
        Parameters:
        -----------
        sources : numpy.ndarray
            Source catalog
        segmentation_map : numpy.ndarray
            Segmentation map
            
        Returns:
        --------
        dict
            Dictionary of contamination masks
        """
        self.logger.debug("Creating contamination masks")
        
        contamination_masks = {}
        
        for i, source in enumerate(sources):
            source_id = i + 1
            
            try:
                # Get source position and size
                x0, y0 = source['x'], source['y']
                a, b = source['a'], source['b']
                
                # Define contamination region around source
                contamination_radius = self.config.contamination_buffer * max(a, b)
                
                # Create circular mask for contamination check
                y_coords, x_coords = np.ogrid[:segmentation_map.shape[0], :segmentation_map.shape[1]]
                distance_map = np.sqrt((x_coords - x0)**2 + (y_coords - y0)**2)
                
                contamination_region = distance_map <= contamination_radius
                
                # Find other sources in contamination region
                contaminating_sources = segmentation_map[contamination_region]
                unique_contaminants = np.unique(contaminating_sources)
                unique_contaminants = unique_contaminants[
                    (unique_contaminants > 0) & (unique_contaminants != source_id)
                ]
                
                # Create contamination mask
                contamination_mask = np.zeros_like(segmentation_map, dtype=bool)
                
                for contaminant_id in unique_contaminants:
                    contaminant_mask = segmentation_map == contaminant_id
                    
                    # Only include contamination if it's significant
                    contaminant_area = np.sum(contaminant_mask & contamination_region)
                    total_contaminant_area = np.sum(contaminant_mask)
                    
                    if (contaminant_area / total_contaminant_area) > self.config.neighbor_threshold:
                        contamination_mask |= contaminant_mask
                
                contamination_masks[source_id] = contamination_mask
                
            except Exception as e:
                self.logger.warning(f"Failed to create contamination mask for source {source_id}: {e}")
                contamination_masks[source_id] = np.zeros_like(segmentation_map, dtype=bool)
        
        return contamination_masks
    
    def _compute_segmentation_statistics(self, segmentation_map: np.ndarray,
                                       source_segments: Dict[int, np.ndarray],
                                       bad_pixel_mask: Optional[np.ndarray],
                                       cosmic_ray_mask: Optional[np.ndarray]) -> Dict[str, Any]:
        """
        Compute segmentation statistics.
        
        Parameters:
        -----------
        segmentation_map : numpy.ndarray
            Segmentation map
        source_segments : dict
            Source segments
        bad_pixel_mask : numpy.ndarray, optional
            Bad pixel mask
        cosmic_ray_mask : numpy.ndarray, optional
            Cosmic ray mask
            
        Returns:
        --------
        dict
            Statistics dictionary
        """
        stats = {}
        
        # Basic statistics
        stats['n_segments'] = len(source_segments)
        segmented_pixels = np.sum(segmentation_map > 0)
        stats['segmented_fraction'] = segmented_pixels / segmentation_map.size
        
        # Segment size statistics
        if source_segments:
            segment_sizes = [np.sum(mask) for mask in source_segments.values()]
            stats['segment_size_stats'] = {
                'min': np.min(segment_sizes),
                'max': np.max(segment_sizes),
                'median': np.median(segment_sizes),
                'mean': np.mean(segment_sizes)
            }
        
        # Bad pixel statistics
        if bad_pixel_mask is not None:
            stats['bad_pixel_fraction'] = np.sum(bad_pixel_mask) / bad_pixel_mask.size
        
        # Cosmic ray statistics
        if cosmic_ray_mask is not None:
            stats['cosmic_ray_fraction'] = np.sum(cosmic_ray_mask) / cosmic_ray_mask.size
        
        return stats
    
    def plot_segmentation_diagnostics(self, results: SegmentationResults,
                                    image: np.ndarray,
                                    sources: np.ndarray,
                                    output_path: Optional[str] = None) -> None:
        """
        Create diagnostic plots for segmentation results.
        
        Parameters:
        -----------
        results : SegmentationResults
            Segmentation results
        image : numpy.ndarray
            Original image
        sources : numpy.ndarray
            Source catalog
        output_path : str, optional
            Path to save the plot
        """
        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            # Original image
            im1 = axes[0, 0].imshow(image, origin='lower', cmap='viridis')
            axes[0, 0].set_title('Original Image')
            plt.colorbar(im1, ax=axes[0, 0])
            
            # Segmentation map
            im2 = axes[0, 1].imshow(results.segmentation_map, origin='lower', cmap='tab20')
            axes[0, 1].set_title(f'Segmentation Map\\n{len(results.source_segments)} segments')
            
            # Bad pixel and cosmic ray masks
            combined_mask = np.zeros_like(image, dtype=int)
            if results.bad_pixel_mask is not None:
                combined_mask[results.bad_pixel_mask] = 1
            if results.cosmic_ray_mask is not None:
                combined_mask[results.cosmic_ray_mask] = 2
            
            im3 = axes[0, 2].imshow(combined_mask, origin='lower', cmap='Set1')
            axes[0, 2].set_title('Bad Pixels & Cosmic Rays')
            
            # Adaptive apertures
            axes[1, 0].imshow(image, origin='lower', cmap='gray', alpha=0.7)
            for source_id, aperture in results.adaptive_apertures.items():
                circle = plt.Circle(
                    (aperture['center_x'], aperture['center_y']),
                    aperture['radius'],
                    fill=False, color='red', linewidth=2
                )
                axes[1, 0].add_patch(circle)
            axes[1, 0].set_title('Adaptive Apertures')
            
            # Kron apertures
            axes[1, 1].imshow(image, origin='lower', cmap='gray', alpha=0.7)
            for source_id, aperture in results.kron_apertures.items():
                from matplotlib.patches import Ellipse
                ellipse = Ellipse(
                    (aperture['center_x'], aperture['center_y']),
                    2 * aperture['semi_major'], 2 * aperture['semi_minor'],
                    angle=np.degrees(aperture['position_angle']),
                    fill=False, color='blue', linewidth=2
                )
                axes[1, 1].add_patch(ellipse)
            axes[1, 1].set_title('Kron Apertures')
            
            # Segment size distribution
            if results.source_segments:
                segment_sizes = [np.sum(mask) for mask in results.source_segments.values()]
                axes[1, 2].hist(segment_sizes, bins=20, alpha=0.7)
                axes[1, 2].set_xlabel('Segment Size (pixels)')
                axes[1, 2].set_ylabel('Count')
                axes[1, 2].set_title('Segment Size Distribution')
            
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                self.logger.info(f"Segmentation diagnostics saved to {output_path}")
            else:
                plt.show()
            
            plt.close()
            
        except Exception as e:
            self.logger.warning(f"Failed to create diagnostic plots: {e}")


# Convenience functions
def create_segmentation_map(image: np.ndarray,
                          sources: np.ndarray,
                          config: Optional[SegmentationConfig] = None) -> np.ndarray:
    """
    Convenience function to create segmentation map.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image
    sources : numpy.ndarray
        Source catalog
    config : SegmentationConfig, optional
        Configuration parameters
        
    Returns:
    --------
    numpy.ndarray
        Segmentation map
    """
    processor = AdvancedSegmentationProcessor(config)
    results = processor.process_segmentation(image, sources)
    return results.segmentation_map


def create_kron_apertures(sources: np.ndarray,
                        image: np.ndarray,
                        config: Optional[SegmentationConfig] = None) -> Dict[int, Dict[str, Any]]:
    """
    Convenience function to create Kron apertures.
    
    Parameters:
    -----------
    sources : numpy.ndarray
        Source catalog
    image : numpy.ndarray
        Input image
    config : SegmentationConfig, optional
        Configuration parameters
        
    Returns:
    --------
    dict
        Kron aperture parameters
    """
    processor = AdvancedSegmentationProcessor(config)
    return processor._create_kron_apertures(sources, image, None)
