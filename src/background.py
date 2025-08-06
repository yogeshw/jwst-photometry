"""
Comprehensive Background Estimation Module for JWST Photometry

This module implements sophisticated background estimation using SEP's Background class,
including multi-scale modeling, spatially varying estimation, and bad pixel handling.
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple, Union, List
from dataclasses import dataclass
import warnings

import sep
from scipy import ndimage
from scipy.ndimage import binary_dilation, binary_erosion
from astropy.stats import sigma_clipped_stats
import matplotlib.pyplot as plt


@dataclass
class BackgroundConfig:
    """Configuration parameters for background estimation."""
    
    # Basic SEP background parameters
    mask_threshold: float = 0.0
    filter_threshold: float = 0.0
    box_size: Tuple[int, int] = (64, 64)
    filter_size: Tuple[int, int] = (3, 3)
    
    # Advanced parameters
    use_global_rms: bool = False
    subtract_median: bool = True
    mask_sources: bool = True
    
    # Source masking parameters
    detection_threshold: float = 2.0
    minarea: int = 5
    dilate_mask: int = 3
    
    # Quality control
    max_iterations: int = 3
    convergence_threshold: float = 0.01
    outlier_rejection: bool = True
    outlier_sigma: float = 3.0
    
    # Multi-scale modeling
    use_multiscale: bool = False
    scales: List[int] = None
    
    # Gradient correction
    correct_gradients: bool = True
    gradient_order: int = 2


class BackgroundEstimator:
    """
    Advanced background estimation for astronomical images using SEP.
    
    This class provides sophisticated background modeling capabilities including:
    - Multi-scale background estimation
    - Spatially varying background maps
    - Source masking and iterative refinement
    - Bad pixel handling and interpolation
    - Background gradient correction
    """
    
    def __init__(self, config: Optional[BackgroundConfig] = None):
        """
        Initialize the background estimator.
        
        Parameters:
        -----------
        config : BackgroundConfig, optional
            Configuration parameters. If None, uses defaults.
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or BackgroundConfig()
        
        # Initialize default scales if not provided
        if self.config.use_multiscale and self.config.scales is None:
            self.config.scales = [32, 64, 128]
    
    def estimate_background(self, 
                          image: np.ndarray,
                          mask: Optional[np.ndarray] = None,
                          weight: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Estimate background map with comprehensive modeling.
        
        Parameters:
        -----------
        image : numpy.ndarray
            Input image data
        mask : numpy.ndarray, optional
            Bad pixel mask (True = bad pixel)
        weight : numpy.ndarray, optional
            Weight map for the image
            
        Returns:
        --------
        tuple
            - Background map
            - Background RMS map
            - Statistics dictionary
            
        Raises:
        -------
        ValueError
            If input validation fails
        """
        self.logger.info("Starting background estimation")
        
        # Validate inputs
        self._validate_inputs(image, mask, weight)
        
        # Initialize working arrays
        work_image = image.copy()
        work_mask = mask.copy() if mask is not None else np.zeros_like(image, dtype=bool)
        
        # Iterative background estimation with source masking
        background_map, rms_map, stats = self._iterative_background_estimation(
            work_image, work_mask, weight
        )
        
        # Apply gradient correction if requested
        if self.config.correct_gradients:
            background_map = self._correct_background_gradients(
                background_map, work_mask
            )
        
        # Final statistics
        final_stats = self._compute_background_statistics(
            image, background_map, rms_map, work_mask
        )
        stats.update(final_stats)
        
        self.logger.info(f"Background estimation completed - Global RMS: {stats['global_rms']:.3e}")
        
        return background_map, rms_map, stats
    
    def _validate_inputs(self, image: np.ndarray, 
                        mask: Optional[np.ndarray], 
                        weight: Optional[np.ndarray]) -> None:
        """Validate input arrays."""
        if image.ndim != 2:
            raise ValueError(f"Image must be 2D, got {image.ndim}D")
        
        if mask is not None and mask.shape != image.shape:
            raise ValueError(f"Mask shape {mask.shape} doesn't match image shape {image.shape}")
        
        if weight is not None and weight.shape != image.shape:
            raise ValueError(f"Weight shape {weight.shape} doesn't match image shape {image.shape}")
        
        # Check for finite values
        finite_pixels = np.isfinite(image)
        if not np.any(finite_pixels):
            raise ValueError("Image contains no finite values")
        
        finite_fraction = np.sum(finite_pixels) / image.size
        if finite_fraction < 0.1:
            self.logger.warning(f"Only {finite_fraction:.1%} of pixels are finite")
    
    def _iterative_background_estimation(self, 
                                       image: np.ndarray,
                                       mask: np.ndarray,
                                       weight: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Perform iterative background estimation with source masking.
        
        Parameters:
        -----------
        image : numpy.ndarray
            Input image
        mask : numpy.ndarray
            Initial mask
        weight : numpy.ndarray, optional
            Weight array
            
        Returns:
        --------
        tuple
            Background map, RMS map, and statistics
        """
        stats = {'iterations': 0, 'convergence_history': []}
        
        # Initial background estimation
        background_map, rms_map = self._compute_sep_background(image, mask, weight)
        previous_rms = np.median(rms_map[~mask])
        
        for iteration in range(self.config.max_iterations):
            self.logger.debug(f"Background iteration {iteration + 1}")
            
            # Subtract current background estimate
            residual_image = image - background_map
            
            # Detect sources in residual image for masking
            if self.config.mask_sources:
                source_mask = self._detect_sources_for_masking(
                    residual_image, rms_map, mask
                )
                combined_mask = mask | source_mask
            else:
                combined_mask = mask
            
            # Re-estimate background with updated mask
            new_background, new_rms = self._compute_sep_background(
                image, combined_mask, weight
            )
            
            # Check convergence
            current_rms = np.median(new_rms[~combined_mask])
            rms_change = abs(current_rms - previous_rms) / previous_rms
            
            stats['convergence_history'].append({
                'iteration': iteration + 1,
                'rms': current_rms,
                'rms_change': rms_change,
                'masked_fraction': np.sum(combined_mask) / combined_mask.size
            })
            
            self.logger.debug(f"  RMS: {current_rms:.3e}, change: {rms_change:.3%}")
            
            # Update for next iteration
            background_map = new_background
            rms_map = new_rms
            mask = combined_mask
            
            # Check convergence
            if rms_change < self.config.convergence_threshold:
                self.logger.debug(f"Converged after {iteration + 1} iterations")
                break
            
            previous_rms = current_rms
        
        stats['iterations'] = iteration + 1
        stats['final_mask_fraction'] = np.sum(mask) / mask.size
        
        return background_map, rms_map, stats
    
    def _compute_sep_background(self, 
                              image: np.ndarray,
                              mask: np.ndarray,
                              weight: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute background using SEP's Background class.
        
        Parameters:
        -----------
        image : numpy.ndarray
            Input image
        mask : numpy.ndarray
            Mask array
        weight : numpy.ndarray, optional
            Weight array
            
        Returns:
        --------
        tuple
            Background map and RMS map
        """
        try:
            # Prepare SEP parameters
            sep_mask = mask.astype(np.uint8) if mask is not None else None
            
            # Create SEP background object
            if self.config.use_multiscale:
                # Multi-scale background estimation
                background_map, rms_map = self._multiscale_background(
                    image, sep_mask, weight
                )
            else:
                # Standard SEP background
                background = sep.Background(
                    image,
                    mask=sep_mask,
                    maskthresh=self.config.mask_threshold,
                    filter_threshold=self.config.filter_threshold,
                    bw=self.config.box_size[0],
                    bh=self.config.box_size[1],
                    fw=self.config.filter_size[0],
                    fh=self.config.filter_size[1]
                )
                
                background_map = background.back()
                
                if self.config.use_global_rms:
                    rms_map = np.full_like(background_map, background.globalrms)
                else:
                    rms_map = background.rms()
            
            # Handle NaN values
            background_map = self._handle_nan_values(background_map, image)
            rms_map = self._handle_nan_values(rms_map, image, fill_value=np.nanmedian(rms_map))
            
            return background_map, rms_map
            
        except Exception as e:
            self.logger.error(f"SEP background estimation failed: {e}")
            # Fallback to simple background estimation
            return self._fallback_background_estimation(image, mask)
    
    def _multiscale_background(self, 
                             image: np.ndarray,
                             mask: Optional[np.ndarray],
                             weight: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Multi-scale background estimation using multiple box sizes.
        
        Parameters:
        -----------
        image : numpy.ndarray
            Input image
        mask : numpy.ndarray, optional
            Mask array
        weight : numpy.ndarray, optional
            Weight array
            
        Returns:
        --------
        tuple
            Combined background and RMS maps
        """
        self.logger.debug("Performing multi-scale background estimation")
        
        background_maps = []
        rms_maps = []
        weights = []
        
        for scale in self.config.scales:
            try:
                # Create background with current scale
                background = sep.Background(
                    image,
                    mask=mask,
                    maskthresh=self.config.mask_threshold,
                    filter_threshold=self.config.filter_threshold,
                    bw=scale,
                    bh=scale,
                    fw=min(scale//4, 7),  # Adaptive filter size
                    fh=min(scale//4, 7)
                )
                
                bg_map = background.back()
                rms_map = background.rms()
                
                # Weight by inverse scale (larger scales get less weight)
                scale_weight = 1.0 / scale
                
                background_maps.append(bg_map)
                rms_maps.append(rms_map)
                weights.append(scale_weight)
                
                self.logger.debug(f"  Scale {scale}: RMS = {background.globalrms:.3e}")
                
            except Exception as e:
                self.logger.warning(f"Failed to compute background at scale {scale}: {e}")
                continue
        
        if not background_maps:
            raise ValueError("All multi-scale background estimations failed")
        
        # Combine backgrounds using weighted average
        weights = np.array(weights)
        weights /= np.sum(weights)  # Normalize weights
        
        combined_background = np.zeros_like(background_maps[0])
        combined_rms = np.zeros_like(rms_maps[0])
        
        for bg_map, rms_map, weight in zip(background_maps, rms_maps, weights):
            combined_background += weight * bg_map
            combined_rms += weight * rms_map
        
        return combined_background, combined_rms
    
    def _detect_sources_for_masking(self, 
                                  residual_image: np.ndarray,
                                  rms_map: np.ndarray,
                                  existing_mask: np.ndarray) -> np.ndarray:
        """
        Detect sources in residual image for background masking.
        
        Parameters:
        -----------
        residual_image : numpy.ndarray
            Background-subtracted image
        rms_map : numpy.ndarray
            RMS map for detection threshold
        existing_mask : numpy.ndarray
            Existing mask to avoid
            
        Returns:
        --------
        numpy.ndarray
            Source mask
        """
        try:
            # Use local RMS for detection threshold
            threshold_map = self.config.detection_threshold * rms_map
            
            # Detect sources using SEP
            sources = sep.extract(
                residual_image,
                thresh=threshold_map,
                minarea=self.config.minarea,
                mask=existing_mask.astype(np.uint8),
                clean=True,
                clean_param=1.0
            )
            
            # Create source mask
            source_mask = np.zeros_like(existing_mask, dtype=bool)
            
            if len(sources) > 0:
                # Create segmentation map
                segmap = np.zeros_like(residual_image, dtype=np.int32)
                
                for i, source in enumerate(sources):
                    # Create elliptical mask for each source
                    y, x = np.ogrid[:residual_image.shape[0], :residual_image.shape[1]]
                    
                    # Source ellipse parameters
                    x0, y0 = source['x'], source['y']
                    a, b = source['a'], source['b']
                    theta = source['theta']
                    
                    # Ellipse equation
                    cos_theta = np.cos(theta)
                    sin_theta = np.sin(theta)
                    
                    dx = x - x0
                    dy = y - y0
                    
                    x_rot = dx * cos_theta + dy * sin_theta
                    y_rot = -dx * sin_theta + dy * cos_theta
                    
                    ellipse_mask = (x_rot/a)**2 + (y_rot/b)**2 <= 1
                    source_mask |= ellipse_mask
                
                # Dilate mask to ensure complete source masking
                if self.config.dilate_mask > 0:
                    structure = np.ones((self.config.dilate_mask, self.config.dilate_mask))
                    source_mask = binary_dilation(source_mask, structure=structure)
                
                self.logger.debug(f"Masked {len(sources)} sources "
                                f"({np.sum(source_mask)/source_mask.size:.2%} of pixels)")
            
            return source_mask
            
        except Exception as e:
            self.logger.warning(f"Source detection for masking failed: {e}")
            return np.zeros_like(existing_mask, dtype=bool)
    
    def _correct_background_gradients(self, 
                                    background_map: np.ndarray,
                                    mask: np.ndarray) -> np.ndarray:
        """
        Correct large-scale gradients in the background map.
        
        Parameters:
        -----------
        background_map : numpy.ndarray
            Original background map
        mask : numpy.ndarray
            Mask for bad pixels and sources
            
        Returns:
        --------
        numpy.ndarray
            Gradient-corrected background map
        """
        try:
            from sklearn.preprocessing import PolynomialFeatures
            from sklearn.linear_model import LinearRegression
            
            # Create coordinate grids
            ny, nx = background_map.shape
            y_coords, x_coords = np.ogrid[:ny, :nx]
            
            # Flatten coordinates and background values
            valid_mask = ~mask & np.isfinite(background_map)
            
            if np.sum(valid_mask) < 100:  # Need sufficient points for fitting
                self.logger.warning("Insufficient points for gradient correction")
                return background_map
            
            x_flat = x_coords[valid_mask].ravel()
            y_flat = y_coords[valid_mask].ravel()
            bg_flat = background_map[valid_mask].ravel()
            
            # Create polynomial features
            coords = np.column_stack([x_flat, y_flat])
            poly_features = PolynomialFeatures(degree=self.config.gradient_order)
            coords_poly = poly_features.fit_transform(coords)
            
            # Fit polynomial surface
            regressor = LinearRegression()
            regressor.fit(coords_poly, bg_flat)
            
            # Predict gradient surface for entire image
            x_grid, y_grid = np.meshgrid(np.arange(nx), np.arange(ny))
            coords_full = np.column_stack([x_grid.ravel(), y_grid.ravel()])
            coords_full_poly = poly_features.transform(coords_full)
            
            gradient_surface = regressor.predict(coords_full_poly)
            gradient_surface = gradient_surface.reshape(ny, nx)
            
            # Subtract gradient from background
            corrected_background = background_map - gradient_surface
            
            # Add back the median level to maintain calibration
            median_correction = np.nanmedian(background_map[valid_mask])
            corrected_background += median_correction
            
            self.logger.debug("Applied gradient correction to background")
            return corrected_background
            
        except ImportError:
            self.logger.warning("scikit-learn not available for gradient correction")
            return background_map
        except Exception as e:
            self.logger.warning(f"Gradient correction failed: {e}")
            return background_map
    
    def _handle_nan_values(self, 
                         array: np.ndarray, 
                         reference_image: np.ndarray,
                         fill_value: Optional[float] = None) -> np.ndarray:
        """
        Handle NaN values in background/RMS arrays.
        
        Parameters:
        -----------
        array : numpy.ndarray
            Array to process
        reference_image : numpy.ndarray
            Reference image for determining fill strategy
        fill_value : float, optional
            Value to use for filling NaNs
            
        Returns:
        --------
        numpy.ndarray
            Array with NaN values handled
        """
        nan_mask = ~np.isfinite(array)
        
        if not np.any(nan_mask):
            return array
        
        if fill_value is None:
            # Use local interpolation for small gaps
            filled_array = array.copy()
            
            # For small isolated NaN regions, use interpolation
            from scipy import ndimage
            
            # Create distance map from valid pixels
            valid_mask = ~nan_mask
            distances = ndimage.distance_transform_edt(nan_mask)
            
            # Only interpolate for NaNs close to valid pixels
            close_nans = nan_mask & (distances <= 5)
            
            if np.any(close_nans):
                # Simple interpolation using nearby valid values
                from scipy.ndimage import generic_filter
                
                def local_mean(values):
                    finite_values = values[np.isfinite(values)]
                    return np.mean(finite_values) if len(finite_values) > 0 else np.nan
                
                interpolated = generic_filter(array, local_mean, size=5, mode='constant', cval=np.nan)
                filled_array[close_nans] = interpolated[close_nans]
            
            # Fill remaining NaNs with global statistics
            remaining_nans = ~np.isfinite(filled_array)
            if np.any(remaining_nans):
                global_fill = np.nanmedian(filled_array)
                filled_array[remaining_nans] = global_fill
        else:
            filled_array = array.copy()
            filled_array[nan_mask] = fill_value
        
        return filled_array
    
    def _fallback_background_estimation(self, 
                                      image: np.ndarray,
                                      mask: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fallback background estimation when SEP fails.
        
        Parameters:
        -----------
        image : numpy.ndarray
            Input image
        mask : numpy.ndarray, optional
            Mask array
            
        Returns:
        --------
        tuple
            Background and RMS maps
        """
        self.logger.warning("Using fallback background estimation")
        
        # Use sigma-clipped statistics in local regions
        from astropy.stats import sigma_clipped_stats
        
        box_size = self.config.box_size[0]
        ny, nx = image.shape
        
        # Create output arrays
        background_map = np.zeros_like(image)
        rms_map = np.zeros_like(image)
        
        # Process in overlapping boxes
        for y in range(0, ny, box_size//2):
            for x in range(0, nx, box_size//2):
                # Define box boundaries
                y1, y2 = y, min(y + box_size, ny)
                x1, x2 = x, min(x + box_size, nx)
                
                # Extract box data
                box_data = image[y1:y2, x1:x2]
                
                if mask is not None:
                    box_mask = mask[y1:y2, x1:x2]
                    valid_data = box_data[~box_mask]
                else:
                    valid_data = box_data.ravel()
                
                # Remove non-finite values
                valid_data = valid_data[np.isfinite(valid_data)]
                
                if len(valid_data) > 10:
                    # Compute sigma-clipped statistics
                    try:
                        mean, median, std = sigma_clipped_stats(valid_data, sigma=3, maxiters=3)
                        background_val = median
                        rms_val = std
                    except:
                        background_val = np.median(valid_data)
                        rms_val = np.std(valid_data)
                else:
                    # Not enough valid data
                    background_val = 0.0
                    rms_val = 1.0
                
                # Fill output arrays
                background_map[y1:y2, x1:x2] = background_val
                rms_map[y1:y2, x1:x2] = rms_val
        
        return background_map, rms_map
    
    def _compute_background_statistics(self, 
                                     original_image: np.ndarray,
                                     background_map: np.ndarray,
                                     rms_map: np.ndarray,
                                     mask: np.ndarray) -> Dict[str, Any]:
        """
        Compute comprehensive background statistics.
        
        Parameters:
        -----------
        original_image : numpy.ndarray
            Original image data
        background_map : numpy.ndarray
            Estimated background
        rms_map : numpy.ndarray
            Background RMS map
        mask : numpy.ndarray
            Final mask used
            
        Returns:
        --------
        dict
            Background statistics
        """
        # Valid pixel mask
        valid_mask = ~mask & np.isfinite(original_image) & np.isfinite(background_map)
        
        if not np.any(valid_mask):
            return {'error': 'No valid pixels for statistics'}
        
        # Background-subtracted image
        residual = original_image - background_map
        
        # Compute statistics
        stats = {
            'global_rms': np.median(rms_map[valid_mask]),
            'background_mean': np.mean(background_map[valid_mask]),
            'background_median': np.median(background_map[valid_mask]),
            'background_std': np.std(background_map[valid_mask]),
            'residual_mean': np.mean(residual[valid_mask]),
            'residual_median': np.median(residual[valid_mask]),
            'residual_std': np.std(residual[valid_mask]),
            'valid_pixel_fraction': np.sum(valid_mask) / valid_mask.size,
            'background_range': (np.min(background_map[valid_mask]), 
                               np.max(background_map[valid_mask])),
            'rms_range': (np.min(rms_map[valid_mask]), 
                         np.max(rms_map[valid_mask]))
        }
        
        return stats
    
    def plot_background_diagnostics(self, 
                                  original_image: np.ndarray,
                                  background_map: np.ndarray,
                                  rms_map: np.ndarray,
                                  mask: np.ndarray,
                                  output_path: Optional[str] = None) -> None:
        """
        Create diagnostic plots for background estimation.
        
        Parameters:
        -----------
        original_image : numpy.ndarray
            Original image
        background_map : numpy.ndarray
            Estimated background
        rms_map : numpy.ndarray
            RMS map
        mask : numpy.ndarray
            Mask used
        output_path : str, optional
            Path to save the plot
        """
        try:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # Original image
            im1 = axes[0, 0].imshow(original_image, origin='lower', cmap='viridis')
            axes[0, 0].set_title('Original Image')
            plt.colorbar(im1, ax=axes[0, 0])
            
            # Background map
            im2 = axes[0, 1].imshow(background_map, origin='lower', cmap='viridis')
            axes[0, 1].set_title('Background Map')
            plt.colorbar(im2, ax=axes[0, 1])
            
            # Background-subtracted
            residual = original_image - background_map
            im3 = axes[0, 2].imshow(residual, origin='lower', cmap='viridis')
            axes[0, 2].set_title('Background Subtracted')
            plt.colorbar(im3, ax=axes[0, 2])
            
            # RMS map
            im4 = axes[1, 0].imshow(rms_map, origin='lower', cmap='plasma')
            axes[1, 0].set_title('RMS Map')
            plt.colorbar(im4, ax=axes[1, 0])
            
            # Mask
            axes[1, 1].imshow(mask.astype(int), origin='lower', cmap='gray')
            axes[1, 1].set_title('Final Mask')
            
            # Background histogram
            valid_mask = ~mask & np.isfinite(background_map)
            axes[1, 2].hist(background_map[valid_mask], bins=50, alpha=0.7, density=True)
            axes[1, 2].set_xlabel('Background Value')
            axes[1, 2].set_ylabel('Density')
            axes[1, 2].set_title('Background Distribution')
            
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                self.logger.info(f"Background diagnostics saved to {output_path}")
            else:
                plt.show()
            
            plt.close()
            
        except Exception as e:
            self.logger.warning(f"Failed to create diagnostic plots: {e}")


# Convenience function for simple background estimation
def estimate_background(image: np.ndarray,
                       mask: Optional[np.ndarray] = None,
                       config: Optional[BackgroundConfig] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function for background estimation.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image
    mask : numpy.ndarray, optional
        Bad pixel mask
    config : BackgroundConfig, optional
        Configuration parameters
        
    Returns:
    --------
    tuple
        Background map and RMS map
    """
    estimator = BackgroundEstimator(config)
    background_map, rms_map, _ = estimator.estimate_background(image, mask)
    return background_map, rms_map
