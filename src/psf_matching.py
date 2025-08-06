"""
PSF Matching and Homogenization Module for JWST Photometry

This module implements sophisticated PSF matching capabilities using Pypher
and other advanced techniques for homogenizing PSFs across different bands
or observations.
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple, List, Union
from dataclasses import dataclass, field
from pathlib import Path

from astropy.io import fits
from astropy.table import Table
from astropy.convolution import Gaussian2DKernel, convolve
from scipy import ndimage, optimize
from scipy.signal import convolve2d, wiener
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

try:
    import pypher
    PYPHER_AVAILABLE = True
except ImportError:
    PYPHER_AVAILABLE = False
    logging.warning("Pypher not available - PSF matching functionality will be limited")


@dataclass
class PSFMatchingConfig:
    """Configuration parameters for PSF matching."""
    
    # General matching parameters
    matching_method: str = "pypher"  # "pypher", "gaussian_approximation", "kernel_regression"
    target_fwhm: Optional[float] = None  # If None, uses worst-seeing PSF
    kernel_size: int = 15
    
    # Pypher-specific parameters
    regularization_parameter: float = 1e-6
    noise_power: Optional[float] = None
    fft_shape: Optional[Tuple[int, int]] = None
    
    # Gaussian approximation parameters
    gaussian_sigma_factor: float = 2.355  # Convert FWHM to sigma
    gaussian_truncate: float = 4.0
    
    # Kernel regression parameters
    kernel_regression_order: int = 2
    kernel_regression_radius: int = 7
    
    # Quality assessment
    assess_quality: bool = True
    quality_metrics: List[str] = field(default_factory=lambda: [
        'flux_conservation', 'noise_amplification', 'residual_power', 'psf_similarity'
    ])
    
    # Spatial variation handling
    handle_spatial_variation: bool = True
    spatial_grid_size: int = 3
    min_overlap_fraction: float = 0.5
    
    # Processing options
    normalize_kernels: bool = True
    apply_noise_correlation: bool = False
    preserve_flux: bool = True
    
    # Output control
    save_kernels: bool = True
    create_diagnostics: bool = True
    verbose: bool = True


@dataclass
class PSFMatchingResults:
    """Container for PSF matching results."""
    
    # Matching kernels
    matching_kernels: Dict[str, np.ndarray]
    target_psf: np.ndarray
    target_fwhm: float
    
    # Matched PSFs
    matched_psfs: Dict[str, np.ndarray]
    
    # Quality assessment
    quality_metrics: Dict[str, Dict[str, float]]
    flux_conservation: Dict[str, float]
    noise_amplification: Dict[str, float]
    
    # Spatial variation information
    spatial_variation: Optional[Dict[str, Any]] = None
    
    # Diagnostic information
    diagnostics: Dict[str, Any] = field(default_factory=dict)


class AdvancedPSFMatcher:
    """
    Advanced PSF matching processor for JWST observations.
    
    This class provides sophisticated PSF matching capabilities including:
    - Pypher-based PSF matching with optimal regularization
    - Gaussian approximation methods for fast processing
    - Kernel regression for complex PSF shapes
    - Spatial variation handling across the field
    - Comprehensive quality assessment
    - Noise correlation preservation
    """
    
    def __init__(self, config: Optional[PSFMatchingConfig] = None):
        """
        Initialize the PSF matcher.
        
        Parameters:
        -----------
        config : PSFMatchingConfig, optional
            PSF matching configuration. If None, uses defaults.
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or PSFMatchingConfig()
        
        if not PYPHER_AVAILABLE and self.config.matching_method == "pypher":
            self.logger.warning("Pypher not available, switching to Gaussian approximation")
            self.config.matching_method = "gaussian_approximation"
    
    def match_psfs(self, 
                   psfs: Dict[str, np.ndarray],
                   psf_properties: Optional[Dict[str, Dict[str, float]]] = None,
                   target_band: Optional[str] = None) -> PSFMatchingResults:
        """
        Match PSFs to a common target PSF.
        
        Parameters:
        -----------
        psfs : dict
            Dictionary of PSFs {band_name: psf_array}
        psf_properties : dict, optional
            Dictionary of PSF properties {band_name: {property: value}}
        target_band : str, optional
            Target band for matching. If None, uses worst-seeing band.
            
        Returns:
        --------
        PSFMatchingResults
            Complete PSF matching results
        """
        self.logger.info("Starting PSF matching")
        
        # Validate inputs
        self._validate_psfs(psfs)
        
        # Determine target PSF
        target_psf, target_fwhm = self._determine_target_psf(
            psfs, psf_properties, target_band
        )
        
        self.logger.info(f"Target FWHM: {target_fwhm:.2f} pixels")
        
        # Generate matching kernels
        matching_kernels = {}
        matched_psfs = {}
        quality_metrics = {}
        flux_conservation = {}
        noise_amplification = {}
        
        for band_name, psf in psfs.items():
            self.logger.debug(f"Processing band: {band_name}")
            
            try:
                # Generate matching kernel
                kernel = self._generate_matching_kernel(psf, target_psf, band_name)
                matching_kernels[band_name] = kernel
                
                # Apply kernel to create matched PSF
                matched_psf = self._apply_kernel(psf, kernel)
                matched_psfs[band_name] = matched_psf
                
                # Assess quality if requested
                if self.config.assess_quality:
                    quality = self._assess_matching_quality(
                        psf, matched_psf, target_psf, kernel
                    )
                    quality_metrics[band_name] = quality
                    flux_conservation[band_name] = quality.get('flux_conservation', 1.0)
                    noise_amplification[band_name] = quality.get('noise_amplification', 1.0)
                
            except Exception as e:
                self.logger.error(f"PSF matching failed for band {band_name}: {e}")
                # Create identity kernel as fallback
                kernel_size = self.config.kernel_size
                kernel = np.zeros((kernel_size, kernel_size))
                kernel[kernel_size//2, kernel_size//2] = 1.0
                matching_kernels[band_name] = kernel
                matched_psfs[band_name] = psf.copy()
                quality_metrics[band_name] = {}
                flux_conservation[band_name] = 1.0
                noise_amplification[band_name] = 1.0
        
        # Handle spatial variation if requested
        spatial_variation = None
        if self.config.handle_spatial_variation:
            spatial_variation = self._model_spatial_kernel_variation(
                matching_kernels, psfs
            )
        
        # Create diagnostics
        diagnostics = self._create_matching_diagnostics(
            psfs, matched_psfs, matching_kernels, target_psf
        )
        
        results = PSFMatchingResults(
            matching_kernels=matching_kernels,
            target_psf=target_psf,
            target_fwhm=target_fwhm,
            matched_psfs=matched_psfs,
            quality_metrics=quality_metrics,
            flux_conservation=flux_conservation,
            noise_amplification=noise_amplification,
            spatial_variation=spatial_variation,
            diagnostics=diagnostics
        )
        
        self.logger.info("PSF matching completed successfully")
        return results
    
    def apply_matching_to_images(self, 
                               images: Dict[str, np.ndarray],
                               matching_results: PSFMatchingResults,
                               preserve_flux: bool = True) -> Dict[str, np.ndarray]:
        """
        Apply PSF matching kernels to images.
        
        Parameters:
        -----------
        images : dict
            Dictionary of images {band_name: image_array}
        matching_results : PSFMatchingResults
            PSF matching results containing kernels
        preserve_flux : bool
            Whether to preserve total flux in matched images
            
        Returns:
        --------
        dict
            Dictionary of PSF-matched images
        """
        self.logger.info("Applying PSF matching to images")
        
        matched_images = {}
        
        for band_name, image in images.items():
            if band_name not in matching_results.matching_kernels:
                self.logger.warning(f"No matching kernel for band {band_name}")
                matched_images[band_name] = image.copy()
                continue
            
            try:
                kernel = matching_results.matching_kernels[band_name]
                
                # Apply kernel
                matched_image = self._apply_kernel(image, kernel)
                
                # Preserve flux if requested
                if preserve_flux:
                    flux_ratio = matching_results.flux_conservation.get(band_name, 1.0)
                    if flux_ratio > 0:
                        matched_image = matched_image / flux_ratio
                
                matched_images[band_name] = matched_image
                
            except Exception as e:
                self.logger.error(f"Failed to apply PSF matching to {band_name}: {e}")
                matched_images[band_name] = image.copy()
        
        return matched_images
    
    def _validate_psfs(self, psfs: Dict[str, np.ndarray]) -> None:
        """Validate PSF dictionary."""
        if not psfs:
            raise ValueError("No PSFs provided")
        
        # Check that all PSFs have the same shape
        shapes = [psf.shape for psf in psfs.values()]
        if len(set(shapes)) > 1:
            raise ValueError(f"PSFs have different shapes: {shapes}")
        
        # Check that PSFs are 2D
        for band_name, psf in psfs.items():
            if psf.ndim != 2:
                raise ValueError(f"PSF for band {band_name} is not 2D")
            
            if not np.isfinite(psf).all():
                raise ValueError(f"PSF for band {band_name} contains non-finite values")
    
    def _determine_target_psf(self, 
                            psfs: Dict[str, np.ndarray],
                            psf_properties: Optional[Dict[str, Dict[str, float]]],
                            target_band: Optional[str]) -> Tuple[np.ndarray, float]:
        """
        Determine the target PSF for matching.
        
        Parameters:
        -----------
        psfs : dict
            Dictionary of PSFs
        psf_properties : dict, optional
            PSF properties
        target_band : str, optional
            Specified target band
            
        Returns:
        --------
        tuple
            (target_psf, target_fwhm)
        """
        if target_band is not None and target_band in psfs:
            target_psf = psfs[target_band]
            if psf_properties and target_band in psf_properties:
                target_fwhm = psf_properties[target_band].get('fwhm', 3.0)
            else:
                target_fwhm = self._measure_psf_fwhm(target_psf)
            return target_psf, target_fwhm
        
        if self.config.target_fwhm is not None:
            # Create Gaussian target PSF with specified FWHM
            target_fwhm = self.config.target_fwhm
            sigma = target_fwhm / self.config.gaussian_sigma_factor
            
            # Use the same shape as input PSFs
            psf_shape = list(psfs.values())[0].shape
            center = ((psf_shape[0] - 1) / 2, (psf_shape[1] - 1) / 2)
            
            y, x = np.ogrid[:psf_shape[0], :psf_shape[1]]
            target_psf = np.exp(-((x - center[1])**2 + (y - center[0])**2) / (2 * sigma**2))
            target_psf = target_psf / np.sum(target_psf)
            
            return target_psf, target_fwhm
        
        # Find the worst-seeing PSF (largest FWHM)
        if psf_properties:
            fwhms = {band: props.get('fwhm', 3.0) 
                    for band, props in psf_properties.items() 
                    if band in psfs}
        else:
            fwhms = {band: self._measure_psf_fwhm(psf) 
                    for band, psf in psfs.items()}
        
        target_band = max(fwhms.keys(), key=lambda x: fwhms[x])
        target_psf = psfs[target_band]
        target_fwhm = fwhms[target_band]
        
        self.logger.info(f"Selected {target_band} as target band (FWHM: {target_fwhm:.2f})")
        return target_psf, target_fwhm
    
    def _measure_psf_fwhm(self, psf: np.ndarray) -> float:
        """
        Measure FWHM of a PSF.
        
        Parameters:
        -----------
        psf : numpy.ndarray
            PSF array
            
        Returns:
        --------
        float
            FWHM in pixels
        """
        # Calculate second moments
        y, x = np.ogrid[:psf.shape[0], :psf.shape[1]]
        
        total_flux = np.sum(psf)
        x_center = np.sum(x * psf) / total_flux
        y_center = np.sum(y * psf) / total_flux
        
        m_xx = np.sum((x - x_center)**2 * psf) / total_flux
        m_yy = np.sum((y - y_center)**2 * psf) / total_flux
        
        fwhm = 2.0 * np.sqrt(np.log(2) * (m_xx + m_yy))
        return fwhm
    
    def _generate_matching_kernel(self, 
                                source_psf: np.ndarray,
                                target_psf: np.ndarray,
                                band_name: str) -> np.ndarray:
        """
        Generate PSF matching kernel.
        
        Parameters:
        -----------
        source_psf : numpy.ndarray
            Source PSF to be matched
        target_psf : numpy.ndarray
            Target PSF
        band_name : str
            Band name for logging
            
        Returns:
        --------
        numpy.ndarray
            Matching kernel
        """
        if self.config.matching_method == "pypher":
            return self._generate_pypher_kernel(source_psf, target_psf, band_name)
        elif self.config.matching_method == "gaussian_approximation":
            return self._generate_gaussian_kernel(source_psf, target_psf, band_name)
        elif self.config.matching_method == "kernel_regression":
            return self._generate_regression_kernel(source_psf, target_psf, band_name)
        else:
            raise ValueError(f"Unknown matching method: {self.config.matching_method}")
    
    def _generate_pypher_kernel(self, 
                              source_psf: np.ndarray,
                              target_psf: np.ndarray,
                              band_name: str) -> np.ndarray:
        """
        Generate PSF matching kernel using Pypher.
        
        Parameters:
        -----------
        source_psf : numpy.ndarray
            Source PSF
        target_psf : numpy.ndarray
            Target PSF
        band_name : str
            Band name for logging
            
        Returns:
        --------
        numpy.ndarray
            Pypher matching kernel
        """
        if not PYPHER_AVAILABLE:
            raise RuntimeError("Pypher is not available")
        
        try:
            # Prepare PSFs for Pypher
            psf1 = source_psf / np.sum(source_psf)
            psf2 = target_psf / np.sum(target_psf)
            
            # Generate kernel
            kernel = pypher.psf_match(
                psf1, psf2, 
                reg_fact=self.config.regularization_parameter,
                noise_power=self.config.noise_power,
                fft_shape=self.config.fft_shape
            )
            
            # Normalize kernel if requested
            if self.config.normalize_kernels:
                kernel = kernel / np.sum(kernel)
            
            return kernel
            
        except Exception as e:
            self.logger.error(f"Pypher kernel generation failed for {band_name}: {e}")
            raise
    
    def _generate_gaussian_kernel(self, 
                                source_psf: np.ndarray,
                                target_psf: np.ndarray,
                                band_name: str) -> np.ndarray:
        """
        Generate PSF matching kernel using Gaussian approximation.
        
        Parameters:
        -----------
        source_psf : numpy.ndarray
            Source PSF
        target_psf : numpy.ndarray
            Target PSF
        band_name : str
            Band name for logging
            
        Returns:
        --------
        numpy.ndarray
            Gaussian matching kernel
        """
        try:
            # Measure FWHMs
            source_fwhm = self._measure_psf_fwhm(source_psf)
            target_fwhm = self._measure_psf_fwhm(target_psf)
            
            # Calculate required convolution kernel FWHM
            if target_fwhm <= source_fwhm:
                # Cannot improve resolution, return delta function
                kernel = np.zeros((self.config.kernel_size, self.config.kernel_size))
                kernel[self.config.kernel_size//2, self.config.kernel_size//2] = 1.0
                return kernel
            
            # Calculate kernel FWHM
            kernel_fwhm = np.sqrt(target_fwhm**2 - source_fwhm**2)
            kernel_sigma = kernel_fwhm / self.config.gaussian_sigma_factor
            
            # Create Gaussian kernel
            kernel = Gaussian2DKernel(
                x_stddev=kernel_sigma,
                y_stddev=kernel_sigma,
                x_size=self.config.kernel_size,
                y_size=self.config.kernel_size
            ).array
            
            # Normalize
            kernel = kernel / np.sum(kernel)
            
            return kernel
            
        except Exception as e:
            self.logger.error(f"Gaussian kernel generation failed for {band_name}: {e}")
            raise
    
    def _generate_regression_kernel(self, 
                                  source_psf: np.ndarray,
                                  target_psf: np.ndarray,
                                  band_name: str) -> np.ndarray:
        """
        Generate PSF matching kernel using kernel regression.
        
        Parameters:
        -----------
        source_psf : numpy.ndarray
            Source PSF
        target_psf : numpy.ndarray
            Target PSF
        band_name : str
            Band name for logging
            
        Returns:
        --------
        numpy.ndarray
            Regression-based matching kernel
        """
        try:
            # This is a simplified kernel regression approach
            # In practice, more sophisticated methods could be implemented
            
            # Deconvolve target PSF by source PSF using Wiener filter
            # This is an approximation to the true deconvolution kernel
            
            # FFT-based approach
            source_fft = np.fft.fft2(source_psf, s=target_psf.shape)
            target_fft = np.fft.fft2(target_psf, s=target_psf.shape)
            
            # Wiener deconvolution with noise estimate
            noise_power = 0.01  # Simple noise estimate
            kernel_fft = target_fft * np.conj(source_fft) / (np.abs(source_fft)**2 + noise_power)
            
            # Convert back to spatial domain
            kernel_full = np.real(np.fft.ifft2(kernel_fft))
            kernel_full = np.fft.fftshift(kernel_full)
            
            # Extract central portion
            center = (kernel_full.shape[0] // 2, kernel_full.shape[1] // 2)
            half_size = self.config.kernel_size // 2
            
            kernel = kernel_full[
                center[0] - half_size:center[0] + half_size + 1,
                center[1] - half_size:center[1] + half_size + 1
            ]
            
            # Ensure proper size
            if kernel.shape != (self.config.kernel_size, self.config.kernel_size):
                # Resize if necessary
                from scipy.ndimage import zoom
                zoom_factors = (
                    self.config.kernel_size / kernel.shape[0],
                    self.config.kernel_size / kernel.shape[1]
                )
                kernel = zoom(kernel, zoom_factors)
            
            # Normalize
            if np.sum(kernel) > 0:
                kernel = kernel / np.sum(kernel)
            else:
                # Fallback to delta function
                kernel = np.zeros((self.config.kernel_size, self.config.kernel_size))
                kernel[self.config.kernel_size//2, self.config.kernel_size//2] = 1.0
            
            return kernel
            
        except Exception as e:
            self.logger.error(f"Regression kernel generation failed for {band_name}: {e}")
            raise
    
    def _apply_kernel(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        Apply convolution kernel to image.
        
        Parameters:
        -----------
        image : numpy.ndarray
            Input image
        kernel : numpy.ndarray
            Convolution kernel
            
        Returns:
        --------
        numpy.ndarray
            Convolved image
        """
        return convolve2d(image, kernel, mode='same', boundary='symm')
    
    def _assess_matching_quality(self, 
                               original_psf: np.ndarray,
                               matched_psf: np.ndarray,
                               target_psf: np.ndarray,
                               kernel: np.ndarray) -> Dict[str, float]:
        """
        Assess the quality of PSF matching.
        
        Parameters:
        -----------
        original_psf : numpy.ndarray
            Original PSF
        matched_psf : numpy.ndarray
            Matched PSF
        target_psf : numpy.ndarray
            Target PSF
        kernel : numpy.ndarray
            Matching kernel
            
        Returns:
        --------
        dict
            Quality metrics
        """
        quality = {}
        
        try:
            # Flux conservation
            original_flux = np.sum(original_psf)
            matched_flux = np.sum(matched_psf)
            quality['flux_conservation'] = matched_flux / original_flux if original_flux > 0 else 1.0
            
            # Noise amplification (sum of squared kernel values)
            quality['noise_amplification'] = np.sqrt(np.sum(kernel**2))
            
            # PSF similarity (correlation with target)
            if matched_psf.shape == target_psf.shape:
                correlation = np.corrcoef(matched_psf.ravel(), target_psf.ravel())[0, 1]
                quality['psf_similarity'] = correlation if np.isfinite(correlation) else 0.0
            
            # Residual power (how well the matching worked)
            residual = matched_psf - target_psf
            quality['residual_power'] = np.sum(residual**2) / np.sum(target_psf**2)
            
            # FWHM comparison
            matched_fwhm = self._measure_psf_fwhm(matched_psf)
            target_fwhm = self._measure_psf_fwhm(target_psf)
            quality['fwhm_ratio'] = matched_fwhm / target_fwhm if target_fwhm > 0 else 1.0
            
        except Exception as e:
            self.logger.warning(f"Quality assessment failed: {e}")
        
        return quality
    
    def _model_spatial_kernel_variation(self, 
                                      matching_kernels: Dict[str, np.ndarray],
                                      psfs: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Model spatial variation of matching kernels.
        
        Parameters:
        -----------
        matching_kernels : dict
            Dictionary of matching kernels
        psfs : dict
            Dictionary of PSFs
            
        Returns:
        --------
        dict
            Spatial variation model
        """
        # This is a placeholder for spatial variation modeling
        # In a full implementation, this would analyze how the kernels
        # need to vary across the field of view
        
        spatial_model = {
            'method': 'uniform',
            'variation_amplitude': 0.0,
            'description': 'Uniform kernel across field (no spatial variation modeled)'
        }
        
        return spatial_model
    
    def _create_matching_diagnostics(self, 
                                   original_psfs: Dict[str, np.ndarray],
                                   matched_psfs: Dict[str, np.ndarray],
                                   kernels: Dict[str, np.ndarray],
                                   target_psf: np.ndarray) -> Dict[str, Any]:
        """
        Create diagnostic information for PSF matching.
        
        Parameters:
        -----------
        original_psfs : dict
            Original PSFs
        matched_psfs : dict
            Matched PSFs
        kernels : dict
            Matching kernels
        target_psf : numpy.ndarray
            Target PSF
            
        Returns:
        --------
        dict
            Diagnostic information
        """
        diagnostics = {
            'n_bands_processed': len(original_psfs),
            'matching_method': self.config.matching_method,
            'target_psf_shape': target_psf.shape,
            'kernel_sizes': {band: kernel.shape for band, kernel in kernels.items()}
        }
        
        # Calculate FWHM statistics
        original_fwhms = {band: self._measure_psf_fwhm(psf) 
                         for band, psf in original_psfs.items()}
        matched_fwhms = {band: self._measure_psf_fwhm(psf) 
                        for band, psf in matched_psfs.items()}
        
        diagnostics['original_fwhms'] = original_fwhms
        diagnostics['matched_fwhms'] = matched_fwhms
        diagnostics['target_fwhm'] = self._measure_psf_fwhm(target_psf)
        
        return diagnostics
    
    def plot_matching_diagnostics(self, 
                                results: PSFMatchingResults,
                                original_psfs: Dict[str, np.ndarray],
                                output_path: Optional[str] = None) -> None:
        """
        Create diagnostic plots for PSF matching.
        
        Parameters:
        -----------
        results : PSFMatchingResults
            PSF matching results
        original_psfs : dict
            Original PSFs before matching
        output_path : str, optional
            Path to save the plot
        """
        try:
            n_bands = len(original_psfs)
            fig, axes = plt.subplots(3, min(n_bands, 4), figsize=(16, 12))
            
            if n_bands == 1:
                axes = axes.reshape(-1, 1)
            
            band_names = list(original_psfs.keys())[:4]  # Limit to 4 bands for display
            
            for i, band_name in enumerate(band_names):
                col = i % 4
                
                # Original PSF
                im1 = axes[0, col].imshow(original_psfs[band_name], origin='lower', 
                                        cmap='viridis')
                axes[0, col].set_title(f'{band_name}\\nOriginal PSF')
                plt.colorbar(im1, ax=axes[0, col])
                
                # Matched PSF
                if band_name in results.matched_psfs:
                    im2 = axes[1, col].imshow(results.matched_psfs[band_name], 
                                            origin='lower', cmap='viridis')
                    axes[1, col].set_title(f'{band_name}\\nMatched PSF')
                    plt.colorbar(im2, ax=axes[1, col])
                
                # Matching kernel
                if band_name in results.matching_kernels:
                    im3 = axes[2, col].imshow(results.matching_kernels[band_name], 
                                            origin='lower', cmap='RdBu')
                    axes[2, col].set_title(f'{band_name}\\nMatching Kernel')
                    plt.colorbar(im3, ax=axes[2, col])
            
            # Remove empty subplots
            for j in range(len(band_names), 4):
                for row in range(3):
                    axes[row, j].remove()
            
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                self.logger.info(f"PSF matching diagnostics saved to {output_path}")
            else:
                plt.show()
            
            plt.close()
            
            # Create quality metrics plot
            if results.quality_metrics:
                self._plot_quality_metrics(results)
            
        except Exception as e:
            self.logger.warning(f"Failed to create diagnostic plots: {e}")
    
    def _plot_quality_metrics(self, results: PSFMatchingResults) -> None:
        """Plot quality metrics."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            band_names = list(results.quality_metrics.keys())
            
            # Flux conservation
            flux_values = [results.flux_conservation.get(band, 1.0) for band in band_names]
            axes[0, 0].bar(band_names, flux_values)
            axes[0, 0].set_title('Flux Conservation')
            axes[0, 0].set_ylabel('Flux Ratio')
            axes[0, 0].axhline(y=1.0, color='red', linestyle='--')
            
            # Noise amplification
            noise_values = [results.noise_amplification.get(band, 1.0) for band in band_names]
            axes[0, 1].bar(band_names, noise_values)
            axes[0, 1].set_title('Noise Amplification')
            axes[0, 1].set_ylabel('Amplification Factor')
            
            # PSF similarity
            similarity_values = [results.quality_metrics[band].get('psf_similarity', 0.0) 
                               for band in band_names if band in results.quality_metrics]
            if similarity_values:
                axes[1, 0].bar(band_names[:len(similarity_values)], similarity_values)
                axes[1, 0].set_title('PSF Similarity')
                axes[1, 0].set_ylabel('Correlation')
            
            # FWHM comparison
            original_fwhms = [results.diagnostics['original_fwhms'].get(band, 0) 
                            for band in band_names]
            matched_fwhms = [results.diagnostics['matched_fwhms'].get(band, 0) 
                           for band in band_names]
            
            x_pos = np.arange(len(band_names))
            width = 0.35
            
            axes[1, 1].bar(x_pos - width/2, original_fwhms, width, label='Original')
            axes[1, 1].bar(x_pos + width/2, matched_fwhms, width, label='Matched')
            axes[1, 1].axhline(y=results.target_fwhm, color='red', linestyle='--', 
                             label='Target')
            axes[1, 1].set_title('FWHM Comparison')
            axes[1, 1].set_ylabel('FWHM (pixels)')
            axes[1, 1].set_xticks(x_pos)
            axes[1, 1].set_xticklabels(band_names)
            axes[1, 1].legend()
            
            plt.tight_layout()
            plt.show()
            plt.close()
            
        except Exception as e:
            self.logger.warning(f"Failed to create quality metrics plot: {e}")


# Convenience functions
def match_psf_simple(source_psf: np.ndarray, 
                    target_psf: np.ndarray,
                    method: str = "pypher",
                    regularization: float = 1e-6) -> np.ndarray:
    """
    Simple PSF matching function.
    
    Parameters:
    -----------
    source_psf : numpy.ndarray
        Source PSF to be matched
    target_psf : numpy.ndarray
        Target PSF
    method : str
        Matching method
    regularization : float
        Regularization parameter
        
    Returns:
    --------
    numpy.ndarray
        Matching kernel
    """
    config = PSFMatchingConfig(
        matching_method=method,
        regularization_parameter=regularization
    )
    
    matcher = AdvancedPSFMatcher(config)
    psfs = {'source': source_psf, 'target': target_psf}
    
    results = matcher.match_psfs(psfs, target_band='target')
    return results.matching_kernels['source']


def apply_psf_matching(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Apply PSF matching kernel to image.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image
    kernel : numpy.ndarray
        Matching kernel
        
    Returns:
    --------
    numpy.ndarray
        PSF-matched image
    """
    return convolve2d(image, kernel, mode='same', boundary='symm')
