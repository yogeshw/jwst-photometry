"""
JWST photometry comprehensive uncertainty estimation module (Phase 4.3).

This module provides advanced uncertainty estimation for JWST NIRCam photometry,
including statistical uncertainties, systematic errors, correlated noise modeling,
and Monte Carlo error propagation.
"""

import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import time

# Core scientific libraries
from scipy import stats, ndimage, optimize
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
import astropy.units as u
from astropy.io import fits
from astropy.table import Table

# Specialized imports with fallbacks
try:
    import sep
except ImportError:
    print("Warning: sep not available for some advanced features")
    sep = None

# Local imports with fallbacks
try:
    from .utils import setup_logger, memory_monitor, validate_array
except ImportError:
    def setup_logger(name): return logging.getLogger(name)
    def memory_monitor(func): return func
    def validate_array(arr): return arr is not None

import numpy as np
import logging
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass, field
from pathlib import Path

from scipy import stats, linalg
from scipy.ndimage import gaussian_filter, uniform_filter
from astropy.io import fits
from astropy.table import Table
from astropy import units as u
from astropy.stats import sigma_clip
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Import our modules
from .utils import setup_logger, memory_monitor, validate_array


@dataclass
class ErrorEstimationConfig:
    """Configuration parameters for error estimation."""
    
    # Statistical uncertainties
    include_poisson_noise: bool = True
    include_readnoise: bool = True
    include_dark_current: bool = True
    effective_gain: float = 2.0  # e-/DN
    read_noise: float = 5.0  # e- RMS
    dark_current: float = 0.01  # e-/s
    
    # Background estimation uncertainties
    background_uncertainty_method: str = "local_rms"  # "local_rms", "global_rms", "model_based"
    background_correlation_length: float = 10.0  # pixels
    n_background_realizations: int = 100
    
    # Systematic uncertainties
    flat_field_uncertainty: float = 0.005  # fractional
    photometric_repeatability: float = 0.01  # fractional
    detector_nonlinearity: float = 0.002  # fractional
    charge_diffusion: float = 0.001  # fractional
    
    # Calibration uncertainties
    zeropoint_uncertainty: float = 0.02  # mag
    aperture_correction_uncertainty: float = 0.01  # fractional
    color_correction_uncertainty: float = 0.005  # fractional
    
    # Crowding and confusion uncertainties
    include_crowding_uncertainty: bool = True
    crowding_analysis_radius: float = 5.0  # pixels
    confusion_limit_estimation: bool = True
    
    # Correlated noise modeling
    model_correlated_noise: bool = True
    correlation_function_type: str = "exponential"  # "exponential", "gaussian", "power_law"
    spatial_correlation_scale: float = 2.0  # pixels
    
    # Cross-band correlation
    estimate_cross_band_correlation: bool = True
    cross_band_correlation_method: str = "empirical"  # "empirical", "theoretical"
    
    # Monte Carlo error propagation
    use_monte_carlo_propagation: bool = True
    n_monte_carlo_samples: int = 1000
    monte_carlo_seed: int = 42
    
    # PSF uncertainties
    psf_uncertainty_fraction: float = 0.02  # fractional uncertainty in PSF model
    psf_size_uncertainty: float = 0.05  # fractional uncertainty in PSF size
    
    # Atmospheric and environmental (for validation with ground-based data)
    include_atmospheric_uncertainty: bool = False
    atmospheric_stability: float = 0.02  # fractional
    
    # Quality control
    flag_high_uncertainty_sources: bool = True
    uncertainty_threshold: float = 0.3  # fractional
    detect_systematic_patterns: bool = True
    
    # Output options
    save_uncertainty_maps: bool = True
    create_correlation_matrices: bool = True
    generate_uncertainty_diagnostics: bool = True


@dataclass
class SourceUncertainties:
    """Container for comprehensive source uncertainties."""
    
    id: int
    
    # Statistical uncertainties (by aperture)
    poisson_errors: Dict[float, float] = field(default_factory=dict)
    background_errors: Dict[float, float] = field(default_factory=dict)
    readnoise_errors: Dict[float, float] = field(default_factory=dict)
    
    # Systematic uncertainties
    flat_field_error: float = 0.0
    calibration_error: float = 0.0
    aperture_correction_error: float = 0.0
    psf_model_error: float = 0.0
    
    # Crowding uncertainties
    crowding_error: float = 0.0
    confusion_error: float = 0.0
    neighbor_contamination: float = 0.0
    
    # Correlated noise contribution
    correlated_noise_error: Dict[float, float] = field(default_factory=dict)
    spatial_correlation_matrix: Optional[np.ndarray] = None
    
    # Total uncertainties (by aperture)
    total_statistical_errors: Dict[float, float] = field(default_factory=dict)
    total_systematic_errors: Dict[float, float] = field(default_factory=dict)
    total_errors: Dict[float, float] = field(default_factory=dict)
    
    # Cross-band correlations
    cross_band_correlations: Dict[str, float] = field(default_factory=dict)
    
    # Quality metrics
    uncertainty_flags: List[str] = field(default_factory=list)
    signal_to_noise: Dict[float, float] = field(default_factory=dict)
    
    # Monte Carlo results
    monte_carlo_uncertainties: Dict[float, np.ndarray] = field(default_factory=dict)
    monte_carlo_statistics: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass
class UncertaintyResults:
    """Container for comprehensive uncertainty analysis results."""
    
    source_uncertainties: List[SourceUncertainties]
    config: ErrorEstimationConfig
    
    # Global uncertainty maps
    uncertainty_maps: Dict[str, np.ndarray] = field(default_factory=dict)
    correlation_maps: Dict[str, np.ndarray] = field(default_factory=dict)
    
    # Statistical summaries
    uncertainty_statistics: Dict[str, Any] = field(default_factory=dict)
    
    # Correlation analysis
    cross_band_correlation_matrix: Optional[np.ndarray] = None
    spatial_correlation_function: Optional[np.ndarray] = None
    
    # Systematic patterns
    systematic_patterns: Dict[str, Any] = field(default_factory=dict)
    
    # Processing information
    processing_time: float = 0.0
    memory_usage: float = 0.0
    
    # Quality assessment
    overall_uncertainty_quality: float = 1.0
    uncertainty_completeness: float = 1.0


class ComprehensiveErrorEstimator:
    """
    Comprehensive error estimation processor for JWST photometry.
    
    This class provides sophisticated uncertainty analysis including:
    - Statistical noise components (Poisson, read noise, dark current)
    - Systematic uncertainties (flat field, calibration, PSF modeling)
    - Correlated noise modeling
    - Crowding and confusion uncertainties
    - Cross-band correlation analysis
    - Monte Carlo error propagation
    """
    
    def __init__(self, config: Optional[ErrorEstimationConfig] = None):
        """
        Initialize the error estimator.
        
        Parameters:
        -----------
        config : ErrorEstimationConfig, optional
            Error estimation configuration. If None, uses defaults.
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or ErrorEstimationConfig()
        
        # Set random seed for reproducibility
        np.random.seed(self.config.monte_carlo_seed)
    
    def estimate_uncertainties(self,
                             images: Dict[str, np.ndarray],
                             photometry_results: Dict[str, Any],
                             calibration_results: Optional[Dict[str, Any]] = None,
                             psf_models: Optional[Dict[str, np.ndarray]] = None,
                             background_maps: Optional[Dict[str, np.ndarray]] = None,
                             rms_maps: Optional[Dict[str, np.ndarray]] = None) -> UncertaintyResults:
        """
        Perform comprehensive uncertainty estimation.
        
        Parameters:
        -----------
        images : dict
            Input images for each band
        photometry_results : dict
            Photometry results for each band
        calibration_results : dict, optional
            Calibration results
        psf_models : dict, optional
            PSF models for each band
        background_maps : dict, optional
            Background maps for each band
        rms_maps : dict, optional
            RMS/noise maps for each band
            
        Returns:
        --------
        UncertaintyResults
            Comprehensive uncertainty analysis results
        """
        self.logger.info("Starting comprehensive uncertainty estimation")
        
        import time
        try:
            import psutil
            initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            initial_memory = 0.0  # Fallback if psutil not available
        start_time = time.time()
        
        # Validate inputs
        self._validate_uncertainty_inputs(images, photometry_results)
        
        # Prepare source uncertainty list
        source_uncertainties = self._prepare_source_uncertainties(photometry_results)
        
        # Estimate statistical uncertainties
        self._estimate_statistical_uncertainties(
            source_uncertainties, images, photometry_results, background_maps, rms_maps
        )
        
        # Estimate systematic uncertainties
        self._estimate_systematic_uncertainties(
            source_uncertainties, calibration_results, psf_models
        )
        
        # Model correlated noise
        if self.config.model_correlated_noise:
            self._model_correlated_noise(source_uncertainties, images, photometry_results)
        
        # Estimate crowding uncertainties
        if self.config.include_crowding_uncertainty:
            self._estimate_crowding_uncertainties(source_uncertainties, images, photometry_results)
        
        # Cross-band correlation analysis
        cross_band_correlations = None
        if self.config.estimate_cross_band_correlation:
            cross_band_correlations = self._analyze_cross_band_correlations(
                source_uncertainties, photometry_results
            )
        
        # Monte Carlo error propagation
        if self.config.use_monte_carlo_propagation:
            self._monte_carlo_error_propagation(source_uncertainties, images, photometry_results)
        
        # Combine uncertainties
        self._combine_uncertainties(source_uncertainties)
        
        # Create uncertainty maps
        uncertainty_maps = self._create_uncertainty_maps(source_uncertainties, images)
        
        # Analyze systematic patterns
        systematic_patterns = self._analyze_systematic_patterns(source_uncertainties, images)
        
        # Compute statistics
        statistics = self._compute_uncertainty_statistics(source_uncertainties)
        
        # Quality assessment
        overall_quality, completeness = self._assess_uncertainty_quality(source_uncertainties)
        
        processing_time = time.time() - start_time
        try:
            import psutil
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            final_memory = 0.0  # Fallback if psutil not available
        
        results = UncertaintyResults(
            source_uncertainties=source_uncertainties,
            config=self.config,
            uncertainty_maps=uncertainty_maps,
            uncertainty_statistics=statistics,
            cross_band_correlation_matrix=cross_band_correlations,
            systematic_patterns=systematic_patterns,
            processing_time=processing_time,
            memory_usage=final_memory - initial_memory,
            overall_uncertainty_quality=overall_quality,
            uncertainty_completeness=completeness
        )
        
        self.logger.info(f"Uncertainty estimation completed in {processing_time:.2f} seconds")
        self.logger.info(f"Overall uncertainty quality: {overall_quality:.3f}")
        
        return results
    
    def _validate_uncertainty_inputs(self, images: Dict[str, np.ndarray], photometry_results: Dict[str, Any]) -> None:
        """Validate uncertainty estimation inputs."""
        if not images:
            raise ValueError("No images provided")
        
        if not photometry_results:
            raise ValueError("No photometry results provided")
        
        # Check that photometry and images have matching bands
        phot_bands = set(photometry_results.keys())
        image_bands = set(images.keys())
        
        if not phot_bands.intersection(image_bands):
            raise ValueError("No matching bands between photometry and images")
        
        # Validate individual images
        for band, image in images.items():
            validate_array(image, name=f"image_{band}")
    
    def _prepare_source_uncertainties(self, photometry_results: Dict[str, Any]) -> List[SourceUncertainties]:
        """Prepare source uncertainty list."""
        source_uncertainties = []
        
        # Determine number of sources from first band
        band_names = list(photometry_results.keys())
        if band_names:
            first_band = photometry_results[band_names[0]]
            n_sources = len(first_band.sources)
            
            for i in range(n_sources):
                source_uncertainties.append(SourceUncertainties(id=i))
        
        return source_uncertainties
    
    def _estimate_statistical_uncertainties(self,
                                          source_uncertainties: List[SourceUncertainties],
                                          images: Dict[str, np.ndarray],
                                          photometry_results: Dict[str, Any],
                                          background_maps: Optional[Dict[str, np.ndarray]],
                                          rms_maps: Optional[Dict[str, np.ndarray]]) -> None:
        """
        Estimate statistical uncertainties for all sources.
        
        Parameters:
        -----------
        source_uncertainties : list
            List of source uncertainties to update
        images : dict
            Input images
        photometry_results : dict
            Photometry results
        background_maps : dict, optional
            Background maps
        rms_maps : dict, optional
            RMS maps
        """
        self.logger.debug("Estimating statistical uncertainties")
        
        for band, phot_results in photometry_results.items():
            if band not in images:
                continue
            
            image = images[band]
            background_map = background_maps.get(band) if background_maps else None
            rms_map = rms_maps.get(band) if rms_maps else None
            
            for i, source in enumerate(phot_results.sources):
                if i >= len(source_uncertainties):
                    continue
                
                source_unc = source_uncertainties[i]
                
                # Estimate uncertainties for each aperture
                for aperture, flux in source.circular_fluxes.items():
                    try:
                        # Poisson noise
                        if self.config.include_poisson_noise:
                            poisson_error = self._estimate_poisson_uncertainty(
                                flux, aperture, source, image, background_map
                            )
                            source_unc.poisson_errors[aperture] = poisson_error
                        
                        # Background uncertainty
                        background_error = self._estimate_background_uncertainty(
                            aperture, source, image, background_map, rms_map
                        )
                        source_unc.background_errors[aperture] = background_error
                        
                        # Read noise
                        if self.config.include_readnoise:
                            readnoise_error = self._estimate_readnoise_uncertainty(
                                aperture, source
                            )
                            source_unc.readnoise_errors[aperture] = readnoise_error
                        
                        # Combine statistical uncertainties
                        total_stat_error = np.sqrt(
                            source_unc.poisson_errors.get(aperture, 0)**2 +
                            source_unc.background_errors.get(aperture, 0)**2 +
                            source_unc.readnoise_errors.get(aperture, 0)**2
                        )
                        source_unc.total_statistical_errors[aperture] = total_stat_error
                        
                        # Signal-to-noise ratio
                        if total_stat_error > 0:
                            source_unc.signal_to_noise[aperture] = flux / total_stat_error
                        else:
                            source_unc.signal_to_noise[aperture] = np.inf
                    
                    except Exception as e:
                        self.logger.debug(f"Statistical uncertainty estimation failed for source {i}, aperture {aperture}: {e}")
                        source_unc.uncertainty_flags.append(f"statistical_error_failed_r{aperture}")
    
    def _estimate_poisson_uncertainty(self,
                                    flux: float,
                                    aperture: float,
                                    source: Any,
                                    image: np.ndarray,
                                    background_map: Optional[np.ndarray]) -> float:
        """Estimate Poisson noise uncertainty."""
        try:
            # Get aperture area
            aperture_area = np.pi * aperture**2
            
            # Estimate background level
            if background_map is not None:
                background_level = background_map[int(source.y), int(source.x)]
            else:
                background_level = source.local_background or 0.0
            
            # Total counts in aperture (source + background)
            total_counts = (flux + background_level * aperture_area) * self.config.effective_gain
            
            # Poisson uncertainty in electrons, converted back to flux units
            if total_counts > 0:
                poisson_uncertainty = np.sqrt(total_counts) / self.config.effective_gain
            else:
                poisson_uncertainty = 0.0
            
            return poisson_uncertainty
        
        except:
            return 0.0
    
    def _estimate_background_uncertainty(self,
                                       aperture: float,
                                       source: Any,
                                       image: np.ndarray,
                                       background_map: Optional[np.ndarray],
                                       rms_map: Optional[np.ndarray]) -> float:
        """Estimate background uncertainty."""
        try:
            aperture_area = np.pi * aperture**2
            
            if self.config.background_uncertainty_method == "local_rms":
                # Use local background RMS measurement
                if hasattr(source, 'background_rms') and source.background_rms is not None:
                    background_rms = source.background_rms
                elif rms_map is not None:
                    background_rms = rms_map[int(source.y), int(source.x)]
                else:
                    background_rms = np.sqrt(source.local_background or 1.0)
                
                # Background uncertainty scales with aperture area
                background_uncertainty = background_rms * np.sqrt(aperture_area)
            
            elif self.config.background_uncertainty_method == "global_rms":
                # Use global background RMS
                if rms_map is not None:
                    global_rms = np.median(rms_map[rms_map > 0])
                    background_uncertainty = global_rms * np.sqrt(aperture_area)
                else:
                    background_uncertainty = np.sqrt(aperture_area)  # Default
            
            else:
                # Model-based approach
                background_uncertainty = self._model_based_background_uncertainty(
                    aperture, source, image, background_map
                )
            
            return background_uncertainty
        
        except:
            return np.sqrt(np.pi * aperture**2)  # Default uncertainty
    
    def _estimate_readnoise_uncertainty(self, aperture: float, source: Any) -> float:
        """Estimate read noise uncertainty."""
        try:
            aperture_area = np.pi * aperture**2
            
            # Read noise uncertainty scales with square root of number of pixels
            readnoise_uncertainty = self.config.read_noise * np.sqrt(aperture_area) / self.config.effective_gain
            
            return readnoise_uncertainty
        
        except:
            return 0.0
    
    def _model_based_background_uncertainty(self,
                                          aperture: float,
                                          source: Any,
                                          image: np.ndarray,
                                          background_map: Optional[np.ndarray]) -> float:
        """Model-based background uncertainty estimation."""
        try:
            # Extract local region around source
            stamp_size = int(4 * aperture)
            x_int, y_int = int(source.x), int(source.y)
            
            x_min = max(0, x_int - stamp_size)
            x_max = min(image.shape[1], x_int + stamp_size + 1)
            y_min = max(0, y_int - stamp_size)
            y_max = min(image.shape[0], y_int + stamp_size + 1)
            
            stamp = image[y_min:y_max, x_min:x_max]
            
            if stamp.size == 0:
                return np.sqrt(np.pi * aperture**2)
            
            # Remove source flux (crude)
            if background_map is not None:
                bg_stamp = background_map[y_min:y_max, x_min:x_max]
                residual_stamp = stamp - bg_stamp
            else:
                residual_stamp = stamp - np.median(stamp)
            
            # Estimate local RMS
            background_rms = np.std(residual_stamp)
            
            # Scale by aperture area
            return background_rms * np.sqrt(np.pi * aperture**2)
        
        except:
            return np.sqrt(np.pi * aperture**2)
    
    def _estimate_systematic_uncertainties(self,
                                         source_uncertainties: List[SourceUncertainties],
                                         calibration_results: Optional[Dict[str, Any]],
                                         psf_models: Optional[Dict[str, np.ndarray]]) -> None:
        """Estimate systematic uncertainties."""
        self.logger.debug("Estimating systematic uncertainties")
        
        for source_unc in source_uncertainties:
            # Flat field uncertainty
            source_unc.flat_field_error = self.config.flat_field_uncertainty
            
            # Calibration uncertainty
            if calibration_results:
                # Use actual calibration uncertainties if available
                cal_uncertainty = 0.0
                for band_cal in calibration_results.get('band_calibrations', {}).values():
                    if hasattr(band_cal, 'systematic_uncertainty'):
                        cal_uncertainty += band_cal.systematic_uncertainty**2
                source_unc.calibration_error = np.sqrt(cal_uncertainty)
            else:
                source_unc.calibration_error = self.config.zeropoint_uncertainty
            
            # Aperture correction uncertainty
            source_unc.aperture_correction_error = self.config.aperture_correction_uncertainty
            
            # PSF model uncertainty
            if psf_models:
                source_unc.psf_model_error = self.config.psf_uncertainty_fraction
            else:
                source_unc.psf_model_error = 0.0
    
    def _model_correlated_noise(self,
                              source_uncertainties: List[SourceUncertainties],
                              images: Dict[str, np.ndarray],
                              photometry_results: Dict[str, Any]) -> None:
        """Model spatially correlated noise."""
        self.logger.debug("Modeling correlated noise")
        
        for band, image in images.items():
            if band not in photometry_results:
                continue
            
            # Estimate spatial correlation function
            correlation_function = self._estimate_spatial_correlation(image)
            
            # Apply correlation to source uncertainties
            phot_results = photometry_results[band]
            for i, source in enumerate(phot_results.sources):
                if i >= len(source_uncertainties):
                    continue
                
                source_unc = source_uncertainties[i]
                
                # Estimate correlated noise contribution for each aperture
                for aperture in source.circular_fluxes.keys():
                    corr_noise = self._estimate_correlated_noise_contribution(
                        aperture, source, correlation_function
                    )
                    source_unc.correlated_noise_error[aperture] = corr_noise
    
    def _estimate_spatial_correlation(self, image: np.ndarray) -> np.ndarray:
        """Estimate spatial correlation function from image."""
        try:
            # Use autocorrelation of residual image
            # Smooth image to estimate large-scale structure
            smoothed = gaussian_filter(image, sigma=5.0)
            residual = image - smoothed
            
            # Compute autocorrelation
            from scipy.signal import correlate2d
            
            # Use central region to avoid edge effects
            center_size = min(50, min(residual.shape) // 4)
            center_y, center_x = residual.shape[0] // 2, residual.shape[1] // 2
            
            central_region = residual[
                center_y - center_size:center_y + center_size,
                center_x - center_size:center_x + center_size
            ]
            
            if central_region.size > 0:
                autocorr = correlate2d(central_region, central_region, mode='same')
                autocorr = autocorr / np.max(autocorr)  # Normalize
                return autocorr
            else:
                return np.array([[1.0]])
        
        except:
            # Return default correlation (uncorrelated)
            return np.array([[1.0]])
    
    def _estimate_correlated_noise_contribution(self,
                                              aperture: float,
                                              source: Any,
                                              correlation_function: np.ndarray) -> float:
        """Estimate correlated noise contribution for aperture."""
        try:
            # Simple model: correlation reduces effective number of independent pixels
            aperture_area = np.pi * aperture**2
            correlation_scale = self.config.spatial_correlation_scale
            
            # Effective correlation factor
            if correlation_scale > 0:
                effective_area = aperture_area / (1 + aperture / correlation_scale)
                correlation_factor = np.sqrt(aperture_area / effective_area)
            else:
                correlation_factor = 1.0
            
            # Apply to statistical uncertainty
            base_uncertainty = source.circular_flux_errors.get(aperture, 0.0)
            correlated_contribution = base_uncertainty * (correlation_factor - 1.0)
            
            return max(0.0, correlated_contribution)
        
        except:
            return 0.0
    
    def _estimate_crowding_uncertainties(self,
                                       source_uncertainties: List[SourceUncertainties],
                                       images: Dict[str, np.ndarray],
                                       photometry_results: Dict[str, Any]) -> None:
        """Estimate crowding and confusion uncertainties."""
        self.logger.debug("Estimating crowding uncertainties")
        
        for band, phot_results in photometry_results.items():
            if band not in images:
                continue
            
            # Create source position array
            source_positions = np.array([[s.x, s.y] for s in phot_results.sources])
            
            for i, source in enumerate(phot_results.sources):
                if i >= len(source_uncertainties):
                    continue
                
                source_unc = source_uncertainties[i]
                
                # Find nearby sources
                distances = np.sqrt(
                    (source_positions[:, 0] - source.x)**2 +
                    (source_positions[:, 1] - source.y)**2
                )
                
                nearby_mask = (distances > 0) & (distances < self.config.crowding_analysis_radius)
                nearby_sources = source_positions[nearby_mask]
                
                # Estimate crowding uncertainty
                if len(nearby_sources) > 0:
                    crowding_error = self._compute_crowding_uncertainty(
                        source, nearby_sources, phot_results.sources
                    )
                    source_unc.crowding_error = crowding_error
                    
                    # Check for high contamination
                    if len(nearby_sources) > 3:
                        source_unc.uncertainty_flags.append("crowded_field")
                else:
                    source_unc.crowding_error = 0.0
    
    def _compute_crowding_uncertainty(self,
                                    source: Any,
                                    nearby_positions: np.ndarray,
                                    all_sources: List[Any]) -> float:
        """Compute uncertainty due to crowding."""
        try:
            # Simplified crowding model
            # Uncertainty increases with number and brightness of neighbors
            
            crowding_factor = 0.0
            
            for pos in nearby_positions:
                distance = np.sqrt((pos[0] - source.x)**2 + (pos[1] - source.y)**2)
                
                if distance > 0:
                    # Weight by inverse distance and neighbor brightness
                    weight = 1.0 / distance
                    
                    # Find corresponding source for brightness
                    neighbor_flux = 1.0  # Default if can't find
                    for other_source in all_sources:
                        if (abs(other_source.x - pos[0]) < 0.5 and 
                            abs(other_source.y - pos[1]) < 0.5):
                            if other_source.circular_fluxes:
                                neighbor_flux = max(other_source.circular_fluxes.values())
                            break
                    
                    crowding_factor += weight * np.sqrt(neighbor_flux)
            
            # Convert to fractional uncertainty
            source_flux = max(source.circular_fluxes.values()) if source.circular_fluxes else 1.0
            crowding_uncertainty = crowding_factor * 0.01 * source_flux  # 1% per crowding unit
            
            return crowding_uncertainty
        
        except:
            return 0.0
    
    def _analyze_cross_band_correlations(self,
                                       source_uncertainties: List[SourceUncertainties],
                                       photometry_results: Dict[str, Any]) -> Optional[np.ndarray]:
        """Analyze cross-band correlations."""
        try:
            bands = list(photometry_results.keys())
            n_bands = len(bands)
            
            if n_bands < 2:
                return None
            
            # Create flux matrix (sources x bands)
            flux_matrix = []
            
            for band in bands:
                band_fluxes = []
                for source in photometry_results[band].sources:
                    if source.circular_fluxes:
                        # Use largest aperture
                        largest_ap = max(source.circular_fluxes.keys())
                        band_fluxes.append(source.circular_fluxes[largest_ap])
                    else:
                        band_fluxes.append(0.0)
                flux_matrix.append(band_fluxes)
            
            flux_matrix = np.array(flux_matrix).T  # Transpose to sources x bands
            
            # Compute correlation matrix
            if flux_matrix.shape[0] > 1 and flux_matrix.shape[1] > 1:
                correlation_matrix = np.corrcoef(flux_matrix, rowvar=False)
                
                # Store cross-band correlations for each source
                for i, source_unc in enumerate(source_uncertainties):
                    if i < flux_matrix.shape[0]:
                        for j, band1 in enumerate(bands):
                            for k, band2 in enumerate(bands):
                                if j != k:
                                    corr_key = f"{band1}_{band2}"
                                    source_unc.cross_band_correlations[corr_key] = correlation_matrix[j, k]
                
                return correlation_matrix
            else:
                return None
        
        except Exception as e:
            self.logger.debug(f"Cross-band correlation analysis failed: {e}")
            return None
    
    def _monte_carlo_error_propagation(self,
                                     source_uncertainties: List[SourceUncertainties],
                                     images: Dict[str, np.ndarray],
                                     photometry_results: Dict[str, Any]) -> None:
        """Perform Monte Carlo error propagation."""
        self.logger.debug("Performing Monte Carlo error propagation")
        
        for i, source_unc in enumerate(source_uncertainties):
            try:
                # Generate Monte Carlo samples
                mc_samples = self._generate_monte_carlo_samples(
                    source_unc, images, photometry_results, i
                )
                
                if mc_samples:
                    # Compute statistics for each aperture
                    for aperture in mc_samples.keys():
                        samples = mc_samples[aperture]
                        
                        source_unc.monte_carlo_uncertainties[aperture] = samples
                        source_unc.monte_carlo_statistics[aperture] = {
                            'mean': np.mean(samples),
                            'std': np.std(samples),
                            'median': np.median(samples),
                            'percentile_16': np.percentile(samples, 16),
                            'percentile_84': np.percentile(samples, 84),
                            'skewness': stats.skew(samples),
                            'kurtosis': stats.kurtosis(samples)
                        }
            
            except Exception as e:
                self.logger.debug(f"Monte Carlo error propagation failed for source {i}: {e}")
                source_unc.uncertainty_flags.append("monte_carlo_failed")
    
    def _generate_monte_carlo_samples(self,
                                    source_unc: SourceUncertainties,
                                    images: Dict[str, np.ndarray],
                                    photometry_results: Dict[str, Any],
                                    source_index: int) -> Dict[float, np.ndarray]:
        """Generate Monte Carlo samples for uncertainty propagation."""
        mc_samples = {}
        
        try:
            # For simplicity, assume Gaussian uncertainties and generate samples
            for aperture in source_unc.total_statistical_errors.keys():
                samples = []
                
                # Base flux value
                base_flux = 0.0
                for band_results in photometry_results.values():
                    if source_index < len(band_results.sources):
                        source = band_results.sources[source_index]
                        if aperture in source.circular_fluxes:
                            base_flux = source.circular_fluxes[aperture]
                            break
                
                # Generate samples
                stat_error = source_unc.total_statistical_errors[aperture]
                sys_error = source_unc.total_systematic_errors.get(aperture, 0.0)
                total_error = np.sqrt(stat_error**2 + sys_error**2)
                
                if total_error > 0:
                    samples = np.random.normal(
                        base_flux, total_error, self.config.n_monte_carlo_samples
                    )
                else:
                    samples = np.full(self.config.n_monte_carlo_samples, base_flux)
                
                mc_samples[aperture] = samples
            
            return mc_samples
        
        except:
            return {}
    
    def _combine_uncertainties(self, source_uncertainties: List[SourceUncertainties]) -> None:
        """Combine all uncertainty components."""
        self.logger.debug("Combining uncertainties")
        
        for source_unc in source_uncertainties:
            for aperture in source_unc.total_statistical_errors.keys():
                # Systematic uncertainties (assumed independent)
                systematic_components = [
                    source_unc.flat_field_error,
                    source_unc.calibration_error,
                    source_unc.aperture_correction_error,
                    source_unc.psf_model_error,
                    source_unc.crowding_error,
                    source_unc.correlated_noise_error.get(aperture, 0.0)
                ]
                
                total_sys_error = np.sqrt(sum(comp**2 for comp in systematic_components))
                source_unc.total_systematic_errors[aperture] = total_sys_error
                
                # Combined total uncertainty
                stat_error = source_unc.total_statistical_errors[aperture]
                total_error = np.sqrt(stat_error**2 + total_sys_error**2)
                source_unc.total_errors[aperture] = total_error
    
    def _create_uncertainty_maps(self,
                               source_uncertainties: List[SourceUncertainties],
                               images: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Create uncertainty maps."""
        uncertainty_maps = {}
        
        try:
            # Use first image as template
            template_shape = list(images.values())[0].shape
            
            # Create statistical uncertainty map
            stat_map = np.zeros(template_shape)
            sys_map = np.zeros(template_shape)
            total_map = np.zeros(template_shape)
            
            # Interpolate uncertainties across image
            for source_unc in source_uncertainties:
                # Find corresponding source position (simplified)
                # In practice, would need to map back to image coordinates
                pass  # Skip detailed implementation for now
            
            uncertainty_maps['statistical'] = stat_map
            uncertainty_maps['systematic'] = sys_map
            uncertainty_maps['total'] = total_map
        
        except Exception as e:
            self.logger.debug(f"Failed to create uncertainty maps: {e}")
        
        return uncertainty_maps
    
    def _analyze_systematic_patterns(self,
                                   source_uncertainties: List[SourceUncertainties],
                                   images: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Analyze systematic patterns in uncertainties."""
        patterns = {}
        
        try:
            if not self.config.detect_systematic_patterns:
                return patterns
            
            # Collect uncertainty values and positions
            uncertainty_values = []
            positions = []
            
            # For now, return empty patterns
            patterns['spatial_trends'] = False
            patterns['detector_patterns'] = False
            patterns['systematic_outliers'] = []
        
        except Exception as e:
            self.logger.debug(f"Systematic pattern analysis failed: {e}")
        
        return patterns
    
    def _compute_uncertainty_statistics(self, source_uncertainties: List[SourceUncertainties]) -> Dict[str, Any]:
        """Compute uncertainty statistics."""
        statistics = {}
        
        # Basic counts
        statistics['n_sources'] = len(source_uncertainties)
        
        # Collect all uncertainties by type
        all_stat_errors = []
        all_sys_errors = []
        all_total_errors = []
        all_snr_values = []
        
        for source_unc in source_uncertainties:
            all_stat_errors.extend(source_unc.total_statistical_errors.values())
            all_sys_errors.extend(source_unc.total_systematic_errors.values())
            all_total_errors.extend(source_unc.total_errors.values())
            all_snr_values.extend([snr for snr in source_unc.signal_to_noise.values() if np.isfinite(snr)])
        
        # Statistical uncertainty statistics
        if all_stat_errors:
            statistics['statistical_uncertainties'] = {
                'median': np.median(all_stat_errors),
                'mean': np.mean(all_stat_errors),
                'std': np.std(all_stat_errors),
                'percentile_90': np.percentile(all_stat_errors, 90)
            }
        
        # Systematic uncertainty statistics
        if all_sys_errors:
            statistics['systematic_uncertainties'] = {
                'median': np.median(all_sys_errors),
                'mean': np.mean(all_sys_errors),
                'std': np.std(all_sys_errors),
                'percentile_90': np.percentile(all_sys_errors, 90)
            }
        
        # Total uncertainty statistics
        if all_total_errors:
            statistics['total_uncertainties'] = {
                'median': np.median(all_total_errors),
                'mean': np.mean(all_total_errors),
                'std': np.std(all_total_errors),
                'percentile_90': np.percentile(all_total_errors, 90)
            }
        
        # Signal-to-noise statistics
        if all_snr_values:
            statistics['signal_to_noise'] = {
                'median': np.median(all_snr_values),
                'mean': np.mean(all_snr_values),
                'percentile_10': np.percentile(all_snr_values, 10),
                'high_snr_fraction': np.sum(np.array(all_snr_values) > 10) / len(all_snr_values)
            }
        
        # Flag statistics
        all_flags = []
        for source_unc in source_uncertainties:
            all_flags.extend(source_unc.uncertainty_flags)
        
        flag_counts = {}
        for flag in set(all_flags):
            flag_counts[flag] = all_flags.count(flag)
        
        statistics['flag_statistics'] = flag_counts
        
        return statistics
    
    def _assess_uncertainty_quality(self, source_uncertainties: List[SourceUncertainties]) -> Tuple[float, float]:
        """Assess overall uncertainty quality."""
        if not source_uncertainties:
            return 0.0, 0.0
        
        # Quality metrics
        quality_scores = []
        completeness_scores = []
        
        for source_unc in source_uncertainties:
            # Quality: inverse of relative uncertainty
            if source_unc.total_errors:
                avg_total_error = np.mean(list(source_unc.total_errors.values()))
                if avg_total_error > 0:
                    quality_score = min(1.0, 1.0 / (1.0 + avg_total_error))
                else:
                    quality_score = 1.0
            else:
                quality_score = 0.0
            
            quality_scores.append(quality_score)
            
            # Completeness: fraction of apertures with uncertainties
            if hasattr(source_unc, 'total_statistical_errors'):
                n_apertures_with_errors = len(source_unc.total_errors)
                n_total_apertures = max(len(source_unc.total_statistical_errors), 1)
                completeness = n_apertures_with_errors / n_total_apertures
            else:
                completeness = 0.0
            
            completeness_scores.append(completeness)
        
        overall_quality = np.mean(quality_scores) if quality_scores else 0.0
        overall_completeness = np.mean(completeness_scores) if completeness_scores else 0.0
        
        return overall_quality, overall_completeness
    
    def plot_uncertainty_diagnostics(self,
                                   results: UncertaintyResults,
                                   output_path: Optional[str] = None) -> None:
        """Create uncertainty diagnostic plots."""
        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            # Collect uncertainty data
            all_stat_errors = []
            all_sys_errors = []
            all_total_errors = []
            all_snr_values = []
            
            for source_unc in results.source_uncertainties:
                all_stat_errors.extend(source_unc.total_statistical_errors.values())
                all_sys_errors.extend(source_unc.total_systematic_errors.values())
                all_total_errors.extend(source_unc.total_errors.values())
                all_snr_values.extend([snr for snr in source_unc.signal_to_noise.values() if np.isfinite(snr)])
            
            # Statistical vs systematic uncertainties
            if all_stat_errors and all_sys_errors:
                axes[0, 0].scatter(all_stat_errors, all_sys_errors, alpha=0.6, s=10)
                axes[0, 0].set_xlabel('Statistical Uncertainty')
                axes[0, 0].set_ylabel('Systematic Uncertainty')
                axes[0, 0].set_title('Statistical vs Systematic Uncertainties')
                axes[0, 0].set_xscale('log')
                axes[0, 0].set_yscale('log')
            
            # Uncertainty distribution
            if all_total_errors:
                axes[0, 1].hist(np.log10(all_total_errors), bins=30, alpha=0.7)
                axes[0, 1].set_xlabel('log10(Total Uncertainty)')
                axes[0, 1].set_ylabel('Count')
                axes[0, 1].set_title('Total Uncertainty Distribution')
            
            # Signal-to-noise distribution
            if all_snr_values:
                axes[0, 2].hist(np.log10(all_snr_values), bins=30, alpha=0.7)
                axes[0, 2].set_xlabel('log10(Signal-to-Noise Ratio)')
                axes[0, 2].set_ylabel('Count')
                axes[0, 2].set_title('S/N Distribution')
            
            # Uncertainty components breakdown
            if all_stat_errors and all_sys_errors:
                stat_fraction = np.array(all_stat_errors) / np.array(all_total_errors)
                sys_fraction = np.array(all_sys_errors) / np.array(all_total_errors)
                
                axes[1, 0].hist([stat_fraction, sys_fraction], bins=20, alpha=0.7, 
                               label=['Statistical', 'Systematic'])
                axes[1, 0].set_xlabel('Uncertainty Fraction')
                axes[1, 0].set_ylabel('Count')
                axes[1, 0].set_title('Uncertainty Component Breakdown')
                axes[1, 0].legend()
            
            # Cross-band correlation matrix
            if results.cross_band_correlation_matrix is not None:
                im = axes[1, 1].imshow(results.cross_band_correlation_matrix, 
                                      cmap='RdBu_r', vmin=-1, vmax=1)
                axes[1, 1].set_title('Cross-Band Correlation Matrix')
                plt.colorbar(im, ax=axes[1, 1])
            
            # Quality metrics
            qualities = []
            for source_unc in results.source_uncertainties:
                if source_unc.total_errors:
                    avg_error = np.mean(list(source_unc.total_errors.values()))
                    quality = 1.0 / (1.0 + avg_error) if avg_error > 0 else 1.0
                    qualities.append(quality)
            
            if qualities:
                axes[1, 2].hist(qualities, bins=20, alpha=0.7)
                axes[1, 2].set_xlabel('Quality Score')
                axes[1, 2].set_ylabel('Count')
                axes[1, 2].set_title('Uncertainty Quality Distribution')
            
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                self.logger.info(f"Uncertainty diagnostics saved to {output_path}")
            else:
                plt.show()
            
            plt.close()
            
        except Exception as e:
            self.logger.warning(f"Failed to create uncertainty diagnostic plots: {e}")


# Utility functions for error budget calculations
def propagate_uncertainties(values: np.ndarray, 
                          uncertainties: np.ndarray, 
                          operation: str = "add") -> Tuple[float, float]:
    """
    Propagate uncertainties through mathematical operations.
    
    Parameters:
    -----------
    values : numpy.ndarray
        Input values
    uncertainties : numpy.ndarray
        Input uncertainties
    operation : str
        Operation type ("add", "multiply", "divide", "power")
        
    Returns:
    --------
    tuple
        Result value and propagated uncertainty
    """
    if operation == "add":
        result = np.sum(values)
        result_uncertainty = np.sqrt(np.sum(uncertainties**2))
    
    elif operation == "multiply":
        result = np.prod(values)
        if result != 0:
            fractional_uncertainties = uncertainties / values
            result_uncertainty = result * np.sqrt(np.sum(fractional_uncertainties**2))
        else:
            result_uncertainty = 0.0
    
    elif operation == "divide":
        if len(values) != 2 or len(uncertainties) != 2:
            raise ValueError("Division requires exactly 2 values")
        
        if values[1] != 0:
            result = values[0] / values[1]
            fractional_uncertainties = uncertainties / values
            result_uncertainty = result * np.sqrt(np.sum(fractional_uncertainties**2))
        else:
            result = np.inf
            result_uncertainty = np.inf
    
    else:
        raise ValueError(f"Unsupported operation: {operation}")
    
    return result, result_uncertainty


def compute_error_budget(components: Dict[str, float]) -> Dict[str, float]:
    """
    Compute error budget from individual components.
    
    Parameters:
    -----------
    components : dict
        Dictionary of error components
        
    Returns:
    --------
    dict
        Error budget with total and fractional contributions
    """
    total_variance = sum(error**2 for error in components.values())
    total_error = np.sqrt(total_variance)
    
    error_budget = {
        'total_error': total_error,
        'total_variance': total_variance
    }
    
    # Fractional contributions
    for component, error in components.items():
        if total_variance > 0:
            fraction = (error**2) / total_variance
        else:
            fraction = 0.0
        error_budget[f'{component}_fraction'] = fraction
    
    return error_budget
