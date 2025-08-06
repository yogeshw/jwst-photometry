"""
JWST photometry advanced PSF fitting module (Phase 3.3).

This module provides sophisticated PSF-fitting photometry for JWST NIRCam data,
including simultaneous multi-band fitting, blend identification and deblending,
and crowded field analysis.
"""

import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from pathlib import Path
import time

# Core scientific libraries
from scipy import optimize, ndimage
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from astropy.io import fits
from astropy.table import Table
import astropy.units as u

# Specialized imports with fallbacks
try:
    import sep
except ImportError:
    print("Warning: sep not available for some advanced features")
    sep = None

try:
    from photutils import CircularAperture
except ImportError:
    print("Warning: photutils not available for some features")

# Local imports with fallbacks
try:
    from .utils import setup_logger, memory_monitor, validate_array
except ImportError:
    def setup_logger(name): return logging.getLogger(name)
    def memory_monitor(func): return func
    def validate_array(arr): return arr is not None

import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple, List, Union
from dataclasses import dataclass, field
from pathlib import Path

from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
from scipy import ndimage, optimize
from scipy.ndimage import center_of_mass, zoom
from scipy.interpolate import griddata
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# Import our PSF modules
from .psf import AdvancedPSFProcessor, PSFModelResults
from .utils import setup_logger, memory_monitor, validate_array


@dataclass
class PSFPhotometryConfig:
    """Configuration parameters for PSF photometry."""
    
    # PSF fitting parameters
    psf_fit_method: str = "least_squares"  # "least_squares", "maximum_likelihood", "mcmc"
    fit_radius: float = 5.0  # Fitting radius in units of PSF FWHM
    background_annulus: Tuple[float, float] = (6.0, 8.0)  # Inner and outer radii for background
    
    # Multi-band fitting
    simultaneous_fitting: bool = True
    force_common_centroid: bool = True
    wavelength_dependent_psf: bool = True
    
    # Crowded field handling
    enable_deblending: bool = True
    deblending_threshold: float = 0.005  # Fraction of peak flux
    max_blend_components: int = 5
    min_separation: float = 0.5  # Minimum separation in PSF FWHM units
    
    # Iteration and convergence
    max_iterations: int = 100
    convergence_tolerance: float = 1e-6
    damping_factor: float = 0.7  # For Levenberg-Marquardt
    
    # Error estimation
    estimate_uncertainties: bool = True
    bootstrap_samples: int = 100
    use_covariance_matrix: bool = True
    include_systematic_errors: bool = True
    
    # Quality control
    min_fit_quality: float = 0.1  # Minimum acceptable fit quality
    max_chi_squared: float = 5.0
    flag_poor_fits: bool = True
    
    # Output control
    save_residuals: bool = True
    save_fit_diagnostics: bool = True
    create_diagnostic_plots: bool = True


@dataclass
class PSFPhotometrySource:
    """Container for PSF photometry source data."""
    
    id: int
    x: float
    y: float
    ra: Optional[float] = None
    dec: Optional[float] = None
    
    # Fitted parameters
    flux: Dict[str, float] = field(default_factory=dict)
    flux_error: Dict[str, float] = field(default_factory=dict)
    magnitude: Dict[str, float] = field(default_factory=dict)
    magnitude_error: Dict[str, float] = field(default_factory=dict)
    
    # Fit quality metrics
    chi_squared: Dict[str, float] = field(default_factory=dict)
    fit_quality: Dict[str, float] = field(default_factory=dict)
    n_iterations: Dict[str, int] = field(default_factory=dict)
    
    # Deblending information
    is_blended: bool = False
    blend_components: List[int] = field(default_factory=list)
    deblending_quality: Optional[float] = None
    
    # Quality flags
    flags: Dict[str, List[str]] = field(default_factory=dict)
    
    # Diagnostic information
    fitted_psf: Dict[str, np.ndarray] = field(default_factory=dict)
    residuals: Dict[str, np.ndarray] = field(default_factory=dict)
    fit_parameters: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass
class PSFPhotometryResults:
    """Container for PSF photometry results."""
    
    sources: List[PSFPhotometrySource]
    psf_models: Dict[str, np.ndarray]
    
    # Global statistics
    statistics: Dict[str, Any] = field(default_factory=dict)
    
    # Quality assessment
    completeness: Dict[str, float] = field(default_factory=dict)
    reliability: Dict[str, float] = field(default_factory=dict)
    
    # Processing information
    processing_time: float = 0.0
    n_sources_processed: int = 0
    n_sources_successful: int = 0
    
    # Diagnostic information
    diagnostics: Dict[str, Any] = field(default_factory=dict)


class AdvancedPSFPhotometry:
    """
    Advanced PSF photometry processor for JWST observations.
    
    This class provides sophisticated PSF fitting capabilities including:
    - Single and multi-band PSF fitting
    - Crowded field deblending
    - Simultaneous fitting with common centroids
    - Comprehensive error estimation
    - Quality assessment and flagging
    - Diagnostic tools and visualization
    """
    
    def __init__(self, config: Optional[PSFPhotometryConfig] = None):
        """
        Initialize the PSF photometry processor.
        
        Parameters:
        -----------
        config : PSFPhotometryConfig, optional
            PSF photometry configuration. If None, uses defaults.
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or PSFPhotometryConfig()
    
    def perform_psf_photometry(self,
                             images: Dict[str, np.ndarray],
                             psf_models: Dict[str, np.ndarray],
                             sources: np.ndarray,
                             background_maps: Optional[Dict[str, np.ndarray]] = None,
                             rms_maps: Optional[Dict[str, np.ndarray]] = None,
                             wcs: Optional[WCS] = None) -> PSFPhotometryResults:
        """
        Perform PSF photometry on detected sources.
        
        Parameters:
        -----------
        images : dict
            Dictionary of images {band_name: image_array}
        psf_models : dict
            Dictionary of PSF models {band_name: psf_array}
        sources : numpy.ndarray
            Detected sources catalog
        background_maps : dict, optional
            Background maps for each band
        rms_maps : dict, optional
            RMS/noise maps for each band
        wcs : astropy.wcs.WCS, optional
            World coordinate system
            
        Returns:
        --------
        PSFPhotometryResults
            Complete PSF photometry results
        """
        self.logger.info("Starting PSF photometry")
        
        import time
        start_time = time.time()
        
        # Validate inputs
        self._validate_inputs(images, psf_models, sources)
        
        # Prepare photometry sources
        phot_sources = self._prepare_sources(sources, wcs)
        
        # Identify blended sources if deblending is enabled
        if self.config.enable_deblending:
            phot_sources = self._identify_blends(phot_sources, psf_models)
        
        # Perform photometry
        if self.config.simultaneous_fitting and len(images) > 1:
            phot_sources = self._perform_simultaneous_photometry(
                phot_sources, images, psf_models, background_maps, rms_maps
            )
        else:
            phot_sources = self._perform_single_band_photometry(
                phot_sources, images, psf_models, background_maps, rms_maps
            )
        
        # Compute global statistics
        statistics = self._compute_statistics(phot_sources, images)
        
        # Assess quality
        completeness, reliability = self._assess_quality(phot_sources, images)
        
        # Create diagnostics
        diagnostics = self._create_diagnostics(phot_sources, psf_models)
        
        processing_time = time.time() - start_time
        
        results = PSFPhotometryResults(
            sources=phot_sources,
            psf_models=psf_models,
            statistics=statistics,
            completeness=completeness,
            reliability=reliability,
            processing_time=processing_time,
            n_sources_processed=len(phot_sources),
            n_sources_successful=len([s for s in phot_sources if any(s.flux.values())]),
            diagnostics=diagnostics
        )
        
        self.logger.info(f"PSF photometry completed in {processing_time:.2f} seconds")
        self.logger.info(f"Successfully processed {results.n_sources_successful}/{results.n_sources_processed} sources")
        
        return results
    
    def _validate_inputs(self, images: Dict[str, np.ndarray], 
                        psf_models: Dict[str, np.ndarray],
                        sources: np.ndarray) -> None:
        """Validate input data."""
        if not images:
            raise ValueError("No images provided")
        
        if not psf_models:
            raise ValueError("No PSF models provided")
        
        if len(sources) == 0:
            raise ValueError("No sources provided")
        
        # Check that we have PSF models for all images
        missing_psfs = set(images.keys()) - set(psf_models.keys())
        if missing_psfs:
            raise ValueError(f"Missing PSF models for bands: {missing_psfs}")
        
        # Validate array shapes
        for band_name, image in images.items():
            validate_array(image, name=f"image_{band_name}")
            
            if band_name in psf_models:
                validate_array(psf_models[band_name], name=f"psf_{band_name}")
    
    def _prepare_sources(self, sources: np.ndarray, wcs: Optional[WCS]) -> List[PSFPhotometrySource]:
        """
        Prepare source list for PSF photometry.
        
        Parameters:
        -----------
        sources : numpy.ndarray
            Source catalog
        wcs : astropy.wcs.WCS, optional
            World coordinate system
            
        Returns:
        --------
        list
            List of PSF photometry sources
        """
        phot_sources = []
        
        for i, source in enumerate(sources):
            phot_source = PSFPhotometrySource(
                id=i,
                x=source['x'],
                y=source['y']
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
        
        return phot_sources
    
    def _identify_blends(self, sources: List[PSFPhotometrySource],
                        psf_models: Dict[str, np.ndarray]) -> List[PSFPhotometrySource]:
        """
        Identify blended sources based on proximity.
        
        Parameters:
        -----------
        sources : list
            List of PSF photometry sources
        psf_models : dict
            PSF models for each band
            
        Returns:
        --------
        list
            Updated sources with blend information
        """
        if len(sources) < 2:
            return sources
        
        # Use the first PSF to estimate FWHM
        first_psf = list(psf_models.values())[0]
        psf_fwhm = self._estimate_psf_fwhm(first_psf)
        min_separation_pixels = self.config.min_separation * psf_fwhm
        
        # Create position array
        positions = np.array([[s.x, s.y] for s in sources])
        
        # Use DBSCAN to identify clusters
        clustering = DBSCAN(eps=min_separation_pixels, min_samples=2)
        labels = clustering.fit_predict(positions)
        
        # Mark blended sources
        for i, label in enumerate(labels):
            if label >= 0:  # Not noise
                # Find all sources in this cluster
                cluster_sources = np.where(labels == label)[0]
                
                if len(cluster_sources) > 1:
                    sources[i].is_blended = True
                    sources[i].blend_components = cluster_sources.tolist()
        
        n_blended = sum(1 for s in sources if s.is_blended)
        if n_blended > 0:
            self.logger.info(f"Identified {n_blended} blended sources")
        
        return sources
    
    def _estimate_psf_fwhm(self, psf: np.ndarray) -> float:
        """Estimate PSF FWHM from PSF model."""
        # Calculate second moments
        y, x = np.ogrid[:psf.shape[0], :psf.shape[1]]
        
        total_flux = np.sum(psf)
        x_center = np.sum(x * psf) / total_flux
        y_center = np.sum(y * psf) / total_flux
        
        m_xx = np.sum((x - x_center)**2 * psf) / total_flux
        m_yy = np.sum((y - y_center)**2 * psf) / total_flux
        
        fwhm = 2.0 * np.sqrt(np.log(2) * (m_xx + m_yy))
        return fwhm
    
    def _perform_simultaneous_photometry(self,
                                       sources: List[PSFPhotometrySource],
                                       images: Dict[str, np.ndarray],
                                       psf_models: Dict[str, np.ndarray],
                                       background_maps: Optional[Dict[str, np.ndarray]],
                                       rms_maps: Optional[Dict[str, np.ndarray]]) -> List[PSFPhotometrySource]:
        """
        Perform simultaneous multi-band PSF photometry.
        
        Parameters:
        -----------
        sources : list
            PSF photometry sources
        images : dict
            Images for each band
        psf_models : dict
            PSF models for each band
        background_maps : dict, optional
            Background maps
        rms_maps : dict, optional
            RMS maps
            
        Returns:
        --------
        list
            Updated sources with photometry results
        """
        self.logger.debug("Performing simultaneous multi-band PSF photometry")
        
        band_names = list(images.keys())
        
        for source in sources:
            try:
                # Extract data for all bands
                fit_data = self._extract_fit_data(
                    source, images, psf_models, background_maps, rms_maps
                )
                
                if fit_data is None:
                    continue
                
                # Perform simultaneous fit
                if source.is_blended and self.config.enable_deblending:
                    fit_result = self._fit_blended_source(source, fit_data, band_names)
                else:
                    fit_result = self._fit_single_source(source, fit_data, band_names)
                
                # Store results
                if fit_result is not None:
                    self._store_fit_results(source, fit_result, band_names)
                
            except Exception as e:
                self.logger.debug(f"Simultaneous photometry failed for source {source.id}: {e}")
                for band in band_names:
                    source.flags.setdefault(band, []).append("simultaneous_fit_failed")
        
        return sources
    
    def _perform_single_band_photometry(self,
                                      sources: List[PSFPhotometrySource],
                                      images: Dict[str, np.ndarray],
                                      psf_models: Dict[str, np.ndarray],
                                      background_maps: Optional[Dict[str, np.ndarray]],
                                      rms_maps: Optional[Dict[str, np.ndarray]]) -> List[PSFPhotometrySource]:
        """
        Perform single-band PSF photometry.
        
        Parameters:
        -----------
        sources : list
            PSF photometry sources
        images : dict
            Images for each band
        psf_models : dict
            PSF models for each band
        background_maps : dict, optional
            Background maps
        rms_maps : dict, optional
            RMS maps
            
        Returns:
        --------
        list
            Updated sources with photometry results
        """
        self.logger.debug("Performing single-band PSF photometry")
        
        for band_name in images.keys():
            for source in sources:
                try:
                    # Extract data for this band
                    fit_data = self._extract_single_band_data(
                        source, band_name, images[band_name], psf_models[band_name],
                        background_maps.get(band_name) if background_maps else None,
                        rms_maps.get(band_name) if rms_maps else None
                    )
                    
                    if fit_data is None:
                        continue
                    
                    # Perform fit
                    if source.is_blended and self.config.enable_deblending:
                        fit_result = self._fit_blended_single_band(source, fit_data, band_name)
                    else:
                        fit_result = self._fit_single_band_source(source, fit_data, band_name)
                    
                    # Store results
                    if fit_result is not None:
                        self._store_single_band_results(source, fit_result, band_name)
                    
                except Exception as e:
                    self.logger.debug(f"Single-band photometry failed for source {source.id} in {band_name}: {e}")
                    source.flags.setdefault(band_name, []).append("single_band_fit_failed")
        
        return sources
    
    def _extract_fit_data(self,
                         source: PSFPhotometrySource,
                         images: Dict[str, np.ndarray],
                         psf_models: Dict[str, np.ndarray],
                         background_maps: Optional[Dict[str, np.ndarray]],
                         rms_maps: Optional[Dict[str, np.ndarray]]) -> Optional[Dict[str, Any]]:
        """
        Extract fitting data for simultaneous photometry.
        
        Parameters:
        -----------
        source : PSFPhotometrySource
            Source to fit
        images : dict
            Images for each band
        psf_models : dict
            PSF models for each band
        background_maps : dict, optional
            Background maps
        rms_maps : dict, optional
            RMS maps
            
        Returns:
        --------
        dict or None
            Fitting data or None if extraction failed
        """
        try:
            # Estimate fitting region size
            first_psf = list(psf_models.values())[0]
            psf_fwhm = self._estimate_psf_fwhm(first_psf)
            fit_radius_pixels = int(self.config.fit_radius * psf_fwhm)
            
            # Extract stamps for all bands
            stamps = {}
            psf_stamps = {}
            weights = {}
            
            for band_name in images.keys():
                # Extract image stamp
                stamp = self._extract_stamp(
                    images[band_name], source.x, source.y, fit_radius_pixels
                )
                
                if stamp is None:
                    return None
                
                # Subtract background if available
                if background_maps and band_name in background_maps:
                    bkg_stamp = self._extract_stamp(
                        background_maps[band_name], source.x, source.y, fit_radius_pixels
                    )
                    if bkg_stamp is not None:
                        stamp = stamp - bkg_stamp
                
                # Extract PSF stamp
                psf_stamp = self._extract_centered_psf(psf_models[band_name], fit_radius_pixels)
                
                if psf_stamp is None:
                    return None
                
                # Create weight map
                if rms_maps and band_name in rms_maps:
                    rms_stamp = self._extract_stamp(
                        rms_maps[band_name], source.x, source.y, fit_radius_pixels
                    )
                    if rms_stamp is not None:
                        weight = 1.0 / (rms_stamp**2 + 1e-10)
                    else:
                        weight = np.ones_like(stamp)
                else:
                    weight = np.ones_like(stamp)
                
                stamps[band_name] = stamp
                psf_stamps[band_name] = psf_stamp
                weights[band_name] = weight
            
            return {
                'stamps': stamps,
                'psf_stamps': psf_stamps,
                'weights': weights,
                'fit_radius': fit_radius_pixels,
                'center_x': source.x,
                'center_y': source.y
            }
            
        except Exception:
            return None
    
    def _extract_single_band_data(self,
                                source: PSFPhotometrySource,
                                band_name: str,
                                image: np.ndarray,
                                psf_model: np.ndarray,
                                background_map: Optional[np.ndarray],
                                rms_map: Optional[np.ndarray]) -> Optional[Dict[str, Any]]:
        """Extract fitting data for single-band photometry."""
        try:
            psf_fwhm = self._estimate_psf_fwhm(psf_model)
            fit_radius_pixels = int(self.config.fit_radius * psf_fwhm)
            
            # Extract image stamp
            stamp = self._extract_stamp(image, source.x, source.y, fit_radius_pixels)
            if stamp is None:
                return None
            
            # Subtract background
            if background_map is not None:
                bkg_stamp = self._extract_stamp(background_map, source.x, source.y, fit_radius_pixels)
                if bkg_stamp is not None:
                    stamp = stamp - bkg_stamp
            
            # Extract PSF stamp
            psf_stamp = self._extract_centered_psf(psf_model, fit_radius_pixels)
            if psf_stamp is None:
                return None
            
            # Create weight map
            if rms_map is not None:
                rms_stamp = self._extract_stamp(rms_map, source.x, source.y, fit_radius_pixels)
                if rms_stamp is not None:
                    weight = 1.0 / (rms_stamp**2 + 1e-10)
                else:
                    weight = np.ones_like(stamp)
            else:
                weight = np.ones_like(stamp)
            
            return {
                'stamp': stamp,
                'psf_stamp': psf_stamp,
                'weight': weight,
                'fit_radius': fit_radius_pixels
            }
            
        except Exception:
            return None
    
    def _extract_stamp(self, image: np.ndarray, x: float, y: float, radius: int) -> Optional[np.ndarray]:
        """Extract a square stamp around a position."""
        try:
            x_int, y_int = int(round(x)), int(round(y))
            
            x_min = max(0, x_int - radius)
            x_max = min(image.shape[1], x_int + radius + 1)
            y_min = max(0, y_int - radius)
            y_max = min(image.shape[0], y_int + radius + 1)
            
            if x_max - x_min < 2 * radius or y_max - y_min < 2 * radius:
                return None
            
            return image[y_min:y_max, x_min:x_max].copy()
            
        except Exception:
            return None
    
    def _extract_centered_psf(self, psf: np.ndarray, radius: int) -> Optional[np.ndarray]:
        """Extract centered PSF stamp."""
        try:
            center_y, center_x = (psf.shape[0] - 1) // 2, (psf.shape[1] - 1) // 2
            
            y_min = max(0, center_y - radius)
            y_max = min(psf.shape[0], center_y + radius + 1)
            x_min = max(0, center_x - radius)
            x_max = min(psf.shape[1], center_x + radius + 1)
            
            if y_max - y_min < 2 * radius or x_max - x_min < 2 * radius:
                return None
            
            return psf[y_min:y_max, x_min:x_max].copy()
            
        except Exception:
            return None
    
    def _fit_single_source(self, source: PSFPhotometrySource, 
                          fit_data: Dict[str, Any],
                          band_names: List[str]) -> Optional[Dict[str, Any]]:
        """
        Fit a single source using simultaneous multi-band fitting.
        
        Parameters:
        -----------
        source : PSFPhotometrySource
            Source to fit
        fit_data : dict
            Fitting data
        band_names : list
            List of band names
            
        Returns:
        --------
        dict or None
            Fit results
        """
        try:
            # Initial parameter guess
            n_bands = len(band_names)
            initial_params = []
            
            # Flux parameters for each band
            for band_name in band_names:
                stamp = fit_data['stamps'][band_name]
                initial_flux = np.max(stamp)
                initial_params.append(initial_flux)
            
            # Position parameters (if not forced to be common)
            if not self.config.force_common_centroid:
                initial_params.extend([0.0, 0.0])  # dx, dy from nominal position
            
            # Background parameters for each band
            for band_name in band_names:
                stamp = fit_data['stamps'][band_name]
                initial_background = np.median(stamp[stamp < np.percentile(stamp, 20)])
                initial_params.append(initial_background)
            
            # Define objective function
            def objective_function(params):
                return self._compute_residuals(params, fit_data, band_names)
            
            # Perform optimization
            result = optimize.least_squares(
                objective_function,
                initial_params,
                max_nfev=self.config.max_iterations,
                ftol=self.config.convergence_tolerance
            )
            
            if result.success:
                return {
                    'parameters': result.x,
                    'chi_squared': np.sum(result.fun**2),
                    'n_iterations': result.nfev,
                    'success': True,
                    'jacobian': result.jac if hasattr(result, 'jac') else None
                }
            else:
                return None
                
        except Exception:
            return None
    
    def _fit_single_band_source(self, source: PSFPhotometrySource,
                               fit_data: Dict[str, Any],
                               band_name: str) -> Optional[Dict[str, Any]]:
        """Fit a single source in a single band."""
        try:
            # Initial parameters: flux, dx, dy, background
            stamp = fit_data['stamp']
            initial_flux = np.max(stamp)
            initial_background = np.median(stamp[stamp < np.percentile(stamp, 20)])
            initial_params = [initial_flux, 0.0, 0.0, initial_background]
            
            def objective_function(params):
                flux, dx, dy, background = params
                
                # Create model
                psf_stamp = fit_data['psf_stamp']
                model = flux * psf_stamp + background
                
                # Apply position shift if needed
                if abs(dx) > 0.1 or abs(dy) > 0.1:
                    from scipy.ndimage import shift
                    model = shift(model, (dy, dx), order=1, mode='constant', cval=background)
                
                # Calculate residuals
                residuals = (stamp - model) * np.sqrt(fit_data['weight'])
                return residuals.ravel()
            
            # Perform optimization
            result = optimize.least_squares(
                objective_function,
                initial_params,
                max_nfev=self.config.max_iterations,
                ftol=self.config.convergence_tolerance
            )
            
            if result.success:
                return {
                    'parameters': result.x,
                    'chi_squared': np.sum(result.fun**2),
                    'n_iterations': result.nfev,
                    'success': True
                }
            else:
                return None
                
        except Exception:
            return None
    
    def _fit_blended_source(self, source: PSFPhotometrySource,
                           fit_data: Dict[str, Any],
                           band_names: List[str]) -> Optional[Dict[str, Any]]:
        """Fit blended sources using deblending."""
        # This is a simplified deblending implementation
        # In practice, more sophisticated algorithms would be used
        try:
            # For now, just fit the primary source and flag as blended
            result = self._fit_single_source(source, fit_data, band_names)
            if result:
                result['deblended'] = True
                result['deblending_quality'] = 0.5  # Placeholder
            return result
        except Exception:
            return None
    
    def _fit_blended_single_band(self, source: PSFPhotometrySource,
                                fit_data: Dict[str, Any],
                                band_name: str) -> Optional[Dict[str, Any]]:
        """Fit blended source in single band."""
        # Simplified single-band deblending
        try:
            result = self._fit_single_band_source(source, fit_data, band_name)
            if result:
                result['deblended'] = True
                result['deblending_quality'] = 0.5  # Placeholder
            return result
        except Exception:
            return None
    
    def _compute_residuals(self, params: np.ndarray, 
                          fit_data: Dict[str, Any],
                          band_names: List[str]) -> np.ndarray:
        """Compute residuals for multi-band fitting."""
        try:
            n_bands = len(band_names)
            
            # Extract parameters
            fluxes = params[:n_bands]
            
            if self.config.force_common_centroid:
                dx, dy = 0.0, 0.0
                backgrounds = params[n_bands:]
            else:
                dx, dy = params[n_bands:n_bands+2]
                backgrounds = params[n_bands+2:]
            
            residuals = []
            
            for i, band_name in enumerate(band_names):
                stamp = fit_data['stamps'][band_name]
                psf_stamp = fit_data['psf_stamps'][band_name]
                weight = fit_data['weights'][band_name]
                
                # Create model
                model = fluxes[i] * psf_stamp + backgrounds[i]
                
                # Apply position shift if needed
                if abs(dx) > 0.1 or abs(dy) > 0.1:
                    from scipy.ndimage import shift
                    model = shift(model, (dy, dx), order=1, mode='constant', cval=backgrounds[i])
                
                # Calculate weighted residuals
                band_residuals = (stamp - model) * np.sqrt(weight)
                residuals.append(band_residuals.ravel())
            
            return np.concatenate(residuals)
            
        except Exception:
            return np.array([1e10])  # Return large residual on error
    
    def _store_fit_results(self, source: PSFPhotometrySource,
                          fit_result: Dict[str, Any],
                          band_names: List[str]) -> None:
        """Store simultaneous fit results."""
        try:
            n_bands = len(band_names)
            params = fit_result['parameters']
            
            # Extract fluxes
            fluxes = params[:n_bands]
            
            # Store results for each band
            for i, band_name in enumerate(band_names):
                source.flux[band_name] = fluxes[i]
                source.chi_squared[band_name] = fit_result['chi_squared'] / n_bands
                source.n_iterations[band_name] = fit_result['n_iterations']
                
                # Convert to magnitude (placeholder zeropoint)
                if fluxes[i] > 0:
                    source.magnitude[band_name] = -2.5 * np.log10(fluxes[i]) + 25.0
                else:
                    source.magnitude[band_name] = 99.0
                    source.flags.setdefault(band_name, []).append("negative_flux")
                
                # Estimate uncertainties
                if self.config.estimate_uncertainties and fit_result.get('jacobian') is not None:
                    # Simplified uncertainty estimation
                    source.flux_error[band_name] = np.sqrt(abs(fluxes[i]))  # Placeholder
                    if source.magnitude[band_name] < 90:
                        source.magnitude_error[band_name] = 1.0857 * source.flux_error[band_name] / fluxes[i]
                    else:
                        source.magnitude_error[band_name] = 99.0
                
                # Quality assessment
                source.fit_quality[band_name] = 1.0 / (1.0 + source.chi_squared[band_name])
                
                if source.chi_squared[band_name] > self.config.max_chi_squared:
                    source.flags.setdefault(band_name, []).append("poor_fit")
            
            # Store deblending information if applicable
            if fit_result.get('deblended', False):
                source.deblending_quality = fit_result.get('deblending_quality', 0.0)
            
        except Exception as e:
            self.logger.debug(f"Failed to store fit results for source {source.id}: {e}")
    
    def _store_single_band_results(self, source: PSFPhotometrySource,
                                  fit_result: Dict[str, Any],
                                  band_name: str) -> None:
        """Store single-band fit results."""
        try:
            params = fit_result['parameters']
            flux, dx, dy, background = params
            
            source.flux[band_name] = flux
            source.chi_squared[band_name] = fit_result['chi_squared']
            source.n_iterations[band_name] = fit_result['n_iterations']
            
            # Convert to magnitude
            if flux > 0:
                source.magnitude[band_name] = -2.5 * np.log10(flux) + 25.0
            else:
                source.magnitude[band_name] = 99.0
                source.flags.setdefault(band_name, []).append("negative_flux")
            
            # Estimate uncertainties (simplified)
            if self.config.estimate_uncertainties:
                source.flux_error[band_name] = np.sqrt(abs(flux))
                if source.magnitude[band_name] < 90:
                    source.magnitude_error[band_name] = 1.0857 * source.flux_error[band_name] / flux
                else:
                    source.magnitude_error[band_name] = 99.0
            
            # Quality assessment
            source.fit_quality[band_name] = 1.0 / (1.0 + source.chi_squared[band_name])
            
            if source.chi_squared[band_name] > self.config.max_chi_squared:
                source.flags.setdefault(band_name, []).append("poor_fit")
            
            # Store position offset information
            source.fit_parameters[band_name] = {
                'flux': flux,
                'dx': dx,
                'dy': dy,
                'background': background
            }
            
        except Exception as e:
            self.logger.debug(f"Failed to store single-band results for source {source.id} in {band_name}: {e}")
    
    def _compute_statistics(self, sources: List[PSFPhotometrySource],
                           images: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Compute global statistics."""
        stats = {}
        
        band_names = list(images.keys())
        
        for band_name in band_names:
            band_stats = {}
            
            # Collect measurements for this band
            fluxes = [s.flux.get(band_name, 0) for s in sources if band_name in s.flux]
            magnitudes = [s.magnitude.get(band_name, 99) for s in sources if band_name in s.magnitude]
            chi_squareds = [s.chi_squared.get(band_name, 1e10) for s in sources if band_name in s.chi_squared]
            
            if fluxes:
                band_stats['n_detections'] = len(fluxes)
                band_stats['flux_median'] = np.median(fluxes)
                band_stats['flux_std'] = np.std(fluxes)
                
                valid_mags = [m for m in magnitudes if m < 90]
                if valid_mags:
                    band_stats['magnitude_median'] = np.median(valid_mags)
                    band_stats['magnitude_range'] = np.max(valid_mags) - np.min(valid_mags)
                
                if chi_squareds:
                    band_stats['chi_squared_median'] = np.median(chi_squareds)
                    band_stats['good_fits_fraction'] = np.mean([c < self.config.max_chi_squared for c in chi_squareds])
            
            stats[band_name] = band_stats
        
        # Global statistics
        stats['total_sources'] = len(sources)
        stats['blended_sources'] = len([s for s in sources if s.is_blended])
        
        return stats
    
    def _assess_quality(self, sources: List[PSFPhotometrySource],
                       images: Dict[str, np.ndarray]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Assess completeness and reliability."""
        completeness = {}
        reliability = {}
        
        band_names = list(images.keys())
        
        for band_name in band_names:
            # Simple quality assessment
            total_sources = len(sources)
            successful_fits = len([s for s in sources if band_name in s.flux and s.flux[band_name] > 0])
            good_fits = len([s for s in sources if band_name in s.chi_squared and s.chi_squared[band_name] < self.config.max_chi_squared])
            
            if total_sources > 0:
                completeness[band_name] = successful_fits / total_sources
                reliability[band_name] = good_fits / successful_fits if successful_fits > 0 else 0.0
            else:
                completeness[band_name] = 0.0
                reliability[band_name] = 0.0
        
        return completeness, reliability
    
    def _create_diagnostics(self, sources: List[PSFPhotometrySource],
                           psf_models: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Create diagnostic information."""
        diagnostics = {
            'config': self.config,
            'psf_fwhms': {band: self._estimate_psf_fwhm(psf) for band, psf in psf_models.items()},
            'processing_summary': {
                'total_sources': len(sources),
                'blended_sources': len([s for s in sources if s.is_blended]),
                'bands_processed': list(psf_models.keys())
            }
        }
        
        return diagnostics
    
    def plot_photometry_diagnostics(self, results: PSFPhotometryResults,
                                   output_path: Optional[str] = None) -> None:
        """Create diagnostic plots for PSF photometry."""
        try:
            band_names = list(results.psf_models.keys())
            n_bands = len(band_names)
            
            fig, axes = plt.subplots(2, min(n_bands, 3), figsize=(15, 10))
            if n_bands == 1:
                axes = axes.reshape(-1, 1)
            
            for i, band_name in enumerate(band_names[:3]):  # Limit to 3 bands
                col = i % 3
                
                # Flux distribution
                fluxes = [s.flux.get(band_name, 0) for s in results.sources if band_name in s.flux]
                if fluxes:
                    axes[0, col].hist(np.log10(np.maximum(fluxes, 1e-10)), bins=30, alpha=0.7)
                    axes[0, col].set_xlabel('Log10(Flux)')
                    axes[0, col].set_ylabel('Count')
                    axes[0, col].set_title(f'{band_name} - Flux Distribution')
                
                # Chi-squared distribution
                chi_squareds = [s.chi_squared.get(band_name, 1e10) for s in results.sources if band_name in s.chi_squared]
                if chi_squareds:
                    axes[1, col].hist(np.log10(chi_squareds), bins=30, alpha=0.7)
                    axes[1, col].axvline(np.log10(self.config.max_chi_squared), color='red', linestyle='--')
                    axes[1, col].set_xlabel('Log10(Chi-squared)')
                    axes[1, col].set_ylabel('Count')
                    axes[1, col].set_title(f'{band_name} - Fit Quality')
            
            # Remove empty subplots
            for j in range(len(band_names), 3):
                for row in range(2):
                    if j < axes.shape[1]:
                        axes[row, j].remove()
            
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                self.logger.info(f"PSF photometry diagnostics saved to {output_path}")
            else:
                plt.show()
            
            plt.close()
            
        except Exception as e:
            self.logger.warning(f"Failed to create diagnostic plots: {e}")


# Convenience functions
def perform_psf_photometry_simple(images: Dict[str, np.ndarray],
                                 psf_models: Dict[str, np.ndarray],
                                 sources: np.ndarray) -> PSFPhotometryResults:
    """
    Simplified PSF photometry function.
    
    Parameters:
    -----------
    images : dict
        Dictionary of images
    psf_models : dict
        Dictionary of PSF models
    sources : numpy.ndarray
        Source catalog
        
    Returns:
    --------
    PSFPhotometryResults
        PSF photometry results
    """
    config = PSFPhotometryConfig()
    processor = AdvancedPSFPhotometry(config)
    
    return processor.perform_psf_photometry(images, psf_models, sources)


def extract_photometry_catalog(results: PSFPhotometryResults) -> Table:
    """
    Extract photometry catalog from results.
    
    Parameters:
    -----------
    results : PSFPhotometryResults
        PSF photometry results
        
    Returns:
    --------
    Table
        Astropy table with photometry results
    """
    catalog_data = {}
    
    # Basic information
    catalog_data['id'] = [s.id for s in results.sources]
    catalog_data['x'] = [s.x for s in results.sources]
    catalog_data['y'] = [s.y for s in results.sources]
    
    if any(s.ra is not None for s in results.sources):
        catalog_data['ra'] = [s.ra if s.ra is not None else np.nan for s in results.sources]
        catalog_data['dec'] = [s.dec if s.dec is not None else np.nan for s in results.sources]
    
    # Add photometry for each band
    band_names = list(results.psf_models.keys())
    
    for band_name in band_names:
        catalog_data[f'flux_{band_name}'] = [s.flux.get(band_name, np.nan) for s in results.sources]
        catalog_data[f'flux_error_{band_name}'] = [s.flux_error.get(band_name, np.nan) for s in results.sources]
        catalog_data[f'mag_{band_name}'] = [s.magnitude.get(band_name, np.nan) for s in results.sources]
        catalog_data[f'mag_error_{band_name}'] = [s.magnitude_error.get(band_name, np.nan) for s in results.sources]
        catalog_data[f'chi2_{band_name}'] = [s.chi_squared.get(band_name, np.nan) for s in results.sources]
    
    # Quality flags
    catalog_data['is_blended'] = [s.is_blended for s in results.sources]
    
    return Table(catalog_data)
