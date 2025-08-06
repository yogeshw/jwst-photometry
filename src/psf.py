"""
Advanced PSF Processing Module for JWST Photometry

This module implements sophisticated PSF generation, modeling, and matching
capabilities specifically designed for JWST NIRCam observations.
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple, List, Union
from dataclasses import dataclass, field
from pathlib import Path

from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.table import Table
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
from scipy import ndimage, optimize
from scipy.ndimage import center_of_mass, zoom, binary_erosion
from scipy.signal import convolve2d
from scipy.interpolate import griddata, interp2d, RegularGridInterpolator
from skimage import filters, measure
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

try:
    import pypher
    PYPHER_AVAILABLE = True
except ImportError:
    PYPHER_AVAILABLE = False
    logging.warning("Pypher not available - PSF matching functionality will be limited")


@dataclass
class PSFConfig:
    """Configuration parameters for PSF processing."""
    
    # Empirical PSF generation
    stamp_size: int = 64
    min_star_snr: float = 50.0
    max_star_magnitude: float = 18.0
    min_star_magnitude: float = 12.0
    star_isolation_radius: float = 10.0
    max_elongation: float = 1.2
    
    # PSF model fitting
    psf_model_type: str = "moffat"  # "gaussian", "moffat", "gaussian_plus_moffat"
    fit_centroid: bool = True
    fit_background: bool = True
    
    # Spatial variation modeling
    model_spatial_variation: bool = True
    spatial_order: int = 2
    min_stars_per_bin: int = 5
    spatial_grid_size: int = 5
    
    # Quality assessment
    quality_checks: bool = True
    max_fwhm_variation: float = 0.5
    max_ellipticity: float = 0.3
    min_flux_ratio: float = 0.1
    max_flux_ratio: float = 10.0
    
    # PSF matching
    matching_method: str = "pypher"  # "pypher", "interpolation"
    regularization_parameter: float = 1e-6
    kernel_size: int = 15
    
    # Output control
    save_diagnostics: bool = True
    diagnostic_plots: bool = True
    verbose: bool = True


@dataclass
class PSFStarData:
    """Container for PSF star information."""
    
    id: int
    x: float
    y: float
    ra: float
    dec: float
    magnitude: float
    snr: float
    fwhm: float
    ellipticity: float
    position_angle: float
    is_valid: bool = True
    rejection_reason: Optional[str] = None
    
    # PSF stamp and fit results
    stamp: Optional[np.ndarray] = None
    fitted_psf: Optional[np.ndarray] = None
    fit_parameters: Optional[Dict[str, float]] = None
    fit_quality: Optional[float] = None


@dataclass
class PSFModelResults:
    """Container for PSF modeling results."""
    
    # Empirical PSF
    empirical_psf: np.ndarray
    psf_fwhm: float
    psf_ellipticity: float
    psf_position_angle: float
    
    # Star selection results
    selected_stars: List[PSFStarData]
    rejected_stars: List[PSFStarData]
    
    # Spatial variation model
    spatial_variation_model: Optional[Dict[str, Any]] = None
    spatially_varying_psf: Optional[np.ndarray] = None
    
    # Quality metrics
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Diagnostic information
    diagnostics: Dict[str, Any] = field(default_factory=dict)


class AdvancedPSFProcessor:
    """
    Advanced PSF processor for JWST NIRCam observations.
    
    This class provides comprehensive PSF processing capabilities including:
    - Robust star selection based on multiple criteria
    - Empirical PSF generation with quality assessment
    - Spatial variation modeling
    - PSF model fitting with multiple models
    - Sub-pixel centering and normalization
    - Comprehensive quality assessment and diagnostics
    """
    
    def __init__(self, config: Optional[PSFConfig] = None):
        """
        Initialize the PSF processor.
        
        Parameters:
        -----------
        config : PSFConfig, optional
            PSF processing configuration. If None, uses defaults.
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or PSFConfig()
        
        if not PYPHER_AVAILABLE and self.config.matching_method == "pypher":
            self.logger.warning("Pypher not available, switching to interpolation method")
            self.config.matching_method = "interpolation"
    
    def generate_empirical_psf(self, 
                             image: np.ndarray,
                             sources: np.ndarray,
                             wcs: Optional[WCS] = None,
                             background_map: Optional[np.ndarray] = None,
                             rms_map: Optional[np.ndarray] = None) -> PSFModelResults:
        """
        Generate empirical PSF from stellar sources in the image.
        
        Parameters:
        -----------
        image : numpy.ndarray
            Input image data
        sources : numpy.ndarray
            Detected sources catalog
        wcs : astropy.wcs.WCS, optional
            World coordinate system
        background_map : numpy.ndarray, optional
            Background map
        rms_map : numpy.ndarray, optional
            RMS/noise map
            
        Returns:
        --------
        PSFModelResults
            Complete PSF modeling results
        """
        self.logger.info("Starting empirical PSF generation")
        
        # Select PSF stars
        psf_stars = self._select_psf_stars(image, sources, wcs, background_map, rms_map)
        
        if len(psf_stars) == 0:
            raise ValueError("No suitable PSF stars found")
        
        self.logger.info(f"Selected {len(psf_stars)} PSF stars from {len(sources)} sources")
        
        # Extract and process PSF stamps
        valid_stars = self._extract_psf_stamps(image, psf_stars, background_map)
        
        if len(valid_stars) == 0:
            raise ValueError("No valid PSF stamps extracted")
        
        # Fit PSF models to individual stars
        fitted_stars = self._fit_individual_psfs(valid_stars)
        
        # Create empirical PSF
        empirical_psf = self._create_empirical_psf(fitted_stars)
        
        # Measure PSF properties
        psf_properties = self._measure_psf_properties(empirical_psf)
        
        # Model spatial variation if requested
        spatial_model = None
        spatially_varying_psf = None
        if self.config.model_spatial_variation and len(fitted_stars) >= 10:
            spatial_model = self._model_spatial_variation(fitted_stars, image.shape)
            spatially_varying_psf = self._generate_spatially_varying_psf(
                spatial_model, image.shape
            )
        
        # Compute quality metrics
        quality_metrics = self._compute_quality_metrics(fitted_stars, empirical_psf)
        
        # Create diagnostics
        diagnostics = self._create_diagnostics(fitted_stars, empirical_psf)
        
        # Separate valid and rejected stars
        valid_stars_list = [star for star in psf_stars if star.is_valid]
        rejected_stars_list = [star for star in psf_stars if not star.is_valid]
        
        results = PSFModelResults(
            empirical_psf=empirical_psf,
            psf_fwhm=psf_properties['fwhm'],
            psf_ellipticity=psf_properties['ellipticity'],
            psf_position_angle=psf_properties['position_angle'],
            selected_stars=valid_stars_list,
            rejected_stars=rejected_stars_list,
            spatial_variation_model=spatial_model,
            spatially_varying_psf=spatially_varying_psf,
            quality_metrics=quality_metrics,
            diagnostics=diagnostics
        )
        
        self.logger.info("Empirical PSF generation completed successfully")
        return results
    
    def _select_psf_stars(self, 
                         image: np.ndarray,
                         sources: np.ndarray,
                         wcs: Optional[WCS],
                         background_map: Optional[np.ndarray],
                         rms_map: Optional[np.ndarray]) -> List[PSFStarData]:
        """
        Select suitable stars for PSF modeling using multiple criteria.
        
        Parameters:
        -----------
        image : numpy.ndarray
            Input image
        sources : numpy.ndarray
            Source catalog
        wcs : astropy.wcs.WCS, optional
            World coordinate system
        background_map : numpy.ndarray, optional
            Background map
        rms_map : numpy.ndarray, optional
            RMS map
            
        Returns:
        --------
        list
            List of PSF star candidates
        """
        self.logger.debug("Selecting PSF stars")
        
        psf_stars = []
        
        for i, source in enumerate(sources):
            star = PSFStarData(
                id=i,
                x=source['x'],
                y=source['y'],
                ra=0.0,  # Will be filled if WCS available
                dec=0.0,
                magnitude=0.0,  # Will be computed
                snr=0.0,
                fwhm=2.0 * np.sqrt(source['a'] * source['b']),
                ellipticity=1.0 - source['b'] / source['a'],
                position_angle=source['theta']
            )
            
            # Convert to sky coordinates if WCS available
            if wcs is not None:
                try:
                    sky_coord = wcs.pixel_to_world(star.x, star.y)
                    star.ra = sky_coord.ra.degree
                    star.dec = sky_coord.dec.degree
                except Exception:
                    pass
            
            # Apply selection criteria
            rejection_reason = self._evaluate_star_criteria(star, source, image, 
                                                          background_map, rms_map)
            
            if rejection_reason is None:
                star.is_valid = True
                psf_stars.append(star)
            else:
                star.is_valid = False
                star.rejection_reason = rejection_reason
                psf_stars.append(star)
        
        # Apply isolation criterion
        self._check_star_isolation(psf_stars)
        
        return psf_stars
    
    def _evaluate_star_criteria(self, 
                               star: PSFStarData,
                               source: np.ndarray,
                               image: np.ndarray,
                               background_map: Optional[np.ndarray],
                               rms_map: Optional[np.ndarray]) -> Optional[str]:
        """
        Evaluate star selection criteria.
        
        Parameters:
        -----------
        star : PSFStarData
            Star data
        source : numpy.ndarray
            Source catalog entry
        image : numpy.ndarray
            Input image
        background_map : numpy.ndarray, optional
            Background map
        rms_map : numpy.ndarray, optional
            RMS map
            
        Returns:
        --------
        str or None
            Rejection reason if star rejected, None if accepted
        """
        # Check boundary conditions
        half_stamp = self.config.stamp_size // 2
        if (star.x < half_stamp or star.x >= image.shape[1] - half_stamp or
            star.y < half_stamp or star.y >= image.shape[0] - half_stamp):
            return "too_close_to_boundary"
        
        # Check elongation (ellipticity)
        if star.ellipticity > (self.config.max_elongation - 1.0):
            return "too_elongated"
        
        # Estimate magnitude and SNR
        flux = source['flux']
        if background_map is not None:
            bkg_value = background_map[int(star.y), int(star.x)]
        else:
            bkg_value = 0.0
        
        if rms_map is not None:
            noise_value = rms_map[int(star.y), int(star.x)]
        else:
            noise_value = np.nanstd(image)
        
        star.snr = flux / noise_value if noise_value > 0 else 0.0
        star.magnitude = -2.5 * np.log10(max(flux, 1e-10)) + 25.0  # Arbitrary zeropoint
        
        # Check SNR
        if star.snr < self.config.min_star_snr:
            return "low_snr"
        
        # Check magnitude range
        if (star.magnitude < self.config.min_star_magnitude or 
            star.magnitude > self.config.max_star_magnitude):
            return "magnitude_out_of_range"
        
        # Check for saturation (very simple check)
        peak_value = image[int(star.y), int(star.x)]
        if peak_value > 0.9 * np.max(image):  # Rough saturation check
            return "saturated"
        
        return None
    
    def _check_star_isolation(self, psf_stars: List[PSFStarData]) -> None:
        """
        Check star isolation criterion.
        
        Parameters:
        -----------
        psf_stars : list
            List of PSF star candidates
        """
        valid_stars = [star for star in psf_stars if star.is_valid]
        
        for i, star1 in enumerate(valid_stars):
            for j, star2 in enumerate(valid_stars[i+1:], i+1):
                distance = np.sqrt((star1.x - star2.x)**2 + (star1.y - star2.y)**2)
                
                if distance < self.config.star_isolation_radius:
                    # Keep the brighter star
                    if star1.snr > star2.snr:
                        star2.is_valid = False
                        star2.rejection_reason = "not_isolated"
                    else:
                        star1.is_valid = False
                        star1.rejection_reason = "not_isolated"
    
    def _extract_psf_stamps(self, 
                           image: np.ndarray,
                           psf_stars: List[PSFStarData],
                           background_map: Optional[np.ndarray]) -> List[PSFStarData]:
        """
        Extract PSF stamps around selected stars.
        
        Parameters:
        -----------
        image : numpy.ndarray
            Input image
        psf_stars : list
            List of PSF stars
        background_map : numpy.ndarray, optional
            Background map
            
        Returns:
        --------
        list
            List of stars with extracted stamps
        """
        self.logger.debug("Extracting PSF stamps")
        
        valid_stars = []
        
        for star in psf_stars:
            if not star.is_valid:
                continue
            
            try:
                # Extract stamp
                cutout = Cutout2D(
                    image, (star.x, star.y), 
                    self.config.stamp_size, mode='trim'
                )
                
                if cutout.data.shape != (self.config.stamp_size, self.config.stamp_size):
                    star.is_valid = False
                    star.rejection_reason = "incomplete_stamp"
                    continue
                
                stamp = cutout.data.copy()
                
                # Subtract background if available
                if background_map is not None:
                    bkg_cutout = Cutout2D(
                        background_map, (star.x, star.y),
                        self.config.stamp_size, mode='trim'
                    )
                    stamp = stamp - bkg_cutout.data
                
                # Check for negative or zero values
                if np.sum(stamp <= 0) > 0.1 * stamp.size:
                    star.is_valid = False
                    star.rejection_reason = "negative_values"
                    continue
                
                # Center the stamp
                centered_stamp = self._center_stamp(stamp)
                
                if centered_stamp is None:
                    star.is_valid = False
                    star.rejection_reason = "centering_failed"
                    continue
                
                star.stamp = centered_stamp
                valid_stars.append(star)
                
            except Exception as e:
                self.logger.debug(f"Failed to extract stamp for star {star.id}: {e}")
                star.is_valid = False
                star.rejection_reason = "extraction_failed"
        
        return valid_stars
    
    def _center_stamp(self, stamp: np.ndarray) -> Optional[np.ndarray]:
        """
        Center a PSF stamp using center of mass.
        
        Parameters:
        -----------
        stamp : numpy.ndarray
            PSF stamp
            
        Returns:
        --------
        numpy.ndarray or None
            Centered stamp, or None if centering failed
        """
        try:
            # Compute center of mass
            com = center_of_mass(stamp)
            
            # Calculate shifts needed
            center_pos = (stamp.shape[0] - 1) / 2.0
            shift_y = center_pos - com[0]
            shift_x = center_pos - com[1]
            
            # Check if shift is reasonable
            if abs(shift_x) > 2 or abs(shift_y) > 2:
                return None
            
            # Apply sub-pixel shift using interpolation
            from scipy.ndimage import shift
            centered_stamp = shift(stamp, (shift_y, shift_x), order=3, mode='constant', cval=0)
            
            return centered_stamp
            
        except Exception:
            return None
    
    def _fit_individual_psfs(self, psf_stars: List[PSFStarData]) -> List[PSFStarData]:
        """
        Fit PSF models to individual stars.
        
        Parameters:
        -----------
        psf_stars : list
            List of PSF stars with stamps
            
        Returns:
        --------
        list
            List of stars with fitted PSF models
        """
        self.logger.debug("Fitting individual PSF models")
        
        fitted_stars = []
        
        for star in psf_stars:
            try:
                if star.stamp is None:
                    continue
                
                # Fit PSF model
                fit_result = self._fit_psf_model(star.stamp)
                
                if fit_result is not None:
                    star.fitted_psf = fit_result['fitted_psf']
                    star.fit_parameters = fit_result['parameters']
                    star.fit_quality = fit_result['quality']
                    fitted_stars.append(star)
                
            except Exception as e:
                self.logger.debug(f"PSF fitting failed for star {star.id}: {e}")
        
        return fitted_stars
    
    def _fit_psf_model(self, stamp: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Fit a PSF model to a stamp.
        
        Parameters:
        -----------
        stamp : numpy.ndarray
            PSF stamp
            
        Returns:
        --------
        dict or None
            Fit results or None if fitting failed
        """
        try:
            # Create coordinate grids
            y, x = np.ogrid[:stamp.shape[0], :stamp.shape[1]]
            center = (stamp.shape[0] - 1) / 2.0
            
            # Initial parameter guess
            amplitude_guess = np.max(stamp)
            fwhm_guess = 3.0
            background_guess = np.median(stamp[stamp < np.percentile(stamp, 20)])
            
            if self.config.psf_model_type == "gaussian":
                result = self._fit_gaussian(stamp, x, y, center, amplitude_guess, fwhm_guess, background_guess)
            elif self.config.psf_model_type == "moffat":
                result = self._fit_moffat(stamp, x, y, center, amplitude_guess, fwhm_guess, background_guess)
            else:
                result = self._fit_moffat(stamp, x, y, center, amplitude_guess, fwhm_guess, background_guess)
            
            return result
            
        except Exception:
            return None
    
    def _fit_moffat(self, stamp: np.ndarray, x: np.ndarray, y: np.ndarray,
                   center: float, amplitude_guess: float, fwhm_guess: float,
                   background_guess: float) -> Optional[Dict[str, Any]]:
        """
        Fit a Moffat profile to the PSF stamp.
        
        Parameters:
        -----------
        stamp : numpy.ndarray
            PSF stamp
        x, y : numpy.ndarray
            Coordinate grids
        center : float
            Center position
        amplitude_guess : float
            Initial amplitude guess
        fwhm_guess : float
            Initial FWHM guess
        background_guess : float
            Initial background guess
            
        Returns:
        --------
        dict or None
            Fit results
        """
        def moffat_2d(coords, amplitude, x0, y0, alpha, beta, background):
            x, y = coords
            r_squared = ((x - x0)**2 + (y - y0)**2) / alpha**2
            return background + amplitude / (1 + r_squared)**beta
        
        # Flatten arrays for fitting
        coords = np.vstack([x.ravel(), y.ravel()])
        data = stamp.ravel()
        
        # Initial parameters
        p0 = [
            amplitude_guess,  # amplitude
            center,           # x0
            center,           # y0
            fwhm_guess / 2.0, # alpha
            2.5,              # beta
            background_guess  # background
        ]
        
        try:
            # Fit the model
            popt, pcov = optimize.curve_fit(
                moffat_2d, coords, data, p0=p0,
                bounds=([0, center-1, center-1, 0.5, 1.0, 0],
                       [np.inf, center+1, center+1, 10.0, 10.0, np.inf]),
                maxfev=1000
            )
            
            # Generate fitted PSF
            fitted_psf = moffat_2d(coords, *popt).reshape(stamp.shape)
            
            # Calculate quality metric (R-squared)
            ss_res = np.sum((data - fitted_psf.ravel())**2)
            ss_tot = np.sum((data - np.mean(data))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            parameters = {
                'amplitude': popt[0],
                'x0': popt[1],
                'y0': popt[2],
                'alpha': popt[3],
                'beta': popt[4],
                'background': popt[5],
                'fwhm': 2 * popt[3] * np.sqrt(2**(1/popt[4]) - 1)
            }
            
            return {
                'fitted_psf': fitted_psf,
                'parameters': parameters,
                'quality': r_squared
            }
            
        except Exception:
            return None
    
    def _fit_gaussian(self, stamp: np.ndarray, x: np.ndarray, y: np.ndarray,
                     center: float, amplitude_guess: float, fwhm_guess: float,
                     background_guess: float) -> Optional[Dict[str, Any]]:
        """
        Fit a Gaussian profile to the PSF stamp.
        
        Parameters:
        -----------
        stamp : numpy.ndarray
            PSF stamp
        x, y : numpy.ndarray
            Coordinate grids
        center : float
            Center position
        amplitude_guess : float
            Initial amplitude guess
        fwhm_guess : float
            Initial FWHM guess
        background_guess : float
            Initial background guess
            
        Returns:
        --------
        dict or None
            Fit results
        """
        def gaussian_2d(coords, amplitude, x0, y0, sigma_x, sigma_y, theta, background):
            x, y = coords
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            
            x_rot = (x - x0) * cos_theta + (y - y0) * sin_theta
            y_rot = -(x - x0) * sin_theta + (y - y0) * cos_theta
            
            gaussian = amplitude * np.exp(-(x_rot**2/(2*sigma_x**2) + y_rot**2/(2*sigma_y**2)))
            return background + gaussian
        
        # Flatten arrays for fitting
        coords = np.vstack([x.ravel(), y.ravel()])
        data = stamp.ravel()
        
        sigma_guess = fwhm_guess / (2 * np.sqrt(2 * np.log(2)))
        
        # Initial parameters
        p0 = [
            amplitude_guess,  # amplitude
            center,           # x0
            center,           # y0
            sigma_guess,      # sigma_x
            sigma_guess,      # sigma_y
            0.0,              # theta
            background_guess  # background
        ]
        
        try:
            # Fit the model
            popt, pcov = optimize.curve_fit(
                gaussian_2d, coords, data, p0=p0,
                bounds=([0, center-1, center-1, 0.5, 0.5, -np.pi, 0],
                       [np.inf, center+1, center+1, 10.0, 10.0, np.pi, np.inf]),
                maxfev=1000
            )
            
            # Generate fitted PSF
            fitted_psf = gaussian_2d(coords, *popt).reshape(stamp.shape)
            
            # Calculate quality metric
            ss_res = np.sum((data - fitted_psf.ravel())**2)
            ss_tot = np.sum((data - np.mean(data))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            fwhm_x = 2 * np.sqrt(2 * np.log(2)) * popt[3]
            fwhm_y = 2 * np.sqrt(2 * np.log(2)) * popt[4]
            
            parameters = {
                'amplitude': popt[0],
                'x0': popt[1],
                'y0': popt[2],
                'sigma_x': popt[3],
                'sigma_y': popt[4],
                'theta': popt[5],
                'background': popt[6],
                'fwhm': np.sqrt(fwhm_x * fwhm_y)
            }
            
            return {
                'fitted_psf': fitted_psf,
                'parameters': parameters,
                'quality': r_squared
            }
            
        except Exception:
            return None
    
    def _create_empirical_psf(self, fitted_stars: List[PSFStarData]) -> np.ndarray:
        """
        Create empirical PSF from fitted individual PSFs.
        
        Parameters:
        -----------
        fitted_stars : list
            List of stars with fitted PSF models
            
        Returns:
        --------
        numpy.ndarray
            Empirical PSF
        """
        if not fitted_stars:
            raise ValueError("No fitted stars available for empirical PSF creation")
        
        # Collect PSF stamps
        psf_stamps = []
        weights = []
        
        for star in fitted_stars:
            if star.fitted_psf is not None and star.fit_quality is not None:
                # Normalize the PSF
                normalized_psf = star.fitted_psf / np.sum(star.fitted_psf)
                psf_stamps.append(normalized_psf)
                
                # Weight by fit quality and SNR
                weight = star.fit_quality * np.log10(star.snr)
                weights.append(weight)
        
        if not psf_stamps:
            # Fallback to stamp data
            for star in fitted_stars:
                if star.stamp is not None:
                    normalized_stamp = star.stamp / np.sum(star.stamp)
                    psf_stamps.append(normalized_stamp)
                    weights.append(star.snr)
        
        psf_stamps = np.array(psf_stamps)
        weights = np.array(weights)
        
        # Create weighted average
        if len(weights) > 0 and np.sum(weights) > 0:
            weights = weights / np.sum(weights)
            empirical_psf = np.average(psf_stamps, axis=0, weights=weights)
        else:
            empirical_psf = np.mean(psf_stamps, axis=0)
        
        # Final normalization
        empirical_psf = empirical_psf / np.sum(empirical_psf)
        
        return empirical_psf
    
    def _measure_psf_properties(self, psf: np.ndarray) -> Dict[str, float]:
        """
        Measure basic properties of the PSF.
        
        Parameters:
        -----------
        psf : numpy.ndarray
            PSF array
            
        Returns:
        --------
        dict
            PSF properties
        """
        # Calculate second moments
        y, x = np.ogrid[:psf.shape[0], :psf.shape[1]]
        
        # Center of mass
        total_flux = np.sum(psf)
        x_center = np.sum(x * psf) / total_flux
        y_center = np.sum(y * psf) / total_flux
        
        # Second moments
        m_xx = np.sum((x - x_center)**2 * psf) / total_flux
        m_yy = np.sum((y - y_center)**2 * psf) / total_flux
        m_xy = np.sum((x - x_center) * (y - y_center) * psf) / total_flux
        
        # Derived properties
        fwhm = 2.0 * np.sqrt(np.log(2) * (m_xx + m_yy))
        ellipticity = np.sqrt(m_xy**2 + ((m_xx - m_yy)/2)**2) / (m_xx + m_yy)
        position_angle = 0.5 * np.arctan2(2 * m_xy, m_xx - m_yy)
        
        return {
            'fwhm': fwhm,
            'ellipticity': ellipticity,
            'position_angle': position_angle,
            'x_center': x_center,
            'y_center': y_center
        }
    
    def _model_spatial_variation(self, fitted_stars: List[PSFStarData], 
                               image_shape: Tuple[int, int]) -> Dict[str, Any]:
        """
        Model spatial variation of PSF across the field.
        
        Parameters:
        -----------
        fitted_stars : list
            List of fitted stars
        image_shape : tuple
            Shape of the image
            
        Returns:
        --------
        dict
            Spatial variation model
        """
        self.logger.debug("Modeling PSF spatial variation")
        
        # Extract star positions and PSF parameters
        positions = np.array([[star.x, star.y] for star in fitted_stars])
        fwhms = np.array([star.fit_parameters.get('fwhm', 3.0) for star in fitted_stars 
                         if star.fit_parameters is not None])
        
        if len(positions) < self.config.min_stars_per_bin:
            return {'model_type': 'constant', 'fwhm': np.median(fwhms)}
        
        # Fit polynomial surface to FWHM variation
        try:
            # Normalize coordinates
            x_norm = positions[:, 0] / image_shape[1]
            y_norm = positions[:, 1] / image_shape[0]
            
            # Create polynomial features
            order = self.config.spatial_order
            features = []
            
            for i in range(order + 1):
                for j in range(order + 1 - i):
                    features.append((x_norm**i) * (y_norm**j))
            
            design_matrix = np.column_stack(features)
            
            # Fit polynomial
            coefficients, residuals, rank, s = np.linalg.lstsq(
                design_matrix, fwhms, rcond=None
            )
            
            # Calculate quality of fit
            predicted_fwhms = design_matrix @ coefficients
            r_squared = 1 - np.sum((fwhms - predicted_fwhms)**2) / np.sum((fwhms - np.mean(fwhms))**2)
            
            return {
                'model_type': 'polynomial',
                'order': order,
                'coefficients': coefficients,
                'r_squared': r_squared,
                'residual_std': np.std(fwhms - predicted_fwhms)
            }
            
        except Exception as e:
            self.logger.warning(f"Spatial variation modeling failed: {e}")
            return {'model_type': 'constant', 'fwhm': np.median(fwhms)}
    
    def _generate_spatially_varying_psf(self, spatial_model: Dict[str, Any],
                                      image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Generate spatially varying PSF map.
        
        Parameters:
        -----------
        spatial_model : dict
            Spatial variation model
        image_shape : tuple
            Shape of the image
            
        Returns:
        --------
        numpy.ndarray
            Spatially varying PSF parameters
        """
        if spatial_model['model_type'] == 'constant':
            return np.full(image_shape, spatial_model['fwhm'])
        
        # Generate coordinate grids
        y_grid, x_grid = np.ogrid[:image_shape[0], :image_shape[1]]
        x_norm = x_grid / image_shape[1]
        y_norm = y_grid / image_shape[0]
        
        # Evaluate polynomial
        order = spatial_model['order']
        coefficients = spatial_model['coefficients']
        
        psf_map = np.zeros(image_shape)
        coeff_idx = 0
        
        for i in range(order + 1):
            for j in range(order + 1 - i):
                psf_map += coefficients[coeff_idx] * (x_norm**i) * (y_norm**j)
                coeff_idx += 1
        
        return psf_map
    
    def _compute_quality_metrics(self, fitted_stars: List[PSFStarData],
                               empirical_psf: np.ndarray) -> Dict[str, float]:
        """
        Compute quality metrics for PSF modeling.
        
        Parameters:
        -----------
        fitted_stars : list
            List of fitted stars
        empirical_psf : numpy.ndarray
            Empirical PSF
            
        Returns:
        --------
        dict
            Quality metrics
        """
        metrics = {}
        
        if fitted_stars:
            # FWHM statistics
            fwhms = [star.fit_parameters.get('fwhm', 3.0) for star in fitted_stars 
                    if star.fit_parameters is not None]
            if fwhms:
                metrics['fwhm_mean'] = np.mean(fwhms)
                metrics['fwhm_std'] = np.std(fwhms)
                metrics['fwhm_variation'] = np.std(fwhms) / np.mean(fwhms)
            
            # Fit quality statistics
            qualities = [star.fit_quality for star in fitted_stars 
                        if star.fit_quality is not None]
            if qualities:
                metrics['fit_quality_mean'] = np.mean(qualities)
                metrics['fit_quality_std'] = np.std(qualities)
            
            # Number of stars used
            metrics['n_stars_used'] = len(fitted_stars)
        
        # PSF properties
        psf_props = self._measure_psf_properties(empirical_psf)
        metrics.update(psf_props)
        
        return metrics
    
    def _create_diagnostics(self, fitted_stars: List[PSFStarData],
                          empirical_psf: np.ndarray) -> Dict[str, Any]:
        """
        Create diagnostic information.
        
        Parameters:
        -----------
        fitted_stars : list
            List of fitted stars
        empirical_psf : numpy.ndarray
            Empirical PSF
            
        Returns:
        --------
        dict
            Diagnostic information
        """
        diagnostics = {
            'n_stars_total': len(fitted_stars),
            'empirical_psf_shape': empirical_psf.shape,
            'empirical_psf_peak': np.max(empirical_psf),
            'empirical_psf_sum': np.sum(empirical_psf)
        }
        
        return diagnostics
    
    def plot_psf_diagnostics(self, results: PSFModelResults, 
                           output_path: Optional[str] = None) -> None:
        """
        Create diagnostic plots for PSF modeling.
        
        Parameters:
        -----------
        results : PSFModelResults
            PSF modeling results
        output_path : str, optional
            Path to save the plot
        """
        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            # Empirical PSF
            im1 = axes[0, 0].imshow(results.empirical_psf, origin='lower', cmap='viridis')
            axes[0, 0].set_title(f'Empirical PSF\\nFWHM: {results.psf_fwhm:.2f}')
            plt.colorbar(im1, ax=axes[0, 0])
            
            # PSF radial profile
            center = (results.empirical_psf.shape[0] - 1) / 2
            y, x = np.ogrid[:results.empirical_psf.shape[0], :results.empirical_psf.shape[1]]
            r = np.sqrt((x - center)**2 + (y - center)**2)
            
            # Azimuthally average the PSF
            r_int = r.astype(int)
            r_max = int(center)
            radii = np.arange(r_max)
            profile = np.array([np.mean(results.empirical_psf[r_int == radius]) for radius in radii])
            
            axes[0, 1].plot(radii, profile, 'b-', linewidth=2)
            axes[0, 1].set_xlabel('Radius (pixels)')
            axes[0, 1].set_ylabel('Normalized Flux')
            axes[0, 1].set_title('PSF Radial Profile')
            axes[0, 1].set_yscale('log')
            
            # Star positions
            if results.selected_stars:
                x_positions = [star.x for star in results.selected_stars]
                y_positions = [star.y for star in results.selected_stars]
                axes[0, 2].scatter(x_positions, y_positions, c='blue', s=50, alpha=0.7)
                axes[0, 2].set_xlabel('X Position')
                axes[0, 2].set_ylabel('Y Position')
                axes[0, 2].set_title(f'PSF Stars ({len(results.selected_stars)} selected)')
                axes[0, 2].grid(True, alpha=0.3)
            
            # FWHM distribution
            if results.selected_stars:
                fwhms = [star.fit_parameters.get('fwhm', 3.0) for star in results.selected_stars 
                        if star.fit_parameters is not None]
                if fwhms:
                    axes[1, 0].hist(fwhms, bins=20, alpha=0.7, edgecolor='black')
                    axes[1, 0].axvline(np.mean(fwhms), color='red', linestyle='--', 
                                     label=f'Mean: {np.mean(fwhms):.2f}')
                    axes[1, 0].set_xlabel('FWHM (pixels)')
                    axes[1, 0].set_ylabel('Count')
                    axes[1, 0].set_title('FWHM Distribution')
                    axes[1, 0].legend()
            
            # Fit quality distribution
            if results.selected_stars:
                qualities = [star.fit_quality for star in results.selected_stars 
                           if star.fit_quality is not None]
                if qualities:
                    axes[1, 1].hist(qualities, bins=20, alpha=0.7, edgecolor='black')
                    axes[1, 1].axvline(np.mean(qualities), color='red', linestyle='--',
                                     label=f'Mean: {np.mean(qualities):.3f}')
                    axes[1, 1].set_xlabel('Fit Quality (R²)')
                    axes[1, 1].set_ylabel('Count')
                    axes[1, 1].set_title('Fit Quality Distribution')
                    axes[1, 1].legend()
            
            # Quality metrics text
            metrics_text = []
            for key, value in results.quality_metrics.items():
                if isinstance(value, float):
                    metrics_text.append(f'{key}: {value:.3f}')
                else:
                    metrics_text.append(f'{key}: {value}')
            
            axes[1, 2].text(0.1, 0.9, '\\n'.join(metrics_text), 
                           transform=axes[1, 2].transAxes, fontsize=10,
                           verticalalignment='top', fontfamily='monospace')
            axes[1, 2].set_title('Quality Metrics')
            axes[1, 2].axis('off')
            
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                self.logger.info(f"PSF diagnostics saved to {output_path}")
            else:
                plt.show()
            
            plt.close()
            
        except Exception as e:
            self.logger.warning(f"Failed to create diagnostic plots: {e}")


# Legacy compatibility functions
def generate_empirical_psf(image, sources, stamp_size=32):
    """
    Legacy function for generating empirical PSF.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image data
    sources : list
        List of detected sources
    stamp_size : int
        Size of PSF stamps
        
    Returns:
    --------
    numpy.ndarray
        Empirical PSF
    """
    config = PSFConfig(stamp_size=stamp_size)
    processor = AdvancedPSFProcessor(config)
    
    # Convert sources to structured array if needed
    if isinstance(sources, list):
        # Create minimal structured array
        dtype = [('x', 'f4'), ('y', 'f4'), ('a', 'f4'), ('b', 'f4'), 
                ('theta', 'f4'), ('flux', 'f4')]
        sources_array = np.array([(s.get('x', 0), s.get('y', 0), s.get('a', 2), 
                                 s.get('b', 2), s.get('theta', 0), s.get('flux', 1000))
                                for s in sources], dtype=dtype)
    else:
        sources_array = sources
    
    try:
        results = processor.generate_empirical_psf(image, sources_array)
        return results.empirical_psf
    except Exception as e:
        logging.error(f"PSF generation failed: {e}")
        # Return simple Gaussian as fallback
        psf_size = stamp_size
        y, x = np.ogrid[:psf_size, :psf_size]
        center = (psf_size - 1) / 2
        sigma = 2.0
        psf = np.exp(-((x - center)**2 + (y - center)**2) / (2 * sigma**2))
        return psf / np.sum(psf)


def match_psf(psf, target_psf, regularization_parameter):
    """
    Legacy function for PSF matching.
    
    Parameters:
    -----------
    psf : numpy.ndarray
        Source PSF
    target_psf : numpy.ndarray
        Target PSF
    regularization_parameter : float
        Regularization parameter
        
    Returns:
    --------
    numpy.ndarray
        PSF matching kernel
    """
    try:
        if PYPHER_AVAILABLE:
            kernel = pypher.psf_match(psf, target_psf, regularization_parameter)
            return kernel
        else:
            logging.warning("Pypher not available, returning identity kernel")
            return np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    except Exception as e:
        logging.error(f"PSF matching failed: {e}")
        return np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])


def apply_kernel(image, kernel):
    """
    Legacy function for applying PSF kernel.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image
    kernel : numpy.ndarray
        PSF matching kernel
        
    Returns:
    --------
    numpy.ndarray
        Convolved image
    """
    try:
        return convolve2d(image, kernel, mode='same')
    except Exception as e:
        logging.error(f"Kernel application failed: {e}")
        return image
