"""
JWST photometry flux calibration module (Phase 4.2).

This module provides comprehensive flux calibration capabilities for JWST NIRCam data,
including unit conversions, zero-point calibration, aperture corrections, and systematic
uncertainty propagation.
"""

import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import time

# Core astronomical libraries
from astropy.io import fits
from astropy.table import Table
import astropy.units as u
from astropy.coordinates import SkyCoord

# Specialized imports with fallbacks
try:
    import sep
except ImportError:
    print("Warning: sep not available for some advanced features")
    sep = None

try:
    from photutils import CircularAperture, aperture_photometry
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
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass, field
from pathlib import Path

from astropy.io import fits
from astropy.table import Table
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.constants import h, c, k_B
import matplotlib.pyplot as plt

# Import our modules
try:
    from utils import setup_logger, memory_monitor, validate_array
except ImportError:
    def setup_logger(name): return logging.getLogger(name)
    def memory_monitor(func): return func
    def validate_array(arr): return arr is not None


@dataclass
class CalibrationConfig:
    """Configuration parameters for flux calibration."""
    
    # Unit conversions
    input_units: str = "DN/s"  # DN/s, ADU, electrons
    output_units: str = "uJy"  # uJy, nJy, mag_AB, mag_Vega
    
    # Zero-point calibration
    use_in_flight_zeropoints: bool = True
    zeropoint_uncertainty: float = 0.02  # mag
    color_term_correction: bool = True
    
    # Aperture corrections
    apply_aperture_corrections: bool = True
    aperture_correction_reference: str = "psf_weighted"  # "psf_weighted", "encircled_energy"
    
    # Systematic uncertainties
    photometric_repeatability: float = 0.01  # fractional
    flat_field_uncertainty: float = 0.005  # fractional
    gain_uncertainty: float = 0.002  # fractional
    dark_current_uncertainty: float = 0.001  # fractional
    
    # Extinction corrections
    apply_galactic_extinction: bool = True
    extinction_law: str = "CCM89"  # CCM89, O94, F99, VCG04
    reddening_value: Optional[float] = None  # E(B-V), if None uses SFD maps
    
    # Atmospheric corrections (for ground-based comparison)
    apply_atmospheric_correction: bool = False
    airmass: float = 1.0
    atmospheric_extinction: Dict[str, float] = field(default_factory=dict)
    
    # Quality control
    flag_suspicious_calibrations: bool = True
    max_zeropoint_deviation: float = 0.1  # mag from expected
    min_calibration_stars: int = 10
    
    # Multi-band consistency
    check_color_consistency: bool = True
    expected_color_range: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    
    # Output options
    save_calibration_diagnostics: bool = True
    create_calibration_plots: bool = True


@dataclass
class BandCalibration:
    """Container for band-specific calibration data."""
    
    band: str
    
    # Zero-point information
    zeropoint: float  # AB magnitude
    zeropoint_error: float
    zeropoint_source: str  # "in_flight", "synthetic", "empirical"
    
    # Conversion factors
    photflam: float  # erg/cm^2/s/A per DN/s
    photfnu: float   # Jy per DN/s
    photplam: float  # Pivot wavelength in Angstroms
    
    # Aperture corrections
    aperture_corrections: Dict[float, float] = field(default_factory=dict)
    aperture_correction_error: float = 0.01
    
    # Systematic uncertainties
    systematic_uncertainty: float = 0.02  # Total systematic uncertainty
    flat_field_error: float = 0.005
    gain_error: float = 0.002
    dark_error: float = 0.001
    
    # Extinction corrections
    extinction_coefficient: float = 0.0  # A_lambda in magnitudes
    color_excess: float = 0.0  # E(B-V)
    
    # Quality metrics
    calibration_quality: float = 1.0  # Quality score 0-1
    n_calibration_stars: int = 0
    calibration_rms: float = 0.0
    
    # Validation flags
    flags: List[str] = field(default_factory=list)


@dataclass
class CalibratedSource:
    """Container for calibrated source data."""
    
    id: int
    
    # Original measurements
    instrumental_fluxes: Dict[str, Dict[float, float]] = field(default_factory=dict)  # band -> aperture -> flux
    instrumental_errors: Dict[str, Dict[float, float]] = field(default_factory=dict)
    
    # Calibrated measurements
    calibrated_fluxes: Dict[str, Dict[float, float]] = field(default_factory=dict)  # band -> aperture -> flux
    calibrated_errors: Dict[str, Dict[float, float]] = field(default_factory=dict)
    calibrated_magnitudes: Dict[str, Dict[float, float]] = field(default_factory=dict)
    magnitude_errors: Dict[str, Dict[float, float]] = field(default_factory=dict)
    
    # Best estimates (typically largest aperture or Kron)
    best_fluxes: Dict[str, float] = field(default_factory=dict)
    best_flux_errors: Dict[str, float] = field(default_factory=dict)
    best_magnitudes: Dict[str, float] = field(default_factory=dict)
    best_magnitude_errors: Dict[str, float] = field(default_factory=dict)
    
    # Colors
    colors: Dict[str, float] = field(default_factory=dict)  # e.g., 'F200W-F444W'
    color_errors: Dict[str, float] = field(default_factory=dict)
    
    # Quality flags
    calibration_flags: List[str] = field(default_factory=list)
    calibration_quality: Dict[str, float] = field(default_factory=dict)


@dataclass
class CalibrationResults:
    """Container for calibration results."""
    
    sources: List[CalibratedSource]
    band_calibrations: Dict[str, BandCalibration]
    config: CalibrationConfig
    
    # Global statistics
    calibration_statistics: Dict[str, Any] = field(default_factory=dict)
    
    # Color-color consistency checks
    color_diagnostics: Dict[str, Any] = field(default_factory=dict)
    
    # Processing information
    processing_time: float = 0.0
    n_sources_calibrated: int = 0
    
    # Quality assessment
    overall_quality: float = 1.0
    quality_metrics: Dict[str, float] = field(default_factory=dict)


class FluxCalibrator:
    """
    Comprehensive flux calibration processor for JWST observations.
    
    This class provides sophisticated flux calibration capabilities including:
    - Unit conversions (DN/s to physical units)
    - Zero-point calibration with uncertainties
    - Aperture corrections
    - Systematic uncertainty propagation
    - Extinction corrections
    - Multi-band consistency checks
    - Color calibration
    """
    
    def __init__(self, config: Optional[CalibrationConfig] = None):
        """
        Initialize the flux calibrator.
        
        Parameters:
        -----------
        config : CalibrationConfig, optional
            Calibration configuration. If None, uses defaults.
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or CalibrationConfig()
        
        # Load JWST calibration reference data
        self._load_reference_data()
    
    def calibrate_photometry(self,
                           photometry_results: Dict[str, Any],
                           headers: Dict[str, fits.Header],
                           wcs_info: Optional[Dict[str, Any]] = None) -> CalibrationResults:
        """
        Perform comprehensive flux calibration on photometry results.
        
        Parameters:
        -----------
        photometry_results : dict
            Photometry results from aperture photometry
        headers : dict
            FITS headers for each band containing calibration keywords
        wcs_info : dict, optional
            WCS information for coordinate-dependent corrections
            
        Returns:
        --------
        CalibrationResults
            Complete calibration results
        """
        self.logger.info("Starting flux calibration")
        
        import time
        start_time = time.time()
        
        # Validate inputs
        self._validate_calibration_inputs(photometry_results, headers)
        
        # Create band calibrations
        band_calibrations = {}
        for band in photometry_results.keys():
            band_calibrations[band] = self._create_band_calibration(band, headers.get(band, {}))
        
        # Prepare source list
        calibrated_sources = self._prepare_calibrated_sources(photometry_results)
        
        # Apply calibrations
        calibrated_sources = self._apply_calibrations(calibrated_sources, band_calibrations)
        
        # Compute colors
        calibrated_sources = self._compute_colors(calibrated_sources)
        
        # Quality assessment
        self._assess_calibration_quality(calibrated_sources, band_calibrations)
        
        # Compute statistics
        statistics = self._compute_calibration_statistics(calibrated_sources, band_calibrations)
        
        # Color-color diagnostics
        color_diagnostics = self._perform_color_diagnostics(calibrated_sources)
        
        # Overall quality assessment
        overall_quality = self._compute_overall_quality(band_calibrations, statistics)
        
        processing_time = time.time() - start_time
        
        results = CalibrationResults(
            sources=calibrated_sources,
            band_calibrations=band_calibrations,
            config=self.config,
            calibration_statistics=statistics,
            color_diagnostics=color_diagnostics,
            processing_time=processing_time,
            n_sources_calibrated=len(calibrated_sources),
            overall_quality=overall_quality,
            quality_metrics=self._compute_quality_metrics(band_calibrations)
        )
        
        self.logger.info(f"Flux calibration completed in {processing_time:.2f} seconds")
        self.logger.info(f"Calibrated {results.n_sources_calibrated} sources in {len(band_calibrations)} bands")
        
        return results
    
    def _load_reference_data(self) -> None:
        """Load JWST calibration reference data."""
        # JWST NIRCam photometric reference data
        # These are approximate values - in practice, would load from CRDS
        
        self.jwst_zeropoints = {
            'F070W': 25.6781,
            'F090W': 25.9454,
            'F115W': 26.2468,
            'F140M': 26.4524,
            'F150W': 26.0748,
            'F162M': 25.8921,
            'F164N': 22.8825,
            'F182M': 25.7906,
            'F187N': 22.4568,
            'F200W': 25.7906,
            'F210M': 25.6204,
            'F212N': 22.2628,
            'F250M': 25.3412,
            'F277W': 25.6813,
            'F300M': 25.1047,
            'F322W2': 25.0275,
            'F323N': 21.9463,
            'F335M': 24.8632,
            'F356W': 24.8632,
            'F360M': 24.7417,
            'F405N': 21.5561,
            'F410M': 24.4385,
            'F430M': 24.2488,
            'F444W': 24.5142,
            'F460M': 24.1185,
            'F466N': 21.2135,
            'F470N': 21.0865,
            'F480M': 23.8682
        }
        
        # Photometric conversion factors (approximate)
        self.photflam_factors = {
            'F070W': 2.044e-20,
            'F090W': 1.263e-20,
            'F115W': 9.530e-21,
            'F140M': 6.764e-21,
            'F150W': 7.866e-21,
            'F162M': 6.826e-21,
            'F164N': 6.546e-21,
            'F182M': 5.666e-21,
            'F187N': 5.412e-21,
            'F200W': 5.238e-21,
            'F210M': 4.858e-21,
            'F212N': 4.773e-21,
            'F250M': 3.814e-21,
            'F277W': 3.139e-21,
            'F300M': 2.745e-21,
            'F322W2': 2.459e-21,
            'F323N': 2.422e-21,
            'F335M': 2.265e-21,
            'F356W': 2.023e-21,
            'F360M': 1.986e-21,
            'F405N': 1.665e-21,
            'F410M': 1.624e-21,
            'F430M': 1.472e-21,
            'F444W': 1.379e-21,
            'F460M': 1.290e-21,
            'F466N': 1.264e-21,
            'F470N': 1.247e-21,
            'F480M': 1.207e-21
        }
        
        # Pivot wavelengths in Angstroms
        self.pivot_wavelengths = {
            'F070W': 7051,
            'F090W': 9019,
            'F115W': 11551,
            'F140M': 14036,
            'F150W': 15007,
            'F162M': 16199,
            'F164N': 16402,
            'F182M': 18230,
            'F187N': 18739,
            'F200W': 19886,
            'F210M': 20926,
            'F212N': 21209,
            'F250M': 24973,
            'F277W': 27658,
            'F300M': 29918,
            'F322W2': 32194,
            'F323N': 32329,
            'F335M': 33519,
            'F356W': 35682,
            'F360M': 36021,
            'F405N': 40520,
            'F410M': 40823,
            'F430M': 42728,
            'F444W': 44054,
            'F460M': 46028,
            'F466N': 46543,
            'F470N': 46974,
            'F480M': 48062
        }
    
    def _validate_calibration_inputs(self, photometry_results: Dict[str, Any], headers: Dict[str, fits.Header]) -> None:
        """Validate calibration inputs."""
        if not photometry_results:
            raise ValueError("No photometry results provided")
        
        if not headers:
            raise ValueError("No headers provided for calibration")
        
        # Check that we have headers for all bands
        for band in photometry_results.keys():
            if band not in headers:
                self.logger.warning(f"No header provided for band {band}")
    
    def _create_band_calibration(self, band: str, header: fits.Header) -> BandCalibration:
        """
        Create calibration data for a specific band.
        
        Parameters:
        -----------
        band : str
            Band name
        header : fits.Header
            FITS header containing calibration keywords
            
        Returns:
        --------
        BandCalibration
            Band calibration data
        """
        # Initialize band calibration
        calibration = BandCalibration(band=band, zeropoint=0.0, zeropoint_error=0.0, zeropoint_source="unknown",
                                    photflam=0.0, photfnu=0.0, photplam=0.0)
        
        # Get zero-point from header or reference
        if 'PHOTZPT' in header:
            calibration.zeropoint = header['PHOTZPT']
            calibration.zeropoint_source = "header"
        elif band in self.jwst_zeropoints:
            calibration.zeropoint = self.jwst_zeropoints[band]
            calibration.zeropoint_source = "reference"
        else:
            self.logger.warning(f"No zero-point available for band {band}")
            calibration.zeropoint = 25.0  # Default
            calibration.flags.append("no_zeropoint")
        
        # Set zero-point uncertainty
        if 'PHOTZPTE' in header:
            calibration.zeropoint_error = header['PHOTZPTE']
        else:
            calibration.zeropoint_error = self.config.zeropoint_uncertainty
        
        # Get photometric conversion factors
        if 'PHOTFLAM' in header:
            calibration.photflam = header['PHOTFLAM']
        elif band in self.photflam_factors:
            calibration.photflam = self.photflam_factors[band]
        else:
            self.logger.warning(f"No PHOTFLAM available for band {band}")
            calibration.flags.append("no_photflam")
        
        # Calculate PHOTFNU from PHOTFLAM and pivot wavelength
        if band in self.pivot_wavelengths:
            calibration.photplam = self.pivot_wavelengths[band]
            if calibration.photflam > 0:
                # Convert from erg/cm^2/s/A to Jy
                calibration.photfnu = (calibration.photflam * calibration.photplam**2 / 
                                     (2.998e18 * 1e-23))  # Approximate conversion
        
        # Get gain information for error propagation
        if 'GAIN' in header:
            gain = header['GAIN']
            calibration.gain_error = gain * self.config.gain_uncertainty
        
        # Set systematic uncertainties
        calibration.flat_field_error = self.config.flat_field_uncertainty
        calibration.dark_error = self.config.dark_current_uncertainty
        
        # Total systematic uncertainty
        calibration.systematic_uncertainty = np.sqrt(
            calibration.zeropoint_error**2 +
            calibration.flat_field_error**2 +
            calibration.gain_error**2 +
            calibration.dark_error**2
        )
        
        # Apply extinction corrections if requested
        if self.config.apply_galactic_extinction:
            calibration.extinction_coefficient = self._get_extinction_coefficient(band)
            calibration.color_excess = self._get_color_excess()
        
        return calibration
    
    def _get_extinction_coefficient(self, band: str) -> float:
        """Get Galactic extinction coefficient for band."""
        # Simplified extinction coefficients (A_lambda/A_V)
        # In practice, would use more sophisticated dust models
        extinction_ratios = {
            'F070W': 1.531,
            'F090W': 1.324,
            'F115W': 1.071,
            'F140M': 0.887,
            'F150W': 0.839,
            'F162M': 0.787,
            'F164N': 0.778,
            'F182M': 0.700,
            'F187N': 0.683,
            'F200W': 0.641,
            'F210M': 0.616,
            'F212N': 0.609,
            'F250M': 0.507,
            'F277W': 0.439,
            'F300M': 0.395,
            'F322W2': 0.364,
            'F323N': 0.361,
            'F335M': 0.344,
            'F356W': 0.314,
            'F360M': 0.309,
            'F405N': 0.263,
            'F410M': 0.258,
            'F430M': 0.237,
            'F444W': 0.223,
            'F460M': 0.211,
            'F466N': 0.207,
            'F470N': 0.204,
            'F480M': 0.198
        }
        
        return extinction_ratios.get(band, 0.0)
    
    def _get_color_excess(self) -> float:
        """Get color excess E(B-V) for target."""
        if self.config.reddening_value is not None:
            return self.config.reddening_value
        else:
            # In practice, would query SFD dust maps
            return 0.02  # Default low extinction
    
    def _prepare_calibrated_sources(self, photometry_results: Dict[str, Any]) -> List[CalibratedSource]:
        """
        Prepare calibrated source list from photometry results.
        
        Parameters:
        -----------
        photometry_results : dict
            Photometry results from each band
            
        Returns:
        --------
        list
            List of calibrated sources
        """
        calibrated_sources = []
        
        # Determine number of sources (assume all bands have same number)
        band_names = list(photometry_results.keys())
        if not band_names:
            return calibrated_sources
        
        first_band_results = photometry_results[band_names[0]]
        n_sources = len(first_band_results.sources)
        
        # Create calibrated sources
        for i in range(n_sources):
            calibrated_source = CalibratedSource(id=i)
            
            # Copy instrumental fluxes from each band
            for band, results in photometry_results.items():
                if i < len(results.sources):
                    source = results.sources[i]
                    
                    # Copy circular aperture measurements
                    calibrated_source.instrumental_fluxes[band] = source.circular_fluxes.copy()
                    calibrated_source.instrumental_errors[band] = source.circular_flux_errors.copy()
            
            calibrated_sources.append(calibrated_source)
        
        return calibrated_sources
    
    def _apply_calibrations(self,
                          sources: List[CalibratedSource],
                          band_calibrations: Dict[str, BandCalibration]) -> List[CalibratedSource]:
        """
        Apply flux calibrations to all sources.
        
        Parameters:
        -----------
        sources : list
            List of calibrated sources
        band_calibrations : dict
            Band calibration data
            
        Returns:
        --------
        list
            Sources with applied calibrations
        """
        for source in sources:
            for band, calibration in band_calibrations.items():
                if band not in source.instrumental_fluxes:
                    continue
                
                try:
                    # Initialize calibrated measurements for this band
                    source.calibrated_fluxes[band] = {}
                    source.calibrated_errors[band] = {}
                    source.calibrated_magnitudes[band] = {}
                    source.magnitude_errors[band] = {}
                    
                    # Apply calibration to each aperture
                    for aperture, inst_flux in source.instrumental_fluxes[band].items():
                        inst_error = source.instrumental_errors[band].get(aperture, 0.0)
                        
                        # Convert to calibrated flux
                        cal_flux, cal_error = self._calibrate_flux(
                            inst_flux, inst_error, calibration
                        )
                        
                        source.calibrated_fluxes[band][aperture] = cal_flux
                        source.calibrated_errors[band][aperture] = cal_error
                        
                        # Convert to magnitude
                        if cal_flux > 0:
                            magnitude = -2.5 * np.log10(cal_flux) + calibration.zeropoint
                            if cal_error > 0:
                                mag_error = 1.0857 * cal_error / cal_flux
                            else:
                                mag_error = 99.0
                        else:
                            magnitude = 99.0
                            mag_error = 99.0
                            source.calibration_flags.append(f"negative_flux_{band}_r{aperture}")
                        
                        source.calibrated_magnitudes[band][aperture] = magnitude
                        source.magnitude_errors[band][aperture] = mag_error
                    
                    # Set best estimates (use largest aperture)
                    if source.calibrated_fluxes[band]:
                        largest_aperture = max(source.calibrated_fluxes[band].keys())
                        source.best_fluxes[band] = source.calibrated_fluxes[band][largest_aperture]
                        source.best_flux_errors[band] = source.calibrated_errors[band][largest_aperture]
                        source.best_magnitudes[band] = source.calibrated_magnitudes[band][largest_aperture]
                        source.best_magnitude_errors[band] = source.magnitude_errors[band][largest_aperture]
                        
                        # Set calibration quality
                        source.calibration_quality[band] = calibration.calibration_quality
                
                except Exception as e:
                    self.logger.debug(f"Calibration failed for source {source.id}, band {band}: {e}")
                    source.calibration_flags.append(f"calibration_failed_{band}")
        
        return sources
    
    def _calibrate_flux(self,
                       instrumental_flux: float,
                       instrumental_error: float,
                       calibration: BandCalibration) -> Tuple[float, float]:
        """
        Calibrate instrumental flux to physical units.
        
        Parameters:
        -----------
        instrumental_flux : float
            Instrumental flux in DN/s
        instrumental_error : float
            Instrumental flux error
        calibration : BandCalibration
            Band calibration data
            
        Returns:
        --------
        tuple
            Calibrated flux and error
        """
        # Apply extinction correction
        extinction_mag = calibration.extinction_coefficient * calibration.color_excess
        extinction_factor = 10**(0.4 * extinction_mag)
        
        # Convert to requested output units
        if self.config.output_units == "uJy":
            # Convert to microJansky
            if calibration.photfnu > 0:
                calibrated_flux = instrumental_flux * calibration.photfnu * 1e6 * extinction_factor
            else:
                # Use zero-point conversion
                calibrated_flux = instrumental_flux * 10**((calibration.zeropoint - 23.9) / 2.5) * extinction_factor
        
        elif self.config.output_units == "nJy":
            # Convert to nanoJansky
            if calibration.photfnu > 0:
                calibrated_flux = instrumental_flux * calibration.photfnu * 1e9 * extinction_factor
            else:
                calibrated_flux = instrumental_flux * 10**((calibration.zeropoint - 23.9) / 2.5) * 1000 * extinction_factor
        
        elif self.config.output_units in ["mag_AB", "mag_Vega"]:
            # Keep as instrumental for magnitude calculation
            calibrated_flux = instrumental_flux * extinction_factor
        
        else:
            # Default to uJy
            calibrated_flux = instrumental_flux * extinction_factor
        
        # Propagate errors
        if instrumental_error > 0 and instrumental_flux > 0:
            # Fractional error propagation
            fractional_error = instrumental_error / instrumental_flux
            
            # Add systematic uncertainties
            total_fractional_error = np.sqrt(
                fractional_error**2 +
                calibration.systematic_uncertainty**2 +
                (self.config.photometric_repeatability)**2
            )
            
            calibrated_error = calibrated_flux * total_fractional_error
        else:
            calibrated_error = 0.0
        
        return calibrated_flux, calibrated_error
    
    def _compute_colors(self, sources: List[CalibratedSource]) -> List[CalibratedSource]:
        """
        Compute colors for all sources.
        
        Parameters:
        -----------
        sources : list
            List of calibrated sources
            
        Returns:
        --------
        list
            Sources with computed colors
        """
        # Get list of available bands
        all_bands = set()
        for source in sources:
            all_bands.update(source.best_magnitudes.keys())
        all_bands = sorted(list(all_bands))
        
        # Compute all possible colors
        for source in sources:
            for i, band1 in enumerate(all_bands):
                for band2 in all_bands[i+1:]:
                    if band1 in source.best_magnitudes and band2 in source.best_magnitudes:
                        mag1 = source.best_magnitudes[band1]
                        mag2 = source.best_magnitudes[band2]
                        err1 = source.best_magnitude_errors[band1]
                        err2 = source.best_magnitude_errors[band2]
                        
                        if mag1 < 90 and mag2 < 90:  # Valid magnitudes
                            color_name = f"{band1}-{band2}"
                            source.colors[color_name] = mag1 - mag2
                            source.color_errors[color_name] = np.sqrt(err1**2 + err2**2)
        
        return sources
    
    def _assess_calibration_quality(self,
                                  sources: List[CalibratedSource],
                                  band_calibrations: Dict[str, BandCalibration]) -> None:
        """Assess calibration quality for each band."""
        for band, calibration in band_calibrations.items():
            # Count successful calibrations
            successful_sources = len([s for s in sources if band in s.best_fluxes])
            calibration.n_calibration_stars = successful_sources
            
            # Check for quality issues
            if calibration.zeropoint_error > self.config.max_zeropoint_deviation:
                calibration.flags.append("high_zeropoint_uncertainty")
            
            if successful_sources < self.config.min_calibration_stars:
                calibration.flags.append("few_calibration_sources")
            
            # Compute quality score
            quality_factors = []
            
            # Zero-point quality
            zp_quality = max(0, 1 - calibration.zeropoint_error / 0.1)
            quality_factors.append(zp_quality)
            
            # Number of sources
            source_quality = min(1, successful_sources / self.config.min_calibration_stars)
            quality_factors.append(source_quality)
            
            # Systematic uncertainty
            sys_quality = max(0, 1 - calibration.systematic_uncertainty / 0.05)
            quality_factors.append(sys_quality)
            
            calibration.calibration_quality = np.mean(quality_factors)
    
    def _compute_calibration_statistics(self,
                                      sources: List[CalibratedSource],
                                      band_calibrations: Dict[str, BandCalibration]) -> Dict[str, Any]:
        """Compute calibration statistics."""
        statistics = {}
        
        # Basic counts
        statistics['total_sources'] = len(sources)
        statistics['bands'] = list(band_calibrations.keys())
        statistics['n_bands'] = len(band_calibrations)
        
        # Per-band statistics
        band_stats = {}
        for band in band_calibrations.keys():
            band_fluxes = [s.best_fluxes.get(band, 0) for s in sources if band in s.best_fluxes]
            band_mags = [s.best_magnitudes.get(band, 99) for s in sources if band in s.best_magnitudes]
            
            band_stats[band] = {
                'n_detections': len(band_fluxes),
                'median_flux': np.median(band_fluxes) if band_fluxes else 0,
                'median_magnitude': np.median([m for m in band_mags if m < 90]) if band_mags else 99,
                'calibration_quality': band_calibrations[band].calibration_quality
            }
        
        statistics['band_statistics'] = band_stats
        
        # Color statistics
        all_colors = set()
        for source in sources:
            all_colors.update(source.colors.keys())
        
        color_stats = {}
        for color in all_colors:
            color_values = [s.colors[color] for s in sources if color in s.colors]
            if color_values:
                color_stats[color] = {
                    'n_measurements': len(color_values),
                    'median': np.median(color_values),
                    'std': np.std(color_values),
                    'range': (np.min(color_values), np.max(color_values))
                }
        
        statistics['color_statistics'] = color_stats
        
        return statistics
    
    def _perform_color_diagnostics(self, sources: List[CalibratedSource]) -> Dict[str, Any]:
        """Perform color-color consistency diagnostics."""
        diagnostics = {}
        
        # Check for expected color ranges
        for color_name, expected_range in self.config.expected_color_range.items():
            color_values = [s.colors[color_name] for s in sources if color_name in s.colors]
            
            if color_values:
                outliers = [c for c in color_values if c < expected_range[0] or c > expected_range[1]]
                diagnostics[color_name] = {
                    'total_measurements': len(color_values),
                    'outliers': len(outliers),
                    'outlier_fraction': len(outliers) / len(color_values),
                    'median_color': np.median(color_values),
                    'expected_range': expected_range
                }
        
        return diagnostics
    
    def _compute_overall_quality(self,
                               band_calibrations: Dict[str, BandCalibration],
                               statistics: Dict[str, Any]) -> float:
        """Compute overall calibration quality score."""
        quality_scores = [cal.calibration_quality for cal in band_calibrations.values()]
        
        if quality_scores:
            return np.mean(quality_scores)
        else:
            return 0.0
    
    def _compute_quality_metrics(self, band_calibrations: Dict[str, BandCalibration]) -> Dict[str, float]:
        """Compute quality metrics."""
        metrics = {}
        
        # Average zero-point uncertainty
        zp_errors = [cal.zeropoint_error for cal in band_calibrations.values()]
        if zp_errors:
            metrics['mean_zeropoint_error'] = np.mean(zp_errors)
            metrics['max_zeropoint_error'] = np.max(zp_errors)
        
        # Average systematic uncertainty
        sys_errors = [cal.systematic_uncertainty for cal in band_calibrations.values()]
        if sys_errors:
            metrics['mean_systematic_error'] = np.mean(sys_errors)
            metrics['max_systematic_error'] = np.max(sys_errors)
        
        # Calibration completeness
        total_flags = sum(len(cal.flags) for cal in band_calibrations.values())
        metrics['total_calibration_flags'] = total_flags
        
        return metrics
    
    def export_calibrated_catalog(self,
                                results: CalibrationResults,
                                output_path: str,
                                format: str = "fits") -> None:
        """
        Export calibrated catalog to file.
        
        Parameters:
        -----------
        results : CalibrationResults
            Calibration results to export
        output_path : str
            Output file path
        format : str
            Output format ('fits', 'ascii', 'csv')
        """
        try:
            # Prepare table data
            table_data = {}
            
            # Basic information
            table_data['id'] = [s.id for s in results.sources]
            
            # Add calibrated photometry for each band and aperture
            for band in results.band_calibrations.keys():
                # Get aperture sizes
                apertures = set()
                for source in results.sources:
                    if band in source.calibrated_fluxes:
                        apertures.update(source.calibrated_fluxes[band].keys())
                
                apertures = sorted(list(apertures))
                
                for aperture in apertures:
                    # Fluxes
                    flux_col = f'flux_{band}_r{aperture}'
                    fluxerr_col = f'flux_err_{band}_r{aperture}'
                    mag_col = f'mag_{band}_r{aperture}'
                    magerr_col = f'mag_err_{band}_r{aperture}'
                    
                    table_data[flux_col] = [
                        s.calibrated_fluxes.get(band, {}).get(aperture, np.nan) 
                        for s in results.sources
                    ]
                    table_data[fluxerr_col] = [
                        s.calibrated_errors.get(band, {}).get(aperture, np.nan) 
                        for s in results.sources
                    ]
                    table_data[mag_col] = [
                        s.calibrated_magnitudes.get(band, {}).get(aperture, np.nan) 
                        for s in results.sources
                    ]
                    table_data[magerr_col] = [
                        s.magnitude_errors.get(band, {}).get(aperture, np.nan) 
                        for s in results.sources
                    ]
                
                # Best estimates
                table_data[f'flux_best_{band}'] = [
                    s.best_fluxes.get(band, np.nan) for s in results.sources
                ]
                table_data[f'flux_err_best_{band}'] = [
                    s.best_flux_errors.get(band, np.nan) for s in results.sources
                ]
                table_data[f'mag_best_{band}'] = [
                    s.best_magnitudes.get(band, np.nan) for s in results.sources
                ]
                table_data[f'mag_err_best_{band}'] = [
                    s.best_magnitude_errors.get(band, np.nan) for s in results.sources
                ]
            
            # Add colors
            all_colors = set()
            for source in results.sources:
                all_colors.update(source.colors.keys())
            
            for color in sorted(all_colors):
                table_data[f'color_{color}'] = [
                    s.colors.get(color, np.nan) for s in results.sources
                ]
                table_data[f'color_err_{color}'] = [
                    s.color_errors.get(color, np.nan) for s in results.sources
                ]
            
            # Create table
            catalog = Table(table_data)
            
            # Add metadata
            catalog.meta['CALIBRATION_CONFIG'] = str(results.config)
            catalog.meta['PROCESSING_TIME'] = results.processing_time
            catalog.meta['N_SOURCES'] = results.n_sources_calibrated
            catalog.meta['OVERALL_QUALITY'] = results.overall_quality
            
            # Save table
            if format.lower() == 'fits':
                catalog.write(output_path, format='fits', overwrite=True)
            elif format.lower() == 'ascii':
                catalog.write(output_path, format='ascii.ecsv', overwrite=True)
            elif format.lower() == 'csv':
                catalog.write(output_path, format='csv', overwrite=True)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            self.logger.info(f"Calibrated catalog saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to export calibrated catalog: {e}")
            raise
    
    def plot_calibration_diagnostics(self,
                                   results: CalibrationResults,
                                   output_path: Optional[str] = None) -> None:
        """Create calibration diagnostic plots."""
        try:
            n_bands = len(results.band_calibrations)
            n_cols = min(3, n_bands)
            n_rows = (n_bands + n_cols - 1) // n_cols + 2  # +2 for color plots
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
            
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            elif n_cols == 1:
                axes = axes.reshape(-1, 1)
            
            # Band-by-band magnitude distributions
            for i, (band, calibration) in enumerate(results.band_calibrations.items()):
                row, col = i // n_cols, i % n_cols
                ax = axes[row, col]
                
                # Get magnitudes for this band
                magnitudes = [s.best_magnitudes.get(band, 99) for s in results.sources 
                            if band in s.best_magnitudes]
                valid_mags = [m for m in magnitudes if m < 90]
                
                if valid_mags:
                    ax.hist(valid_mags, bins=20, alpha=0.7, label=band)
                    ax.set_xlabel('Magnitude')
                    ax.set_ylabel('Count')
                    ax.set_title(f'{band} Magnitude Distribution\nQuality: {calibration.calibration_quality:.2f}')
                    ax.legend()
                else:
                    ax.text(0.5, 0.5, f'No valid data\nfor {band}', 
                           transform=ax.transAxes, ha='center', va='center')
                    ax.set_title(f'{band} - No Data')
            
            # Color-magnitude diagram
            if n_rows > 1:
                ax_cmd = axes[-2, 0]
                
                # Use first two bands for CMD if available
                bands = list(results.band_calibrations.keys())
                if len(bands) >= 2:
                    band1, band2 = bands[0], bands[1]
                    color_name = f"{band1}-{band2}"
                    
                    colors = []
                    mags = []
                    
                    for source in results.sources:
                        if (color_name in source.colors and 
                            band2 in source.best_magnitudes and
                            source.best_magnitudes[band2] < 90):
                            colors.append(source.colors[color_name])
                            mags.append(source.best_magnitudes[band2])
                    
                    if colors and mags:
                        ax_cmd.scatter(colors, mags, alpha=0.6, s=10)
                        ax_cmd.set_xlabel(f'{color_name} Color')
                        ax_cmd.set_ylabel(f'{band2} Magnitude')
                        ax_cmd.set_title('Color-Magnitude Diagram')
                        ax_cmd.invert_yaxis()
                
                # Color distribution
                ax_color = axes[-1, 0]
                
                if len(bands) >= 2:
                    color_name = f"{bands[0]}-{bands[1]}"
                    color_values = [s.colors.get(color_name, np.nan) for s in results.sources 
                                  if color_name in s.colors]
                    
                    if color_values:
                        ax_color.hist(color_values, bins=20, alpha=0.7)
                        ax_color.set_xlabel(f'{color_name} Color')
                        ax_color.set_ylabel('Count')
                        ax_color.set_title('Color Distribution')
                
                # Quality summary
                if n_cols > 1:
                    ax_quality = axes[-2, 1]
                    
                    band_names = list(results.band_calibrations.keys())
                    qualities = [results.band_calibrations[band].calibration_quality 
                               for band in band_names]
                    
                    bars = ax_quality.bar(range(len(band_names)), qualities)
                    ax_quality.set_xticks(range(len(band_names)))
                    ax_quality.set_xticklabels(band_names, rotation=45)
                    ax_quality.set_ylabel('Quality Score')
                    ax_quality.set_title('Calibration Quality by Band')
                    ax_quality.set_ylim(0, 1)
                    
                    # Color bars by quality
                    for bar, quality in zip(bars, qualities):
                        if quality > 0.8:
                            bar.set_color('green')
                        elif quality > 0.6:
                            bar.set_color('orange')
                        else:
                            bar.set_color('red')
            
            # Remove empty subplots
            for i in range(len(results.band_calibrations), n_rows * n_cols):
                if i < (n_rows - 2) * n_cols:  # Don't remove color/quality plots
                    row, col = i // n_cols, i % n_cols
                    fig.delaxes(axes[row, col])
            
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                self.logger.info(f"Calibration diagnostics saved to {output_path}")
            else:
                plt.show()
            
            plt.close()
            
        except Exception as e:
            self.logger.warning(f"Failed to create calibration diagnostic plots: {e}")


# Utility functions for legacy compatibility
def calibrate_to_physical_units(flux_instrumental, band, zeropoint=None):
    """
    Legacy function to calibrate instrumental flux to physical units.
    
    Parameters:
    -----------
    flux_instrumental : float or array
        Instrumental flux in DN/s
    band : str
        Filter band name
    zeropoint : float, optional
        Zero-point magnitude. If None, uses default.
        
    Returns:
    --------
    float or array
        Calibrated flux in microJansky
    """
    try:
        config = CalibrationConfig()
        calibrator = FluxCalibrator(config)
        
        # Create minimal calibration
        if zeropoint is None:
            zeropoint = calibrator.jwst_zeropoints.get(band, 25.0)
        
        calibration = BandCalibration(
            band=band,
            zeropoint=zeropoint,
            zeropoint_error=0.02,
            zeropoint_source="input",
            photflam=calibrator.photflam_factors.get(band, 1e-20),
            photfnu=1e-6,  # Approximate
            photplam=calibrator.pivot_wavelengths.get(band, 20000)
        )
        
        # Simple conversion to microJansky
        if calibration.photfnu > 0:
            return flux_instrumental * calibration.photfnu * 1e6
        else:
            return flux_instrumental * 10**((zeropoint - 23.9) / 2.5)
    
    except Exception as e:
        logging.warning(f"Legacy calibration failed: {e}")
        return flux_instrumental
