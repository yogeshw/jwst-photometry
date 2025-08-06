#!/usr/bin/env python3
"""
Multi-Epoch Handling Module for JWST NIRCam

This module provides multi-epoch analysis capabilities including:
- Multi-epoch source matching
- Variability detection and characterization
- Proper motion measurement
- Long-term systematic monitoring
- Time-series photometry

Author: JWST Photometry Pipeline
Date: August 2025
"""

import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import warnings
import time
from datetime import datetime

try:
    from astropy.io import fits
    from astropy.table import Table, vstack
    from astropy import units as u
    from astropy.time import Time
    from astropy.coordinates import SkyCoord
    from astropy.stats import sigma_clipped_stats
    from astropy.timeseries import LombScargle
    ASTROPY_AVAILABLE = True
except ImportError:
    ASTROPY_AVAILABLE = False
    warnings.warn("Astropy not available - multi-epoch capabilities will be limited")

try:
    from scipy import stats, optimize, interpolate
    from scipy.spatial import cKDTree
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("SciPy not available - some multi-epoch features will be limited")

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("matplotlib not available - plotting capabilities limited")


@dataclass
class MultiEpochConfig:
    """Configuration for multi-epoch analysis."""
    
    # Source matching
    match_radius: float = 1.0  # arcseconds
    match_method: str = 'positional'  # 'positional', 'probabilistic', 'wcs_based'
    proper_motion_model: bool = True
    
    # Variability detection
    variability_threshold: float = 3.0  # sigma
    min_epochs: int = 3
    variability_tests: List[str] = field(default_factory=lambda: ['chi2', 'iqr', 'eta'])
    
    # Proper motion measurement
    pm_min_epochs: int = 3
    pm_min_timespan: float = 30.0  # days
    pm_sigma_threshold: float = 5.0
    pm_fit_method: str = 'linear'  # 'linear', 'weighted', 'robust'
    
    # Time-series analysis
    enable_time_series: bool = True
    period_search_range: Tuple[float, float] = (0.1, 100.0)  # days
    ls_false_alarm_probability: float = 0.01
    
    # Quality control
    outlier_rejection: bool = True
    outlier_sigma: float = 3.0
    check_systematic_trends: bool = True
    
    # Output options
    create_lightcurves: bool = True
    create_proper_motion_plots: bool = True
    save_time_series: bool = True
    plot_format: str = 'png'

@dataclass
class EpochData:
    """Data for a single observational epoch."""
    
    epoch_id: str
    observation_time: float  # MJD
    filter_name: str
    
    # Source measurements
    sources: List[Dict[str, Any]]
    
    # Image metadata
    exposure_time: float = 0.0
    airmass: float = 0.0
    seeing: float = 0.0
    sky_background: float = 0.0
    
    # Calibration information
    zero_point: float = 0.0
    zero_point_err: float = 0.0
    
    # Quality flags
    quality_flag: int = 0
    weather_conditions: str = 'unknown'
    
    # WCS information
    wcs: Optional[Any] = None
    astrometric_rms: float = 0.0

@dataclass
class MultiEpochSource:
    """Source with multi-epoch measurements."""
    
    source_id: str
    
    # Mean position
    mean_ra: float
    mean_dec: float
    mean_ra_err: float = 0.0
    mean_dec_err: float = 0.0
    
    # Epoch measurements
    epochs: List[float] = field(default_factory=list)  # MJD
    ra_measurements: List[float] = field(default_factory=list)
    dec_measurements: List[float] = field(default_factory=list)
    ra_errors: List[float] = field(default_factory=list)
    dec_errors: List[float] = field(default_factory=list)
    
    # Photometric measurements by filter
    photometry: Dict[str, Dict[str, List[float]]] = field(default_factory=dict)
    
    # Proper motion
    pmra: float = 0.0  # mas/year
    pmdec: float = 0.0  # mas/year
    pmra_err: float = 0.0
    pmdec_err: float = 0.0
    pm_significance: float = 0.0
    
    # Variability properties
    is_variable: bool = False
    variability_type: str = 'unknown'
    variability_amplitude: Dict[str, float] = field(default_factory=dict)
    variability_period: Dict[str, float] = field(default_factory=dict)
    variability_significance: Dict[str, float] = field(default_factory=dict)
    
    # Quality metrics
    n_epochs: int = 0
    time_span: float = 0.0  # days
    chi2_reduced: Dict[str, float] = field(default_factory=dict)
    
    # Flags
    flags: int = 0

@dataclass
class VariabilityResult:
    """Results from variability analysis."""
    
    source_id: str
    filter_name: str
    
    # Variability tests
    chi2_statistic: float = 0.0
    chi2_pvalue: float = 1.0
    eta_statistic: float = 0.0
    iqr_ratio: float = 0.0
    
    # Time-series properties
    mean_magnitude: float = 0.0
    magnitude_rms: float = 0.0
    amplitude: float = 0.0
    
    # Periodicity
    best_period: float = 0.0
    period_power: float = 0.0
    period_fap: float = 1.0
    
    # Classification
    variability_class: str = 'unknown'
    confidence: float = 0.0
    
    # Light curve properties
    n_points: int = 0
    time_span: float = 0.0

@dataclass
class MultiEpochResults:
    """Complete multi-epoch analysis results."""
    
    # Matched sources
    sources: List[MultiEpochSource]
    
    # Epoch information
    epochs: List[EpochData]
    n_epochs: int = 0
    total_time_span: float = 0.0
    
    # Matching statistics
    matching_stats: Dict[str, Any] = field(default_factory=dict)
    
    # Proper motion results
    proper_motion_catalog: Optional[List[Dict[str, Any]]] = None
    pm_detection_rate: float = 0.0
    
    # Variability results
    variability_catalog: Optional[List[VariabilityResult]] = None
    variable_fraction: Dict[str, float] = field(default_factory=dict)
    
    # Systematic trends
    systematic_analysis: Dict[str, Any] = field(default_factory=dict)
    
    # Quality assessment
    overall_quality: str = 'Unknown'
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Generated products
    output_files: Dict[str, str] = field(default_factory=dict)
    
    # Processing metadata
    processing_time: float = 0.0


class MultiEpochProcessor:
    """
    Multi-epoch analysis processor for JWST NIRCam.
    
    This class provides comprehensive multi-epoch analysis including source
    matching, variability detection, proper motion measurement, and systematic
    monitoring.
    """
    
    def __init__(self, config: Optional[MultiEpochConfig] = None):
        """
        Initialize the multi-epoch processor.
        
        Parameters:
        -----------
        config : MultiEpochConfig, optional
            Multi-epoch configuration. If None, uses defaults.
        """
        self.config = config or MultiEpochConfig()
        self.logger = logging.getLogger(__name__)
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate the configuration parameters."""
        if self.config.match_radius <= 0:
            raise ValueError("Match radius must be positive")
        
        if self.config.min_epochs < 2:
            raise ValueError("Need at least 2 epochs for multi-epoch analysis")
        
        if self.config.pm_min_epochs < 3:
            raise ValueError("Need at least 3 epochs for proper motion measurement")
    
    def process_multi_epoch(self,
                          epoch_data_list: List[EpochData],
                          output_dir: Optional[Path] = None) -> MultiEpochResults:
        """
        Process multi-epoch data for variability and proper motion.
        
        Parameters:
        -----------
        epoch_data_list : list
            List of EpochData objects for each observation
        output_dir : Path, optional
            Directory for output files
            
        Returns:
        --------
        MultiEpochResults
            Complete multi-epoch analysis results
        """
        start_time = time.time()
        self.logger.info(f"Processing {len(epoch_data_list)} epochs")
        
        # Sort epochs by observation time
        sorted_epochs = sorted(epoch_data_list, key=lambda x: x.observation_time)
        
        # Match sources across epochs
        self.logger.info("Matching sources across epochs")
        matched_sources = self._match_sources_across_epochs(sorted_epochs)
        
        # Compute matching statistics
        matching_stats = self._compute_matching_statistics(matched_sources, sorted_epochs)
        
        # Measure proper motions
        self.logger.info("Measuring proper motions")
        pm_catalog = self._measure_proper_motions(matched_sources)
        
        # Detect variability
        self.logger.info("Detecting variability")
        variability_catalog = self._detect_variability(matched_sources)
        
        # Analyze systematic trends
        systematic_analysis = {}
        if self.config.check_systematic_trends:
            self.logger.info("Analyzing systematic trends")
            systematic_analysis = self._analyze_systematic_trends(matched_sources, sorted_epochs)
        
        # Assess overall quality
        quality, quality_metrics = self._assess_multi_epoch_quality(
            matched_sources, sorted_epochs, matching_stats
        )
        
        # Create output products
        output_files = {}
        if output_dir:
            self.logger.info("Creating output products")
            output_files = self._create_output_products(
                matched_sources, pm_catalog, variability_catalog, sorted_epochs, output_dir
            )
        
        # Create results
        processing_time = time.time() - start_time
        total_time_span = (sorted_epochs[-1].observation_time - sorted_epochs[0].observation_time)
        
        # Compute detection rates
        pm_detection_rate = len([s for s in matched_sources if s.pm_significance > self.config.pm_sigma_threshold]) / max(len(matched_sources), 1)
        
        variable_fraction = {}
        for filter_name in set(epoch.filter_name for epoch in sorted_epochs):
            n_variable = len([r for r in variability_catalog if r.filter_name == filter_name and r.chi2_pvalue < 0.01])
            n_total = len([s for s in matched_sources if filter_name in s.photometry])
            variable_fraction[filter_name] = n_variable / max(n_total, 1)
        
        results = MultiEpochResults(
            sources=matched_sources,
            epochs=sorted_epochs,
            n_epochs=len(sorted_epochs),
            total_time_span=total_time_span,
            matching_stats=matching_stats,
            proper_motion_catalog=pm_catalog,
            pm_detection_rate=pm_detection_rate,
            variability_catalog=variability_catalog,
            variable_fraction=variable_fraction,
            systematic_analysis=systematic_analysis,
            overall_quality=quality,
            quality_metrics=quality_metrics,
            output_files=output_files,
            processing_time=processing_time
        )
        
        self.logger.info(f"Multi-epoch processing completed in {processing_time:.1f}s")
        self.logger.info(f"Matched {len(matched_sources)} sources across {len(sorted_epochs)} epochs")
        self.logger.info(f"Proper motion detection rate: {pm_detection_rate:.1%}")
        
        return results
    
    def _match_sources_across_epochs(self, epochs: List[EpochData]) -> List[MultiEpochSource]:
        """Match sources across all epochs."""
        if not ASTROPY_AVAILABLE:
            raise RuntimeError("Astropy required for source matching")
        
        matched_sources = []
        
        # Use first epoch as reference
        reference_epoch = epochs[0]
        reference_sources = reference_epoch.sources
        
        # Initialize matched sources from reference epoch
        for i, ref_source in enumerate(reference_sources):
            source_id = f"source_{i:06d}"
            
            matched_source = MultiEpochSource(
                source_id=source_id,
                mean_ra=ref_source.get('ra', 0.0),
                mean_dec=ref_source.get('dec', 0.0),
                n_epochs=1
            )
            
            # Add first epoch data
            matched_source.epochs.append(reference_epoch.observation_time)
            matched_source.ra_measurements.append(ref_source.get('ra', 0.0))
            matched_source.dec_measurements.append(ref_source.get('dec', 0.0))
            matched_source.ra_errors.append(ref_source.get('ra_err', 0.1))
            matched_source.dec_errors.append(ref_source.get('dec_err', 0.1))
            
            # Initialize photometry dictionary
            filter_name = reference_epoch.filter_name
            if filter_name not in matched_source.photometry:
                matched_source.photometry[filter_name] = {
                    'magnitudes': [],
                    'magnitude_errors': [],
                    'epochs': []
                }
            
            matched_source.photometry[filter_name]['magnitudes'].append(
                ref_source.get('magnitude', 99.0)
            )
            matched_source.photometry[filter_name]['magnitude_errors'].append(
                ref_source.get('magnitude_error', 1.0)
            )
            matched_source.photometry[filter_name]['epochs'].append(
                reference_epoch.observation_time
            )
            
            matched_sources.append(matched_source)
        
        # Match sources in subsequent epochs
        for epoch in epochs[1:]:
            self._match_epoch_to_catalog(epoch, matched_sources)
        
        # Update statistics and compute mean positions
        for source in matched_sources:
            source.time_span = max(source.epochs) - min(source.epochs)
            
            # Compute weighted mean position
            if len(source.ra_measurements) > 1:
                weights = 1.0 / np.array(source.ra_errors)**2
                source.mean_ra = np.average(source.ra_measurements, weights=weights)
                source.mean_ra_err = 1.0 / np.sqrt(np.sum(weights))
                
                weights = 1.0 / np.array(source.dec_errors)**2
                source.mean_dec = np.average(source.dec_measurements, weights=weights)
                source.mean_dec_err = 1.0 / np.sqrt(np.sum(weights))
        
        return matched_sources
    
    def _match_epoch_to_catalog(self, epoch: EpochData, matched_sources: List[MultiEpochSource]) -> None:
        """Match sources in an epoch to the existing catalog."""
        if not ASTROPY_AVAILABLE:
            return
        
        # Create coordinate objects
        catalog_coords = SkyCoord(
            ra=[s.mean_ra for s in matched_sources] * u.degree,
            dec=[s.mean_dec for s in matched_sources] * u.degree
        )
        
        epoch_coords = SkyCoord(
            ra=[s.get('ra', 0.0) for s in epoch.sources] * u.degree,
            dec=[s.get('dec', 0.0) for s in epoch.sources] * u.degree
        )
        
        if len(epoch_coords) == 0:
            return
        
        # Find matches
        idx, d2d, d3d = epoch_coords.match_to_catalog_sky(catalog_coords)
        
        # Process matches
        for i, (source_data, catalog_idx, separation) in enumerate(zip(epoch.sources, idx, d2d)):
            if separation < self.config.match_radius * u.arcsec:
                # Found a match
                matched_source = matched_sources[catalog_idx]
                
                # Add position data
                matched_source.epochs.append(epoch.observation_time)
                matched_source.ra_measurements.append(source_data.get('ra', 0.0))
                matched_source.dec_measurements.append(source_data.get('dec', 0.0))
                matched_source.ra_errors.append(source_data.get('ra_err', 0.1))
                matched_source.dec_errors.append(source_data.get('dec_err', 0.1))
                matched_source.n_epochs += 1
                
                # Add photometry data
                filter_name = epoch.filter_name
                if filter_name not in matched_source.photometry:
                    matched_source.photometry[filter_name] = {
                        'magnitudes': [],
                        'magnitude_errors': [],
                        'epochs': []
                    }
                
                matched_source.photometry[filter_name]['magnitudes'].append(
                    source_data.get('magnitude', 99.0)
                )
                matched_source.photometry[filter_name]['magnitude_errors'].append(
                    source_data.get('magnitude_error', 1.0)
                )
                matched_source.photometry[filter_name]['epochs'].append(
                    epoch.observation_time
                )
    
    def _compute_matching_statistics(self, 
                                   matched_sources: List[MultiEpochSource], 
                                   epochs: List[EpochData]) -> Dict[str, Any]:
        """Compute statistics for source matching."""
        stats = {}
        
        # Total sources per epoch
        sources_per_epoch = [len(epoch.sources) for epoch in epochs]
        stats['mean_sources_per_epoch'] = np.mean(sources_per_epoch)
        stats['std_sources_per_epoch'] = np.std(sources_per_epoch)
        
        # Multi-epoch detection statistics
        n_epochs_per_source = [source.n_epochs for source in matched_sources]
        stats['mean_epochs_per_source'] = np.mean(n_epochs_per_source)
        stats['median_epochs_per_source'] = np.median(n_epochs_per_source)
        
        # Completeness as function of epoch
        epoch_completeness = []
        for i, epoch in enumerate(epochs):
            # Sources detected in this epoch
            detected = sum(1 for source in matched_sources 
                          if epoch.observation_time in source.epochs)
            completeness = detected / len(epoch.sources) if len(epoch.sources) > 0 else 0
            epoch_completeness.append(completeness)
        
        stats['epoch_completeness'] = epoch_completeness
        stats['mean_completeness'] = np.mean(epoch_completeness)
        
        return stats
    
    def _measure_proper_motions(self, matched_sources: List[MultiEpochSource]) -> List[Dict[str, Any]]:
        """Measure proper motions for sources with sufficient epochs."""
        pm_catalog = []
        
        for source in matched_sources:
            if (source.n_epochs >= self.config.pm_min_epochs and 
                source.time_span >= self.config.pm_min_timespan):
                
                pm_result = self._fit_proper_motion(source)
                
                if pm_result['significance'] > self.config.pm_sigma_threshold:
                    source.pmra = pm_result['pmra']
                    source.pmdec = pm_result['pmdec']
                    source.pmra_err = pm_result['pmra_err']
                    source.pmdec_err = pm_result['pmdec_err']
                    source.pm_significance = pm_result['significance']
                    
                    pm_catalog.append({
                        'source_id': source.source_id,
                        'ra': source.mean_ra,
                        'dec': source.mean_dec,
                        'pmra': pm_result['pmra'],
                        'pmdec': pm_result['pmdec'],
                        'pmra_err': pm_result['pmra_err'],
                        'pmdec_err': pm_result['pmdec_err'],
                        'pm_total': np.sqrt(pm_result['pmra']**2 + pm_result['pmdec']**2),
                        'significance': pm_result['significance'],
                        'n_epochs': source.n_epochs,
                        'time_span': source.time_span
                    })
        
        return pm_catalog
    
    def _fit_proper_motion(self, source: MultiEpochSource) -> Dict[str, float]:
        """Fit proper motion for a single source."""
        # Convert times to years from reference epoch
        times = np.array(source.epochs)
        ref_time = times[0]
        time_years = (times - ref_time) / 365.25
        
        # Position measurements in arcseconds
        ra_arcsec = (np.array(source.ra_measurements) - source.mean_ra) * 3600.0
        dec_arcsec = (np.array(source.dec_measurements) - source.mean_dec) * 3600.0
        
        # Convert to proper motion coordinates (account for declination)
        ra_arcsec *= np.cos(np.radians(source.mean_dec))
        
        # Error arrays
        ra_errors = np.array(source.ra_errors) * 3600.0 * np.cos(np.radians(source.mean_dec))
        dec_errors = np.array(source.dec_errors) * 3600.0
        
        # Fit linear proper motion
        if SCIPY_AVAILABLE and len(time_years) >= 3:
            # Weighted linear fit for RA
            weights_ra = 1.0 / ra_errors**2
            ra_fit = np.polyfit(time_years, ra_arcsec, 1, w=weights_ra)
            pmra = ra_fit[0] * 1000.0  # convert arcsec/year to mas/year
            
            # Weighted linear fit for Dec
            weights_dec = 1.0 / dec_errors**2
            dec_fit = np.polyfit(time_years, dec_arcsec, 1, w=weights_dec)
            pmdec = dec_fit[0] * 1000.0  # convert arcsec/year to mas/year
            
            # Estimate errors (simplified)
            ra_residuals = ra_arcsec - np.polyval(ra_fit, time_years)
            dec_residuals = dec_arcsec - np.polyval(dec_fit, time_years)
            
            ra_chi2 = np.sum((ra_residuals / ra_errors)**2)
            dec_chi2 = np.sum((dec_residuals / dec_errors)**2)
            
            # Proper motion errors (simplified)
            pmra_err = np.sqrt(ra_chi2 / (len(time_years) - 2)) / np.sqrt(np.sum(weights_ra)) * 1000.0
            pmdec_err = np.sqrt(dec_chi2 / (len(time_years) - 2)) / np.sqrt(np.sum(weights_dec)) * 1000.0
            
            # Combined significance
            pm_total = np.sqrt(pmra**2 + pmdec**2)
            pm_total_err = np.sqrt((pmra * pmra_err)**2 + (pmdec * pmdec_err)**2) / max(pm_total, 1e-6)
            significance = pm_total / pm_total_err
            
        else:
            # Simple difference for 2 epochs
            if len(time_years) == 2:
                dt = time_years[1] - time_years[0]
                pmra = (ra_arcsec[1] - ra_arcsec[0]) / dt * 1000.0
                pmdec = (dec_arcsec[1] - dec_arcsec[0]) / dt * 1000.0
                pmra_err = np.sqrt(ra_errors[0]**2 + ra_errors[1]**2) / dt * 1000.0
                pmdec_err = np.sqrt(dec_errors[0]**2 + dec_errors[1]**2) / dt * 1000.0
                
                pm_total = np.sqrt(pmra**2 + pmdec**2)
                pm_total_err = np.sqrt((pmra * pmra_err)**2 + (pmdec * pmdec_err)**2) / max(pm_total, 1e-6)
                significance = pm_total / pm_total_err
            else:
                pmra = pmdec = pmra_err = pmdec_err = significance = 0.0
        
        return {
            'pmra': pmra,
            'pmdec': pmdec,
            'pmra_err': pmra_err,
            'pmdec_err': pmdec_err,
            'significance': significance
        }
    
    def _detect_variability(self, matched_sources: List[MultiEpochSource]) -> List[VariabilityResult]:
        """Detect and characterize variability in multi-epoch sources."""
        variability_results = []
        
        for source in matched_sources:
            if source.n_epochs < self.config.min_epochs:
                continue
            
            # Analyze each filter separately
            for filter_name, phot_data in source.photometry.items():
                if len(phot_data['magnitudes']) < self.config.min_epochs:
                    continue
                
                result = self._analyze_light_curve(source, filter_name, phot_data)
                variability_results.append(result)
                
                # Update source variability properties
                if result.chi2_pvalue < 0.01:  # Significant variability
                    source.is_variable = True
                    source.variability_amplitude[filter_name] = result.amplitude
                    source.variability_significance[filter_name] = -np.log10(result.chi2_pvalue)
                    
                    if result.best_period > 0:
                        source.variability_period[filter_name] = result.best_period
                
                source.chi2_reduced[filter_name] = result.chi2_statistic / max(len(phot_data['magnitudes']) - 1, 1)
        
        return variability_results
    
    def _analyze_light_curve(self, 
                           source: MultiEpochSource, 
                           filter_name: str,
                           phot_data: Dict[str, List[float]]) -> VariabilityResult:
        """Analyze a single light curve for variability."""
        magnitudes = np.array(phot_data['magnitudes'])
        mag_errors = np.array(phot_data['magnitude_errors'])
        epochs = np.array(phot_data['epochs'])
        
        # Remove outliers if requested
        if self.config.outlier_rejection:
            good_mask = np.abs(magnitudes - np.median(magnitudes)) < self.config.outlier_sigma * np.std(magnitudes)
            magnitudes = magnitudes[good_mask]
            mag_errors = mag_errors[good_mask]
            epochs = epochs[good_mask]
        
        result = VariabilityResult(
            source_id=source.source_id,
            filter_name=filter_name,
            n_points=len(magnitudes),
            time_span=(np.max(epochs) - np.min(epochs))
        )
        
        if len(magnitudes) < 3:
            return result
        
        # Basic statistics
        result.mean_magnitude = np.mean(magnitudes)
        result.magnitude_rms = np.std(magnitudes)
        result.amplitude = np.max(magnitudes) - np.min(magnitudes)
        
        # Chi-squared test for constant brightness
        if SCIPY_AVAILABLE:
            weights = 1.0 / mag_errors**2
            weighted_mean = np.average(magnitudes, weights=weights)
            chi2 = np.sum(weights * (magnitudes - weighted_mean)**2)
            dof = len(magnitudes) - 1
            
            result.chi2_statistic = chi2
            result.chi2_pvalue = 1.0 - stats.chi2.cdf(chi2, dof)
        
        # IQR-based variability measure
        q75, q25 = np.percentile(magnitudes, [75, 25])
        iqr = q75 - q25
        median_error = np.median(mag_errors)
        result.iqr_ratio = iqr / median_error if median_error > 0 else 0
        
        # Eta variability index
        if len(magnitudes) > 2:
            result.eta_statistic = self._compute_eta_statistic(magnitudes, epochs)
        
        # Period search using Lomb-Scargle if enabled
        if (self.config.enable_time_series and ASTROPY_AVAILABLE and 
            len(epochs) >= 5 and result.time_span > 1.0):
            
            try:
                ls = LombScargle(epochs, magnitudes, mag_errors)
                
                # Define frequency grid
                min_freq = 1.0 / self.config.period_search_range[1]
                max_freq = 1.0 / self.config.period_search_range[0]
                frequency = np.linspace(min_freq, max_freq, 1000)
                
                # Compute periodogram
                power = ls.power(frequency)
                
                # Find best period
                best_freq_idx = np.argmax(power)
                result.best_period = 1.0 / frequency[best_freq_idx]
                result.period_power = power[best_freq_idx]
                
                # Compute false alarm probability
                result.period_fap = ls.false_alarm_probability(power[best_freq_idx])
                
            except Exception as e:
                self.logger.warning(f"Period search failed for {source.source_id}: {e}")
        
        # Classify variability type (basic classification)
        result.variability_class = self._classify_variability(result)
        
        return result
    
    def _compute_eta_statistic(self, magnitudes: np.ndarray, epochs: np.ndarray) -> float:
        """Compute eta variability statistic."""
        if len(magnitudes) < 3:
            return 0.0
        
        # Sort by time
        sort_idx = np.argsort(epochs)
        sorted_mags = magnitudes[sort_idx]
        
        # Compute consecutive differences
        diff_squared = np.diff(sorted_mags)**2
        mean_diff_squared = np.mean(diff_squared)
        
        # Total variance
        total_var = np.var(sorted_mags)
        
        # Eta statistic
        eta = mean_diff_squared / (2 * total_var) if total_var > 0 else 0
        
        return eta
    
    def _classify_variability(self, result: VariabilityResult) -> str:
        """Classify variability type based on light curve properties."""
        # Simple classification based on period and amplitude
        
        if result.period_fap < 0.01 and result.best_period > 0:
            if 0.1 < result.best_period < 1.0:
                return 'short_period_variable'
            elif 1.0 < result.best_period < 100.0:
                return 'long_period_variable'
            else:
                return 'periodic_variable'
        
        elif result.amplitude > 0.1:
            if result.iqr_ratio > 3.0:
                return 'irregular_variable'
            else:
                return 'slow_variable'
        
        elif result.chi2_pvalue < 0.01:
            return 'low_amplitude_variable'
        
        else:
            return 'constant'
    
    def _analyze_systematic_trends(self, 
                                 matched_sources: List[MultiEpochSource],
                                 epochs: List[EpochData]) -> Dict[str, Any]:
        """Analyze systematic trends across epochs."""
        systematic_analysis = {}
        
        # Magnitude trends vs time
        filter_trends = {}
        
        for filter_name in set(epoch.filter_name for epoch in epochs):
            filter_epochs = [epoch for epoch in epochs if epoch.filter_name == filter_name]
            
            if len(filter_epochs) < 3:
                continue
            
            # Collect all magnitude measurements
            all_times = []
            all_mags = []
            
            for epoch in filter_epochs:
                obs_time = epoch.observation_time
                
                for source in matched_sources:
                    if filter_name in source.photometry:
                        if obs_time in source.photometry[filter_name]['epochs']:
                            idx = source.photometry[filter_name]['epochs'].index(obs_time)
                            mag = source.photometry[filter_name]['magnitudes'][idx]
                            
                            if mag < 90:  # Valid magnitude
                                all_times.append(obs_time)
                                all_mags.append(mag)
            
            if len(all_times) > 10:
                # Fit linear trend
                if SCIPY_AVAILABLE and len(np.unique(all_times)) > 1:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(all_times, all_mags)
                    
                    filter_trends[filter_name] = {
                        'slope': slope,  # mag/day
                        'slope_err': std_err,
                        'correlation': r_value,
                        'p_value': p_value,
                        'n_points': len(all_times)
                    }
                else:
                    # Handle case where all times are identical or scipy unavailable
                    filter_trends[filter_name] = {
                        'slope': 0.0,
                        'slope_err': 0.0,
                        'correlation': 0.0,
                        'p_value': 1.0,
                        'n_points': len(all_times)
                    }
        
        systematic_analysis['magnitude_trends'] = filter_trends
        
        # Astrometric trends
        if len(matched_sources) > 0:
            ra_trends = []
            dec_trends = []
            
            for source in matched_sources:
                if len(source.epochs) >= 3:
                    times = np.array(source.epochs)
                    ras = np.array(source.ra_measurements)
                    decs = np.array(source.dec_measurements)
                    
                    if SCIPY_AVAILABLE:
                        # Remove proper motion trend
                        if source.pmra != 0 or source.pmdec != 0:
                            time_years = (times - times[0]) / 365.25
                            ra_predicted = source.mean_ra + source.pmra * time_years / (3600.0 * 1000.0)
                            dec_predicted = source.mean_dec + source.pmdec * time_years / (3600.0 * 1000.0)
                            
                            ra_residuals = ras - ra_predicted
                            dec_residuals = decs - dec_predicted
                            
                            ra_trends.extend(ra_residuals)
                            dec_trends.extend(dec_residuals)
            
            if len(ra_trends) > 10:
                systematic_analysis['astrometric_systematics'] = {
                    'ra_rms': np.std(ra_trends) * 3600.0,  # arcsec
                    'dec_rms': np.std(dec_trends) * 3600.0,
                    'n_measurements': len(ra_trends)
                }
        
        return systematic_analysis
    
    def _assess_multi_epoch_quality(self,
                                   matched_sources: List[MultiEpochSource],
                                   epochs: List[EpochData],
                                   matching_stats: Dict[str, Any]) -> Tuple[str, Dict[str, float]]:
        """Assess overall quality of multi-epoch analysis."""
        quality_metrics = {}
        
        # Completeness metric
        completeness = matching_stats.get('mean_completeness', 0.0)
        quality_metrics['completeness'] = completeness
        
        # Multi-epoch detection rate
        multi_epoch_sources = sum(1 for s in matched_sources if s.n_epochs >= 3)
        multi_epoch_rate = multi_epoch_sources / max(len(matched_sources), 1)
        quality_metrics['multi_epoch_rate'] = multi_epoch_rate
        
        # Time baseline
        total_timespan = (epochs[-1].observation_time - epochs[0].observation_time)
        quality_metrics['time_baseline'] = total_timespan  # days
        
        # Proper motion detection capability
        pm_detectable = sum(1 for s in matched_sources 
                           if s.n_epochs >= 3 and s.time_span >= 30.0)
        pm_detection_capability = pm_detectable / max(len(matched_sources), 1)
        quality_metrics['pm_detection_capability'] = pm_detection_capability
        
        # Overall quality assessment
        if (completeness > 0.8 and multi_epoch_rate > 0.5 and 
            total_timespan > 100.0 and pm_detection_capability > 0.3):
            overall_quality = 'Excellent'
        elif (completeness > 0.7 and multi_epoch_rate > 0.3 and 
              total_timespan > 30.0 and pm_detection_capability > 0.1):
            overall_quality = 'Good'
        elif completeness > 0.5 and multi_epoch_rate > 0.2:
            overall_quality = 'Fair'
        else:
            overall_quality = 'Poor'
        
        return overall_quality, quality_metrics
    
    def _create_output_products(self,
                              matched_sources: List[MultiEpochSource],
                              pm_catalog: List[Dict[str, Any]],
                              variability_catalog: List[VariabilityResult],
                              epochs: List[EpochData],
                              output_dir: Path) -> Dict[str, str]:
        """Create output products for multi-epoch analysis."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_files = {}
        
        # Save multi-epoch source catalog
        if ASTROPY_AVAILABLE:
            self._save_multi_epoch_catalog(matched_sources, output_dir / 'multi_epoch_catalog.fits')
            output_files['multi_epoch_catalog'] = str(output_dir / 'multi_epoch_catalog.fits')
        
        # Save proper motion catalog
        if pm_catalog:
            self._save_proper_motion_catalog(pm_catalog, output_dir / 'proper_motion_catalog.fits')
            output_files['proper_motion_catalog'] = str(output_dir / 'proper_motion_catalog.fits')
        
        # Save variability catalog
        if variability_catalog:
            self._save_variability_catalog(variability_catalog, output_dir / 'variability_catalog.fits')
            output_files['variability_catalog'] = str(output_dir / 'variability_catalog.fits')
        
        # Create plots if requested
        if self.config.create_lightcurves and MATPLOTLIB_AVAILABLE:
            self._create_sample_lightcurves(matched_sources, output_dir)
            output_files['sample_lightcurves'] = str(output_dir / 'sample_lightcurves.png')
        
        if self.config.create_proper_motion_plots and pm_catalog and MATPLOTLIB_AVAILABLE:
            self._create_proper_motion_plots(pm_catalog, output_dir)
            output_files['proper_motion_plots'] = str(output_dir / 'proper_motion_analysis.png')
        
        return output_files
    
    def _save_multi_epoch_catalog(self, sources: List[MultiEpochSource], output_path: Path) -> None:
        """Save multi-epoch source catalog to FITS file."""
        if not ASTROPY_AVAILABLE:
            return
        
        # Create table columns
        columns = [
            fits.Column(name='SOURCE_ID', array=[s.source_id for s in sources], format='20A'),
            fits.Column(name='MEAN_RA', array=[s.mean_ra for s in sources], format='D', unit='deg'),
            fits.Column(name='MEAN_DEC', array=[s.mean_dec for s in sources], format='D', unit='deg'),
            fits.Column(name='MEAN_RA_ERR', array=[s.mean_ra_err for s in sources], format='E', unit='arcsec'),
            fits.Column(name='MEAN_DEC_ERR', array=[s.mean_dec_err for s in sources], format='E', unit='arcsec'),
            fits.Column(name='PMRA', array=[s.pmra for s in sources], format='E', unit='mas/yr'),
            fits.Column(name='PMDEC', array=[s.pmdec for s in sources], format='E', unit='mas/yr'),
            fits.Column(name='PMRA_ERR', array=[s.pmra_err for s in sources], format='E', unit='mas/yr'),
            fits.Column(name='PMDEC_ERR', array=[s.pmdec_err for s in sources], format='E', unit='mas/yr'),
            fits.Column(name='PM_SIG', array=[s.pm_significance for s in sources], format='E'),
            fits.Column(name='N_EPOCHS', array=[s.n_epochs for s in sources], format='J'),
            fits.Column(name='TIME_SPAN', array=[s.time_span for s in sources], format='E', unit='day'),
            fits.Column(name='IS_VARIABLE', array=[s.is_variable for s in sources], format='L'),
            fits.Column(name='FLAGS', array=[s.flags for s in sources], format='J')
        ]
        
        # Create HDU and write
        hdu = fits.BinTableHDU.from_columns(columns)
        hdu.header['EXTNAME'] = 'MULTI_EPOCH_CATALOG'
        hdu.writeto(output_path, overwrite=True)
    
    def _save_proper_motion_catalog(self, pm_catalog: List[Dict[str, Any]], output_path: Path) -> None:
        """Save proper motion catalog to FITS file."""
        if not ASTROPY_AVAILABLE or not pm_catalog:
            return
        
        # Create table from catalog
        table = Table(pm_catalog)
        table.write(output_path, format='fits', overwrite=True)
    
    def _save_variability_catalog(self, variability_catalog: List[VariabilityResult], output_path: Path) -> None:
        """Save variability catalog to FITS file."""
        if not ASTROPY_AVAILABLE or not variability_catalog:
            return
        
        # Convert to dictionary format
        catalog_dict = {}
        for attr in ['source_id', 'filter_name', 'chi2_statistic', 'chi2_pvalue',
                     'eta_statistic', 'iqr_ratio', 'mean_magnitude', 'magnitude_rms',
                     'amplitude', 'best_period', 'period_power', 'period_fap',
                     'variability_class', 'n_points', 'time_span']:
            catalog_dict[attr] = [getattr(result, attr) for result in variability_catalog]
        
        table = Table(catalog_dict)
        table.write(output_path, format='fits', overwrite=True)
    
    def _create_sample_lightcurves(self, sources: List[MultiEpochSource], output_dir: Path) -> None:
        """Create sample light curves for variable sources."""
        if not MATPLOTLIB_AVAILABLE:
            return
        
        # Find most variable sources
        variable_sources = [s for s in sources if s.is_variable and s.n_epochs >= 5]
        
        if not variable_sources:
            return
        
        # Sort by variability significance and take top 6
        variable_sources.sort(key=lambda s: max(s.variability_significance.values(), default=0), reverse=True)
        sample_sources = variable_sources[:6]
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, source in enumerate(sample_sources):
            ax = axes[i]
            
            # Plot each filter
            colors = ['blue', 'green', 'red', 'orange', 'purple']
            
            for j, (filter_name, phot_data) in enumerate(source.photometry.items()):
                epochs = np.array(phot_data['epochs'])
                mags = np.array(phot_data['magnitudes'])
                errors = np.array(phot_data['magnitude_errors'])
                
                # Convert MJD to relative days
                rel_days = epochs - epochs[0]
                
                ax.errorbar(rel_days, mags, yerr=errors, 
                           color=colors[j % len(colors)], 
                           label=filter_name, fmt='o-', alpha=0.7)
            
            ax.set_xlabel('Days from first observation')
            ax.set_ylabel('Magnitude')
            ax.set_title(f'{source.source_id}')
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.invert_yaxis()  # Brighter magnitudes at top
        
        plt.tight_layout()
        plt.savefig(output_dir / f'sample_lightcurves.{self.config.plot_format}', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_proper_motion_plots(self, pm_catalog: List[Dict[str, Any]], output_dir: Path) -> None:
        """Create proper motion analysis plots."""
        if not MATPLOTLIB_AVAILABLE or not pm_catalog:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Extract data
        pmra = [entry['pmra'] for entry in pm_catalog]
        pmdec = [entry['pmdec'] for entry in pm_catalog]
        pm_total = [entry['pm_total'] for entry in pm_catalog]
        significance = [entry['significance'] for entry in pm_catalog]
        
        # Proper motion vector plot
        ax1.scatter(pmra, pmdec, c=significance, cmap='viridis', alpha=0.7)
        ax1.set_xlabel('PM RA (mas/yr)')
        ax1.set_ylabel('PM Dec (mas/yr)')
        ax1.set_title('Proper Motion Vectors')
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')
        
        # Proper motion magnitude histogram
        ax2.hist(pm_total, bins=20, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Total Proper Motion (mas/yr)')
        ax2.set_ylabel('Number of Sources')
        ax2.set_title('Proper Motion Distribution')
        ax2.grid(True, alpha=0.3)
        
        # Significance histogram
        ax3.hist(significance, bins=20, alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Detection Significance (σ)')
        ax3.set_ylabel('Number of Sources')
        ax3.set_title('PM Detection Significance')
        ax3.grid(True, alpha=0.3)
        
        # PM vs significance
        ax4.scatter(pm_total, significance, alpha=0.7)
        ax4.set_xlabel('Total Proper Motion (mas/yr)')
        ax4.set_ylabel('Detection Significance (σ)')
        ax4.set_title('PM Magnitude vs Significance')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'proper_motion_analysis.{self.config.plot_format}',
                   dpi=300, bbox_inches='tight')
        plt.close()


# Convenience functions

def quick_variability_check(source_epochs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Quick variability check for a source across epochs.
    
    Parameters:
    -----------
    source_epochs : list
        List of magnitude measurements across epochs
        
    Returns:
    --------
    dict
        Basic variability statistics
    """
    if len(source_epochs) < 3:
        return {'variable': False, 'reason': 'Insufficient epochs'}
    
    magnitudes = np.array([epoch['magnitude'] for epoch in source_epochs])
    mag_errors = np.array([epoch['magnitude_error'] for epoch in source_epochs])
    
    # Chi-squared test
    weights = 1.0 / mag_errors**2
    weighted_mean = np.average(magnitudes, weights=weights)
    chi2 = np.sum(weights * (magnitudes - weighted_mean)**2)
    dof = len(magnitudes) - 1
    
    if SCIPY_AVAILABLE:
        p_value = 1.0 - stats.chi2.cdf(chi2, dof)
        is_variable = p_value < 0.01
    else:
        is_variable = chi2 / dof > 3.0  # Simple threshold
        p_value = -1
    
    return {
        'variable': is_variable,
        'chi2': chi2,
        'p_value': p_value,
        'amplitude': np.max(magnitudes) - np.min(magnitudes),
        'rms': np.std(magnitudes),
        'n_epochs': len(magnitudes)
    }


if __name__ == "__main__":
    # Example usage
    print("JWST Multi-Epoch Analysis Module")
    print("This module provides multi-epoch analysis capabilities for JWST photometry")
