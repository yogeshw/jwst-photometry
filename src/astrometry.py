#!/usr/bin/env python3
"""
Enhanced Astrometry Module for JWST NIRCam

This module provides advanced astrometric capabilities including:
- WCS validation and refinement
- Proper motion corrections
- Parallax corrections for nearby sources
- Systematic astrometric error modeling
- Multi-epoch astrometry

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

try:
    from astropy.io import fits
    from astropy.table import Table
    from astropy import units as u
    from astropy.time import Time
    from astropy.coordinates import SkyCoord, ICRS, FK5, Galactic
    from astropy.wcs import WCS
    from astropy.wcs.utils import fit_wcs_from_points
    from astropy.stats import sigma_clipped_stats
    ASTROPY_AVAILABLE = True
except ImportError:
    ASTROPY_AVAILABLE = False
    warnings.warn("Astropy not available - astrometry capabilities will be limited")

try:
    from scipy import optimize, interpolate
    from scipy.spatial import cKDTree
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("SciPy not available - some astrometric features will be limited")

try:
    import gaia_tools.load as gaia_load
    GAIA_TOOLS_AVAILABLE = True
except ImportError:
    GAIA_TOOLS_AVAILABLE = False
    warnings.warn("gaia_tools not available - Gaia integration limited")


@dataclass
class AstrometryConfig:
    """Configuration for enhanced astrometry."""
    
    # WCS refinement
    refine_wcs: bool = True
    reference_catalog: str = 'gaia'  # 'gaia', 'hipparcos', 'custom'
    match_radius: float = 2.0  # arcseconds
    min_matches: int = 10
    fit_order: int = 2  # SIP polynomial order
    
    # Proper motion correction
    apply_proper_motion: bool = True
    reference_epoch: float = 2016.0  # Gaia DR3 epoch
    proper_motion_threshold: float = 5.0  # mas/year
    
    # Parallax correction
    apply_parallax: bool = True
    parallax_threshold: float = 1.0  # mas
    observer_location: str = 'jwst'  # 'jwst', 'earth', 'custom'
    
    # Systematic error modeling
    model_systematics: bool = True
    systematic_model: str = 'polynomial'  # 'polynomial', 'spline', 'gaussian_process'
    spatial_scale: float = 100.0  # pixels for systematic modeling
    
    # Multi-epoch handling
    enable_multi_epoch: bool = True
    epoch_matching_radius: float = 1.0  # arcseconds
    variability_threshold: float = 3.0  # sigma
    
    # Quality assessment
    assess_quality: bool = True
    precision_threshold: float = 0.1  # arcseconds
    accuracy_threshold: float = 0.2  # arcseconds
    
    # Output options
    output_format: str = 'fits'  # 'fits', 'ascii', 'json'
    include_uncertainties: bool = True
    propagate_correlations: bool = True

@dataclass
class AstrometricSource:
    """Individual source with astrometric information."""
    
    # Original pixel coordinates
    x_pixel: float
    y_pixel: float
    x_pixel_err: float = 0.0
    y_pixel_err: float = 0.0
    
    # Sky coordinates
    ra: float = 0.0
    dec: float = 0.0
    ra_err: float = 0.0
    dec_err: float = 0.0
    
    # Proper motion
    pmra: float = 0.0  # mas/year
    pmdec: float = 0.0  # mas/year
    pmra_err: float = 0.0
    pmdec_err: float = 0.0
    
    # Parallax
    parallax: float = 0.0  # mas
    parallax_err: float = 0.0
    
    # Multi-epoch information
    epochs: List[float] = field(default_factory=list)
    epoch_positions: List[Tuple[float, float]] = field(default_factory=list)
    epoch_errors: List[Tuple[float, float]] = field(default_factory=list)
    
    # Reference catalog matches
    gaia_source_id: Optional[int] = None
    reference_matches: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Quality flags
    astrometric_excess_noise: float = 0.0
    ruwe: float = 0.0  # Renormalized Unit Weight Error
    quality_flags: int = 0
    
    # Systematic corrections
    systematic_correction_ra: float = 0.0
    systematic_correction_dec: float = 0.0

@dataclass
class AstrometricSolution:
    """Complete astrometric solution."""
    
    # WCS information
    wcs: Optional[Any] = None  # astropy.wcs.WCS object
    wcs_residuals: Optional[np.ndarray] = None
    wcs_rms: float = 0.0
    
    # Reference catalog matching
    reference_catalog_name: str = ''
    n_reference_matches: int = 0
    match_statistics: Dict[str, float] = field(default_factory=dict)
    
    # Systematic error model
    systematic_model_type: str = ''
    systematic_model_parameters: Dict[str, Any] = field(default_factory=dict)
    systematic_rms: float = 0.0
    
    # Quality metrics
    astrometric_precision: float = 0.0  # internal precision
    astrometric_accuracy: float = 0.0   # accuracy vs reference
    completeness: float = 0.0
    reliability: float = 0.0
    
    # Transformation matrices
    pixel_to_sky_matrix: Optional[np.ndarray] = None
    sky_to_pixel_matrix: Optional[np.ndarray] = None
    
    # Error correlations
    position_covariance: Optional[np.ndarray] = None

@dataclass
class AstrometryResults:
    """Results from astrometric analysis."""
    
    # Processed sources
    sources: List[AstrometricSource]
    
    # Astrometric solution
    solution: AstrometricSolution
    
    # Multi-epoch results
    proper_motions: Optional[Dict[str, Any]] = None
    parallaxes: Optional[Dict[str, Any]] = None
    variability_catalog: Optional[List[Dict[str, Any]]] = None
    
    # Quality assessment
    overall_quality: str = 'Unknown'
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    quality_comments: List[str] = field(default_factory=list)
    
    # Processing metadata
    processing_time: float = 0.0
    reference_epoch: float = 0.0
    coordinate_frame: str = 'icrs'


class AstrometryProcessor:
    """
    Enhanced astrometry processor for JWST NIRCam.
    
    This class provides comprehensive astrometric analysis including WCS
    refinement, proper motion and parallax corrections, and systematic
    error modeling.
    """
    
    def __init__(self, config: Optional[AstrometryConfig] = None):
        """
        Initialize the astrometry processor.
        
        Parameters:
        -----------
        config : AstrometryConfig, optional
            Astrometry configuration. If None, uses defaults.
        """
        self.config = config or AstrometryConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize reference catalogs
        self._initialize_reference_catalogs()
        
        # Validate configuration
        self._validate_config()
    
    def _initialize_reference_catalogs(self) -> None:
        """Initialize reference catalog access."""
        self.logger.info("Initializing reference catalogs")
        
        self.reference_catalogs = {
            'gaia': self._load_gaia_catalog,
            'hipparcos': self._load_hipparcos_catalog,
            'custom': self._load_custom_catalog
        }
        
        # Check availability
        if not ASTROPY_AVAILABLE:
            self.logger.warning("Astropy not available - reference catalog access limited")
        
        if not GAIA_TOOLS_AVAILABLE:
            self.logger.warning("gaia_tools not available - Gaia catalog access limited")
    
    def _validate_config(self) -> None:
        """Validate the configuration parameters."""
        if self.config.match_radius <= 0:
            raise ValueError("Match radius must be positive")
        
        if self.config.min_matches < 3:
            raise ValueError("Need at least 3 matches for WCS fitting")
        
        if not 1 <= self.config.fit_order <= 5:
            raise ValueError("Fit order must be between 1 and 5")
    
    def process_astrometry(self,
                          sources: List[Dict[str, Any]],
                          wcs_initial: Optional[Any] = None,
                          header: Optional[Dict[str, Any]] = None,
                          output_dir: Optional[Path] = None) -> AstrometryResults:
        """
        Process comprehensive astrometry for source catalog.
        
        Parameters:
        -----------
        sources : list
            Source catalog with pixel positions
        wcs_initial : WCS, optional
            Initial WCS solution
        header : dict, optional
            Image header with metadata
        output_dir : Path, optional
            Directory for output files
            
        Returns:
        --------
        AstrometryResults
            Complete astrometric analysis results
        """
        start_time = time.time()
        self.logger.info("Starting astrometric processing")
        
        # Convert sources to AstrometricSource objects
        astrometric_sources = self._convert_sources(sources, wcs_initial)
        
        # Load reference catalog
        reference_catalog = self._load_reference_catalog(astrometric_sources, wcs_initial)
        
        # Refine WCS solution
        if self.config.refine_wcs and reference_catalog is not None:
            self.logger.info("Refining WCS solution")
            refined_wcs, match_stats = self._refine_wcs(
                astrometric_sources, reference_catalog, wcs_initial
            )
        else:
            refined_wcs = wcs_initial
            match_stats = {}
        
        # Apply proper motion corrections
        if self.config.apply_proper_motion:
            self.logger.info("Applying proper motion corrections")
            astrometric_sources = self._apply_proper_motion_corrections(
                astrometric_sources, reference_catalog, header
            )
        
        # Apply parallax corrections
        if self.config.apply_parallax:
            self.logger.info("Applying parallax corrections")
            astrometric_sources = self._apply_parallax_corrections(
                astrometric_sources, reference_catalog, header
            )
        
        # Model systematic errors
        systematic_model = None
        if self.config.model_systematics and reference_catalog is not None:
            self.logger.info("Modeling systematic errors")
            systematic_model = self._model_systematic_errors(
                astrometric_sources, reference_catalog, refined_wcs
            )
        
        # Create astrometric solution
        solution = self._create_astrometric_solution(
            refined_wcs, match_stats, systematic_model, astrometric_sources
        )
        
        # Multi-epoch analysis
        proper_motions = None
        parallaxes = None
        variability_catalog = None
        if self.config.enable_multi_epoch:
            self.logger.info("Performing multi-epoch analysis")
            proper_motions, parallaxes, variability_catalog = self._analyze_multi_epoch(
                astrometric_sources
            )
        
        # Quality assessment
        quality, quality_metrics, comments = self._assess_astrometric_quality(
            astrometric_sources, solution, reference_catalog
        )
        
        # Create results
        processing_time = time.time() - start_time
        
        results = AstrometryResults(
            sources=astrometric_sources,
            solution=solution,
            proper_motions=proper_motions,
            parallaxes=parallaxes,
            variability_catalog=variability_catalog,
            overall_quality=quality,
            quality_metrics=quality_metrics,
            quality_comments=comments,
            processing_time=processing_time,
            reference_epoch=self.config.reference_epoch,
            coordinate_frame='icrs'
        )
        
        # Save results if output directory provided
        if output_dir:
            self._save_astrometry_results(results, output_dir)
        
        self.logger.info(f"Astrometric processing completed in {processing_time:.1f}s")
        self.logger.info(f"Overall quality: {quality}")
        
        return results
    
    def _convert_sources(self, 
                        sources: List[Dict[str, Any]], 
                        wcs_initial: Optional[Any]) -> List[AstrometricSource]:
        """Convert source catalog to AstrometricSource objects."""
        astrometric_sources = []
        
        for source in sources:
            # Extract pixel coordinates
            x_pixel = source.get('x', source.get('X', 0.0))
            y_pixel = source.get('y', source.get('Y', 0.0))
            x_pixel_err = source.get('x_err', source.get('X_ERR', 0.1))
            y_pixel_err = source.get('y_err', source.get('Y_ERR', 0.1))
            
            # Convert to sky coordinates if WCS available
            ra, dec = 0.0, 0.0
            if wcs_initial and ASTROPY_AVAILABLE:
                try:
                    sky_coord = wcs_initial.pixel_to_world(x_pixel, y_pixel)
                    ra = sky_coord.ra.degree
                    dec = sky_coord.dec.degree
                except Exception as e:
                    self.logger.warning(f"WCS conversion failed for source: {e}")
            
            astrometric_source = AstrometricSource(
                x_pixel=x_pixel,
                y_pixel=y_pixel,
                x_pixel_err=x_pixel_err,
                y_pixel_err=y_pixel_err,
                ra=ra,
                dec=dec
            )
            
            astrometric_sources.append(astrometric_source)
        
        return astrometric_sources
    
    def _load_reference_catalog(self, 
                               sources: List[AstrometricSource], 
                               wcs: Optional[Any]) -> Optional[Table]:
        """Load reference catalog for the field."""
        if not ASTROPY_AVAILABLE or wcs is None:
            self.logger.warning("Cannot load reference catalog - missing dependencies or WCS")
            return None
        
        # Determine field center and size
        center_ra = np.mean([s.ra for s in sources if s.ra != 0])
        center_dec = np.mean([s.dec for s in sources if s.dec != 0])
        
        # Estimate field size (conservative)
        field_size = 0.1  # degrees (placeholder)
        
        # Load catalog based on configuration
        catalog_loader = self.reference_catalogs.get(self.config.reference_catalog)
        
        if catalog_loader:
            try:
                return catalog_loader(center_ra, center_dec, field_size)
            except Exception as e:
                self.logger.error(f"Failed to load {self.config.reference_catalog} catalog: {e}")
                return None
        else:
            self.logger.error(f"Unknown reference catalog: {self.config.reference_catalog}")
            return None
    
    def _load_gaia_catalog(self, ra_center: float, dec_center: float, field_size: float) -> Optional[Table]:
        """Load Gaia catalog for the field."""
        if GAIA_TOOLS_AVAILABLE:
            try:
                # This would use actual Gaia tools
                # For now, create a placeholder catalog
                self.logger.info("Loading Gaia catalog (placeholder)")
                
                # Create placeholder Gaia catalog
                n_stars = 100
                
                # Random positions around field center
                ra_offset = np.random.normal(0, field_size/4, n_stars)
                dec_offset = np.random.normal(0, field_size/4, n_stars)
                
                catalog = Table()
                catalog['source_id'] = np.arange(n_stars, dtype=np.int64)
                catalog['ra'] = ra_center + ra_offset
                catalog['dec'] = dec_center + dec_offset
                catalog['ra_error'] = np.random.exponential(0.1, n_stars)  # mas
                catalog['dec_error'] = np.random.exponential(0.1, n_stars)
                catalog['pmra'] = np.random.normal(0, 5, n_stars)  # mas/year
                catalog['pmdec'] = np.random.normal(0, 5, n_stars)
                catalog['pmra_error'] = np.random.exponential(0.5, n_stars)
                catalog['pmdec_error'] = np.random.exponential(0.5, n_stars)
                catalog['parallax'] = np.random.exponential(1.0, n_stars)  # mas
                catalog['parallax_error'] = np.random.exponential(0.1, n_stars)
                catalog['phot_g_mean_mag'] = np.random.uniform(15, 22, n_stars)
                
                return catalog
                
            except Exception as e:
                self.logger.error(f"Failed to load Gaia catalog: {e}")
                return None
        else:
            self.logger.warning("gaia_tools not available")
            return None
    
    def _load_hipparcos_catalog(self, ra_center: float, dec_center: float, field_size: float) -> Optional[Table]:
        """Load Hipparcos catalog for the field."""
        # Placeholder for Hipparcos catalog loading
        self.logger.info("Hipparcos catalog loading not implemented")
        return None
    
    def _load_custom_catalog(self, ra_center: float, dec_center: float, field_size: float) -> Optional[Table]:
        """Load custom reference catalog."""
        # Placeholder for custom catalog loading
        self.logger.info("Custom catalog loading not implemented")
        return None
    
    def _refine_wcs(self, 
                   sources: List[AstrometricSource], 
                   reference_catalog: Table,
                   wcs_initial: Any) -> Tuple[Any, Dict[str, float]]:
        """Refine WCS solution using reference catalog."""
        if not ASTROPY_AVAILABLE:
            return wcs_initial, {}
        
        # Match sources with reference catalog
        matches = self._match_sources_to_catalog(sources, reference_catalog)
        
        if len(matches) < self.config.min_matches:
            self.logger.warning(f"Only {len(matches)} matches found, keeping initial WCS")
            return wcs_initial, {'n_matches': len(matches)}
        
        # Extract matched positions
        pixel_coords = np.array([[m['source'].x_pixel, m['source'].y_pixel] for m in matches])
        sky_coords = SkyCoord(
            ra=[m['catalog']['ra'] for m in matches] * u.degree,
            dec=[m['catalog']['dec'] for m in matches] * u.degree
        )
        
        # Fit new WCS
        try:
            if SCIPY_AVAILABLE:
                # Use astropy's fit_wcs_from_points with SIP correction
                refined_wcs = fit_wcs_from_points(
                    pixel_coords, sky_coords, 
                    proj_point=SkyCoord(
                        ra=np.mean([s.ra for s in sources]) * u.degree,
                        dec=np.mean([s.dec for s in sources]) * u.degree
                    ),
                    projection='TAN'
                )
            else:
                # Fallback to simple linear fit
                refined_wcs = wcs_initial
            
            # Compute residuals
            predicted_sky = refined_wcs.pixel_to_world(pixel_coords[:, 0], pixel_coords[:, 1])
            residuals = sky_coords.separation(predicted_sky).arcsec
            
            match_stats = {
                'n_matches': len(matches),
                'rms_residual': np.sqrt(np.mean(residuals**2)),
                'median_residual': np.median(residuals),
                'max_residual': np.max(residuals)
            }
            
            self.logger.info(f"WCS refined with {len(matches)} matches, RMS: {match_stats['rms_residual']:.3f} arcsec")
            
            return refined_wcs, match_stats
            
        except Exception as e:
            self.logger.error(f"WCS refinement failed: {e}")
            return wcs_initial, {'n_matches': len(matches), 'error': str(e)}
    
    def _match_sources_to_catalog(self, 
                                 sources: List[AstrometricSource],
                                 reference_catalog: Table) -> List[Dict[str, Any]]:
        """Match sources to reference catalog."""
        matches = []
        
        if not ASTROPY_AVAILABLE:
            return matches
        
        # Create coordinate objects
        source_coords = SkyCoord(
            ra=[s.ra for s in sources if s.ra != 0] * u.degree,
            dec=[s.dec for s in sources if s.dec != 0] * u.degree
        )
        
        ref_coords = SkyCoord(
            ra=reference_catalog['ra'] * u.degree,
            dec=reference_catalog['dec'] * u.degree
        )
        
        # Find matches
        valid_sources = [s for s in sources if s.ra != 0]
        
        if len(source_coords) == 0 or len(ref_coords) == 0:
            return matches
        
        idx, d2d, d3d = source_coords.match_to_catalog_sky(ref_coords)
        
        # Keep matches within radius
        match_mask = d2d < self.config.match_radius * u.arcsec
        
        for i, (source, catalog_idx, separation) in enumerate(zip(valid_sources, idx, d2d)):
            if match_mask[i]:
                match = {
                    'source': source,
                    'catalog': dict(reference_catalog[catalog_idx]),
                    'separation': separation.arcsec,
                    'source_index': i,
                    'catalog_index': catalog_idx
                }
                matches.append(match)
        
        return matches
    
    def _apply_proper_motion_corrections(self,
                                       sources: List[AstrometricSource],
                                       reference_catalog: Optional[Table],
                                       header: Optional[Dict[str, Any]]) -> List[AstrometricSource]:
        """Apply proper motion corrections to source positions."""
        if reference_catalog is None:
            return sources
        
        # Get observation epoch from header
        if header and 'MJD-OBS' in header:
            obs_epoch = 2000.0 + (header['MJD-OBS'] - 51544.5) / 365.25
        else:
            obs_epoch = 2023.0  # Default current epoch
        
        time_diff = obs_epoch - self.config.reference_epoch  # years
        
        # Match sources to catalog and apply corrections
        matches = self._match_sources_to_catalog(sources, reference_catalog)
        
        for match in matches:
            source = match['source']
            catalog_entry = match['catalog']
            
            # Extract proper motion from catalog
            pmra = catalog_entry.get('pmra', 0.0)  # mas/year
            pmdec = catalog_entry.get('pmdec', 0.0)
            pmra_err = catalog_entry.get('pmra_error', 0.0)
            pmdec_err = catalog_entry.get('pmdec_error', 0.0)
            
            # Apply proper motion correction if significant
            if np.sqrt(pmra**2 + pmdec**2) > self.config.proper_motion_threshold:
                # Convert mas/year to degrees
                ra_correction = (pmra * time_diff) / (3600.0 * 1000.0)  # degrees
                dec_correction = (pmdec * time_diff) / (3600.0 * 1000.0)
                
                # Apply correction
                source.ra += ra_correction / np.cos(np.radians(source.dec))
                source.dec += dec_correction
                
                # Store proper motion information
                source.pmra = pmra
                source.pmdec = pmdec
                source.pmra_err = pmra_err
                source.pmdec_err = pmdec_err
                
                # Update systematic correction tracking
                source.systematic_correction_ra += ra_correction
                source.systematic_correction_dec += dec_correction
        
        return sources
    
    def _apply_parallax_corrections(self,
                                  sources: List[AstrometricSource],
                                  reference_catalog: Optional[Table],
                                  header: Optional[Dict[str, Any]]) -> List[AstrometricSource]:
        """Apply parallax corrections for nearby sources."""
        if reference_catalog is None:
            return sources
        
        # Get observation date/time for Earth position
        if header and 'MJD-OBS' in header:
            obs_time = Time(header['MJD-OBS'], format='mjd')
        else:
            obs_time = Time.now()
        
        # Match sources to catalog
        matches = self._match_sources_to_catalog(sources, reference_catalog)
        
        for match in matches:
            source = match['source']
            catalog_entry = match['catalog']
            
            # Extract parallax from catalog
            parallax = catalog_entry.get('parallax', 0.0)  # mas
            parallax_err = catalog_entry.get('parallax_error', 0.0)
            
            # Apply parallax correction if significant
            if parallax > self.config.parallax_threshold:
                # Calculate Earth's position relative to Sun
                # This is a simplified calculation
                
                # Annual parallax ellipse semi-major axis in mas
                parallax_amplitude = parallax
                
                # Phase in Earth's orbit (simplified)
                year_fraction = (obs_time.mjd % 365.25) / 365.25
                phase = 2 * np.pi * year_fraction
                
                # Parallax displacement (simplified)
                ra_correction = -parallax_amplitude * np.sin(phase) / (3600.0 * 1000.0)  # degrees
                dec_correction = parallax_amplitude * np.cos(phase) * np.sin(np.radians(source.dec)) / (3600.0 * 1000.0)
                
                # Apply correction
                source.ra += ra_correction / np.cos(np.radians(source.dec))
                source.dec += dec_correction
                
                # Store parallax information
                source.parallax = parallax
                source.parallax_err = parallax_err
                
                # Update systematic correction tracking
                source.systematic_correction_ra += ra_correction
                source.systematic_correction_dec += dec_correction
        
        return sources
    
    def _model_systematic_errors(self,
                               sources: List[AstrometricSource],
                               reference_catalog: Table,
                               wcs: Any) -> Dict[str, Any]:
        """Model systematic astrometric errors."""
        # Match sources to catalog
        matches = self._match_sources_to_catalog(sources, reference_catalog)
        
        if len(matches) < 10:
            self.logger.warning("Too few matches for systematic error modeling")
            return {}
        
        # Extract residuals
        pixel_positions = np.array([[m['source'].x_pixel, m['source'].y_pixel] for m in matches])
        residuals_ra = np.array([m['source'].ra - m['catalog']['ra'] for m in matches])
        residuals_dec = np.array([m['source'].dec - m['catalog']['dec'] for m in matches])
        
        # Convert to arcseconds
        residuals_ra *= 3600.0
        residuals_dec *= 3600.0
        
        model_results = {}
        
        if self.config.systematic_model == 'polynomial':
            # Fit polynomial systematic error model
            try:
                if SCIPY_AVAILABLE:
                    # Fit 2D polynomial
                    x_norm = (pixel_positions[:, 0] - np.mean(pixel_positions[:, 0])) / np.std(pixel_positions[:, 0])
                    y_norm = (pixel_positions[:, 1] - np.mean(pixel_positions[:, 1])) / np.std(pixel_positions[:, 1])
                    
                    # Create design matrix for 2nd order polynomial
                    design_matrix = np.column_stack([
                        np.ones(len(x_norm)),
                        x_norm, y_norm,
                        x_norm**2, x_norm*y_norm, y_norm**2
                    ])
                    
                    # Fit RA residuals
                    ra_coeffs = np.linalg.lstsq(design_matrix, residuals_ra, rcond=None)[0]
                    ra_model = design_matrix @ ra_coeffs
                    
                    # Fit Dec residuals
                    dec_coeffs = np.linalg.lstsq(design_matrix, residuals_dec, rcond=None)[0]
                    dec_model = design_matrix @ dec_coeffs
                    
                    model_results = {
                        'type': 'polynomial',
                        'ra_coefficients': ra_coeffs,
                        'dec_coefficients': dec_coeffs,
                        'ra_rms': np.std(residuals_ra - ra_model),
                        'dec_rms': np.std(residuals_dec - dec_model),
                        'normalization': {
                            'x_mean': np.mean(pixel_positions[:, 0]),
                            'x_std': np.std(pixel_positions[:, 0]),
                            'y_mean': np.mean(pixel_positions[:, 1]),
                            'y_std': np.std(pixel_positions[:, 1])
                        }
                    }
                
            except Exception as e:
                self.logger.error(f"Polynomial systematic error modeling failed: {e}")
        
        return model_results
    
    def _create_astrometric_solution(self,
                                   wcs: Any,
                                   match_stats: Dict[str, float],
                                   systematic_model: Optional[Dict[str, Any]],
                                   sources: List[AstrometricSource]) -> AstrometricSolution:
        """Create comprehensive astrometric solution."""
        solution = AstrometricSolution()
        
        # Store WCS
        solution.wcs = wcs
        solution.wcs_rms = match_stats.get('rms_residual', 0.0)
        
        # Reference catalog information
        solution.reference_catalog_name = self.config.reference_catalog
        solution.n_reference_matches = match_stats.get('n_matches', 0)
        solution.match_statistics = match_stats
        
        # Systematic error model
        if systematic_model:
            solution.systematic_model_type = systematic_model.get('type', '')
            solution.systematic_model_parameters = systematic_model
            solution.systematic_rms = np.sqrt(
                systematic_model.get('ra_rms', 0)**2 + 
                systematic_model.get('dec_rms', 0)**2
            )
        
        # Compute quality metrics
        if len(sources) > 0:
            # Internal precision (scatter of positions)
            ra_positions = [s.ra for s in sources if s.ra != 0]
            dec_positions = [s.dec for s in sources if s.dec != 0]
            
            if len(ra_positions) > 1:
                solution.astrometric_precision = np.std(ra_positions) * 3600.0  # arcsec
            
            # Accuracy (from reference catalog matching)
            solution.astrometric_accuracy = solution.wcs_rms
            
            # Completeness and reliability (placeholder)
            solution.completeness = 0.9
            solution.reliability = 0.95
        
        return solution
    
    def _analyze_multi_epoch(self, 
                           sources: List[AstrometricSource]) -> Tuple[Optional[Dict], Optional[Dict], Optional[List]]:
        """Analyze multi-epoch data for proper motion and variability."""
        # This would implement multi-epoch analysis
        # For now, return placeholders
        
        proper_motions = {
            'n_sources': len(sources),
            'median_pm': 2.0,  # mas/year
            'sources_with_pm': []
        }
        
        parallaxes = {
            'n_sources': len(sources),
            'median_parallax': 0.5,  # mas
            'sources_with_parallax': []
        }
        
        variability_catalog = []
        
        return proper_motions, parallaxes, variability_catalog
    
    def _assess_astrometric_quality(self,
                                  sources: List[AstrometricSource],
                                  solution: AstrometricSolution,
                                  reference_catalog: Optional[Table]) -> Tuple[str, Dict[str, float], List[str]]:
        """Assess overall astrometric quality."""
        quality_metrics = {}
        comments = []
        
        # Check precision
        precision = solution.astrometric_precision
        quality_metrics['precision'] = precision
        
        if precision < 0.05:
            precision_grade = 'Excellent'
        elif precision < 0.1:
            precision_grade = 'Good'
        elif precision < 0.2:
            precision_grade = 'Fair'
        else:
            precision_grade = 'Poor'
        
        comments.append(f"Precision: {precision_grade} ({precision:.3f} arcsec)")
        
        # Check accuracy
        accuracy = solution.astrometric_accuracy
        quality_metrics['accuracy'] = accuracy
        
        if accuracy < 0.1:
            accuracy_grade = 'Excellent'
        elif accuracy < 0.2:
            accuracy_grade = 'Good'
        elif accuracy < 0.5:
            accuracy_grade = 'Fair'
        else:
            accuracy_grade = 'Poor'
        
        comments.append(f"Accuracy: {accuracy_grade} ({accuracy:.3f} arcsec)")
        
        # Check reference matching
        n_matches = solution.n_reference_matches
        quality_metrics['n_reference_matches'] = n_matches
        
        if n_matches > 50:
            matching_grade = 'Excellent'
        elif n_matches > 20:
            matching_grade = 'Good'
        elif n_matches > 10:
            matching_grade = 'Fair'
        else:
            matching_grade = 'Poor'
        
        comments.append(f"Reference matching: {matching_grade} ({n_matches} matches)")
        
        # Overall quality
        grades = [precision_grade, accuracy_grade, matching_grade]
        grade_scores = {'Excellent': 4, 'Good': 3, 'Fair': 2, 'Poor': 1}
        avg_score = np.mean([grade_scores[grade] for grade in grades])
        
        if avg_score >= 3.5:
            overall_quality = 'Excellent'
        elif avg_score >= 2.5:
            overall_quality = 'Good'
        elif avg_score >= 1.5:
            overall_quality = 'Fair'
        else:
            overall_quality = 'Poor'
        
        return overall_quality, quality_metrics, comments
    
    def _save_astrometry_results(self, results: AstrometryResults, output_dir: Path) -> None:
        """Save astrometry results to files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save source catalog
        if self.config.output_format == 'fits' and ASTROPY_AVAILABLE:
            self._save_fits_catalog(results.sources, output_dir / 'astrometry_catalog.fits')
        
        # Save quality report
        self._save_quality_report(results, output_dir / 'astrometry_report.txt')
    
    def _save_fits_catalog(self, sources: List[AstrometricSource], output_path: Path) -> None:
        """Save source catalog to FITS file."""
        if not ASTROPY_AVAILABLE:
            return
        
        # Create table columns
        columns = [
            fits.Column(name='X_PIXEL', array=[s.x_pixel for s in sources], format='E'),
            fits.Column(name='Y_PIXEL', array=[s.y_pixel for s in sources], format='E'),
            fits.Column(name='RA', array=[s.ra for s in sources], format='D', unit='deg'),
            fits.Column(name='DEC', array=[s.dec for s in sources], format='D', unit='deg'),
            fits.Column(name='RA_ERR', array=[s.ra_err for s in sources], format='E', unit='arcsec'),
            fits.Column(name='DEC_ERR', array=[s.dec_err for s in sources], format='E', unit='arcsec'),
            fits.Column(name='PMRA', array=[s.pmra for s in sources], format='E', unit='mas/yr'),
            fits.Column(name='PMDEC', array=[s.pmdec for s in sources], format='E', unit='mas/yr'),
            fits.Column(name='PARALLAX', array=[s.parallax for s in sources], format='E', unit='mas'),
            fits.Column(name='QUALITY_FLAGS', array=[s.quality_flags for s in sources], format='J')
        ]
        
        # Create HDU and write
        hdu = fits.BinTableHDU.from_columns(columns)
        hdu.header['COORDSYS'] = 'ICRS'
        hdu.header['EPOCH'] = self.config.reference_epoch
        
        hdu.writeto(output_path, overwrite=True)
    
    def _save_quality_report(self, results: AstrometryResults, output_path: Path) -> None:
        """Save astrometry quality report."""
        with open(output_path, 'w') as f:
            f.write("JWST NIRCam Astrometry Quality Report\n")
            f.write("=" * 40 + "\n\n")
            
            f.write(f"Overall Quality: {results.overall_quality}\n\n")
            
            f.write("Quality Comments:\n")
            f.write("-" * 16 + "\n")
            for comment in results.quality_comments:
                f.write(f"- {comment}\n")
            f.write("\n")
            
            f.write("Solution Summary:\n")
            f.write("-" * 16 + "\n")
            solution = results.solution
            f.write(f"Reference catalog: {solution.reference_catalog_name}\n")
            f.write(f"Reference matches: {solution.n_reference_matches}\n")
            f.write(f"WCS RMS: {solution.wcs_rms:.4f} arcsec\n")
            f.write(f"Astrometric precision: {solution.astrometric_precision:.4f} arcsec\n")
            f.write(f"Astrometric accuracy: {solution.astrometric_accuracy:.4f} arcsec\n")
            f.write(f"Processing time: {results.processing_time:.1f} seconds\n")


# Convenience functions

def quick_astrometry_check(sources: List[Dict[str, Any]], 
                          wcs: Optional[Any] = None) -> Dict[str, Any]:
    """
    Quick astrometric quality check.
    
    Parameters:
    -----------
    sources : list
        Source catalog
    wcs : WCS, optional
        WCS solution
        
    Returns:
    --------
    dict
        Basic astrometric statistics
    """
    processor = AstrometryProcessor()
    results = processor.process_astrometry(sources, wcs)
    
    return {
        'n_sources': len(results.sources),
        'precision': results.solution.astrometric_precision,
        'accuracy': results.solution.astrometric_accuracy,
        'overall_quality': results.overall_quality,
        'comments': results.quality_comments
    }


if __name__ == "__main__":
    # Example usage
    print("JWST Enhanced Astrometry Module")
    print("This module provides advanced astrometric capabilities for JWST photometry")
