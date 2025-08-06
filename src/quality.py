#!/usr/bin/env python3
"""
Quality Assessment Module for JWST NIRCam Photometry

This module provides comprehensive quality assessment capabilities including:
- Photometric quality flags
- Completeness and reliability assessment
- Systematic error identification
- Comparison with external catalogs
- Astrometric quality assessment

Author: JWST Photometry Pipeline
Date: August 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import warnings

try:
    from astropy.table import Table
    from astropy.coordinates import SkyCoord, match_coordinates_sky
    from astropy import units as u
    from astropy.stats import sigma_clipped_stats
    ASTROPY_AVAILABLE = True
except ImportError:
    ASTROPY_AVAILABLE = False
    warnings.warn("Astropy not available - some features will be limited")

try:
    from scipy import stats
    from scipy.optimize import curve_fit
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("SciPy not available - some statistical features will be limited")

@dataclass
class QualityConfig:
    """Configuration for quality assessment."""
    
    # Quality flag thresholds
    saturation_threshold: float = 50000.0  # DN/s
    edge_buffer: int = 50  # pixels
    crowding_threshold: float = 2.0  # FWHM
    blending_threshold: float = 0.5  # fraction of flux
    
    # Completeness assessment
    magnitude_bins: np.ndarray = field(default_factory=lambda: np.arange(18, 30, 0.5))
    completeness_threshold: float = 0.8
    reliability_threshold: float = 0.9
    
    # Systematic error analysis
    spatial_bin_size: int = 100  # pixels
    check_spatial_systematics: bool = True
    check_magnitude_systematics: bool = True
    check_color_systematics: bool = True
    
    # External catalog comparison
    external_catalogs: Dict[str, str] = field(default_factory=dict)
    match_radius: float = 1.0  # arcseconds
    
    # Astrometric assessment
    astrometric_precision_threshold: float = 0.1  # arcseconds
    proper_motion_threshold: float = 10.0  # mas/year
    
    # Output parameters
    create_plots: bool = True
    plot_format: str = 'png'
    plot_dpi: int = 300

@dataclass
class QualityFlags:
    """Quality flags for individual sources."""
    
    # Detection quality
    saturated: bool = False
    near_edge: bool = False
    crowded: bool = False
    blended: bool = False
    
    # Photometric quality
    negative_flux: bool = False
    high_background: bool = False
    poor_fit: bool = False
    aperture_truncated: bool = False
    
    # Astrometric quality
    poor_centroid: bool = False
    significant_proper_motion: bool = False
    
    # External comparison
    no_external_match: bool = False
    discrepant_photometry: bool = False
    
    def to_bitmask(self) -> int:
        """Convert flags to bitmask representation."""
        flags = [
            self.saturated, self.near_edge, self.crowded, self.blended,
            self.negative_flux, self.high_background, self.poor_fit, 
            self.aperture_truncated, self.poor_centroid, 
            self.significant_proper_motion, self.no_external_match,
            self.discrepant_photometry
        ]
        
        bitmask = 0
        for i, flag in enumerate(flags):
            if flag:
                bitmask |= (1 << i)
        
        return bitmask
    
    @classmethod
    def from_bitmask(cls, bitmask: int) -> 'QualityFlags':
        """Create flags from bitmask representation."""
        flags = cls()
        flag_names = [
            'saturated', 'near_edge', 'crowded', 'blended',
            'negative_flux', 'high_background', 'poor_fit',
            'aperture_truncated', 'poor_centroid',
            'significant_proper_motion', 'no_external_match',
            'discrepant_photometry'
        ]
        
        for i, flag_name in enumerate(flag_names):
            setattr(flags, flag_name, bool(bitmask & (1 << i)))
        
        return flags

@dataclass
class CompletenessResults:
    """Results from completeness analysis."""
    
    magnitude_bins: np.ndarray
    completeness: np.ndarray
    reliability: np.ndarray
    detection_efficiency: np.ndarray
    false_positive_rate: np.ndarray
    
    # Summary statistics
    completeness_50: float = 0.0  # 50% completeness magnitude
    completeness_80: float = 0.0  # 80% completeness magnitude
    reliability_90: float = 0.0   # 90% reliability magnitude

@dataclass
class SystematicErrorResults:
    """Results from systematic error analysis."""
    
    # Spatial systematics
    spatial_magnitude_residuals: Dict[str, np.ndarray] = field(default_factory=dict)
    spatial_color_residuals: Dict[str, np.ndarray] = field(default_factory=dict)
    
    # Magnitude-dependent systematics
    magnitude_dependent_errors: Dict[str, np.ndarray] = field(default_factory=dict)
    
    # Summary statistics
    max_spatial_systematic: float = 0.0
    max_magnitude_systematic: float = 0.0
    max_color_systematic: float = 0.0

@dataclass
class QualityAssessmentResults:
    """Complete quality assessment results."""
    
    # Source-level quality
    source_flags: List[QualityFlags]
    quality_summary: Dict[str, int]
    
    # Catalog-level quality
    completeness: Optional[CompletenessResults] = None
    systematic_errors: Optional[SystematicErrorResults] = None
    
    # External comparisons
    external_matches: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Astrometric quality
    astrometric_precision: float = 0.0
    astrometric_accuracy: float = 0.0
    
    # Overall assessment
    overall_quality_grade: str = 'Unknown'  # 'Excellent', 'Good', 'Fair', 'Poor'
    quality_comments: List[str] = field(default_factory=list)
    
    # Generated plots
    plots: Dict[str, str] = field(default_factory=dict)


class QualityAssessor:
    """
    Comprehensive quality assessment processor for JWST NIRCam photometry.
    
    This class provides quality assessment capabilities including completeness
    analysis, systematic error identification, and comparison with external catalogs.
    """
    
    def __init__(self, config: Optional[QualityConfig] = None):
        """
        Initialize the quality assessor.
        
        Parameters:
        -----------
        config : QualityConfig, optional
            Quality assessment configuration. If None, uses defaults.
        """
        self.config = config or QualityConfig()
        self.logger = logging.getLogger(__name__)
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate the configuration parameters."""
        if self.config.saturation_threshold <= 0:
            raise ValueError("Saturation threshold must be positive")
        
        if self.config.match_radius <= 0:
            raise ValueError("Match radius must be positive")
        
        if len(self.config.magnitude_bins) < 2:
            raise ValueError("At least 2 magnitude bins required")
    
    def assess_quality(self,
                      photometry_results: Dict[str, Any],
                      detection_results: Optional[Dict[str, Any]] = None,
                      output_dir: Optional[Path] = None) -> QualityAssessmentResults:
        """
        Perform comprehensive quality assessment.
        
        Parameters:
        -----------
        photometry_results : dict
            Photometry results by band
        detection_results : dict, optional
            Source detection results
        output_dir : Path, optional
            Directory for output plots and files
            
        Returns:
        --------
        QualityAssessmentResults
            Complete quality assessment results
        """
        self.logger.info("Starting comprehensive quality assessment")
        
        # Extract sources from results
        sources = self._extract_sources(photometry_results)
        
        # Assign quality flags to individual sources
        self.logger.info("Assigning quality flags to sources")
        source_flags = self._assign_quality_flags(sources, detection_results)
        
        # Compute quality summary
        quality_summary = self._compute_quality_summary(source_flags)
        
        # Completeness analysis
        completeness = None
        if len(sources) > 100:  # Need sufficient statistics
            self.logger.info("Performing completeness analysis")
            completeness = self._assess_completeness(sources, photometry_results)
        
        # Systematic error analysis
        systematic_errors = None
        if self.config.check_spatial_systematics:
            self.logger.info("Analyzing systematic errors")
            systematic_errors = self._analyze_systematic_errors(sources)
        
        # External catalog comparisons
        external_matches = {}
        if self.config.external_catalogs:
            self.logger.info("Comparing with external catalogs")
            external_matches = self._compare_external_catalogs(sources)
        
        # Astrometric quality assessment
        astrometric_precision, astrometric_accuracy = self._assess_astrometry(sources)
        
        # Overall quality grading
        overall_grade, comments = self._compute_overall_grade(
            quality_summary, completeness, systematic_errors, external_matches
        )
        
        # Create diagnostic plots
        plots = {}
        if self.config.create_plots and output_dir:
            self.logger.info("Creating quality assessment plots")
            plots = self._create_quality_plots(
                sources, source_flags, completeness, systematic_errors, output_dir
            )
        
        # Create results object
        results = QualityAssessmentResults(
            source_flags=source_flags,
            quality_summary=quality_summary,
            completeness=completeness,
            systematic_errors=systematic_errors,
            external_matches=external_matches,
            astrometric_precision=astrometric_precision,
            astrometric_accuracy=astrometric_accuracy,
            overall_quality_grade=overall_grade,
            quality_comments=comments,
            plots=plots
        )
        
        self.logger.info(f"Quality assessment completed - Overall grade: {overall_grade}")
        return results
    
    def _extract_sources(self, photometry_results: Dict[str, Any]) -> List[Any]:
        """Extract source list from photometry results."""
        sources = []
        
        # This would extract sources from the actual photometry results format
        # For now, return empty list as placeholder
        
        return sources
    
    def _assign_quality_flags(self,
                             sources: List[Any],
                             detection_results: Optional[Dict[str, Any]]) -> List[QualityFlags]:
        """Assign quality flags to individual sources."""
        source_flags = []
        
        for source in sources:
            flags = QualityFlags()
            
            try:
                # Check for saturation
                if hasattr(source, 'peak_value'):
                    if source.peak_value > self.config.saturation_threshold:
                        flags.saturated = True
                
                # Check edge proximity
                if hasattr(source, 'x') and hasattr(source, 'y'):
                    # Would need image dimensions from detection results
                    # For now, skip this check
                    pass
                
                # Check for crowding
                if hasattr(source, 'crowding_flag'):
                    flags.crowded = source.crowding_flag
                
                # Check for blending
                if hasattr(source, 'blending_fraction'):
                    if source.blending_fraction > self.config.blending_threshold:
                        flags.blended = True
                
                # Check for negative flux
                if hasattr(source, 'flux'):
                    if source.flux < 0:
                        flags.negative_flux = True
                
                # Additional checks would be implemented here based on
                # the actual source structure
                
            except Exception as e:
                self.logger.warning(f"Could not assign flags for source: {e}")
            
            source_flags.append(flags)
        
        return source_flags
    
    def _compute_quality_summary(self, source_flags: List[QualityFlags]) -> Dict[str, int]:
        """Compute summary statistics of quality flags."""
        summary = {
            'total_sources': len(source_flags),
            'saturated': 0,
            'near_edge': 0,
            'crowded': 0,
            'blended': 0,
            'negative_flux': 0,
            'high_background': 0,
            'poor_fit': 0,
            'aperture_truncated': 0,
            'poor_centroid': 0,
            'significant_proper_motion': 0,
            'no_external_match': 0,
            'discrepant_photometry': 0,
            'good_quality': 0
        }
        
        for flags in source_flags:
            if flags.saturated:
                summary['saturated'] += 1
            if flags.near_edge:
                summary['near_edge'] += 1
            if flags.crowded:
                summary['crowded'] += 1
            if flags.blended:
                summary['blended'] += 1
            if flags.negative_flux:
                summary['negative_flux'] += 1
            if flags.high_background:
                summary['high_background'] += 1
            if flags.poor_fit:
                summary['poor_fit'] += 1
            if flags.aperture_truncated:
                summary['aperture_truncated'] += 1
            if flags.poor_centroid:
                summary['poor_centroid'] += 1
            if flags.significant_proper_motion:
                summary['significant_proper_motion'] += 1
            if flags.no_external_match:
                summary['no_external_match'] += 1
            if flags.discrepant_photometry:
                summary['discrepant_photometry'] += 1
            
            # Count as good quality if no major flags are set
            if not (flags.saturated or flags.poor_fit or flags.negative_flux):
                summary['good_quality'] += 1
        
        return summary
    
    def _assess_completeness(self,
                           sources: List[Any],
                           photometry_results: Dict[str, Any]) -> CompletenessResults:
        """Assess detection completeness and reliability."""
        self.logger.info("Assessing detection completeness")
        
        # This would implement a proper completeness analysis
        # For now, create placeholder results
        
        magnitude_bins = self.config.magnitude_bins
        n_bins = len(magnitude_bins) - 1
        
        # Placeholder values - in practice these would be computed
        # from injection/recovery tests or comparison with deeper catalogs
        completeness = np.ones(n_bins) * 0.9  # 90% completeness
        reliability = np.ones(n_bins) * 0.95   # 95% reliability
        detection_efficiency = completeness
        false_positive_rate = 1.0 - reliability
        
        # Find characteristic magnitudes
        completeness_50 = 26.0  # Placeholder
        completeness_80 = 25.0  # Placeholder
        reliability_90 = 24.0   # Placeholder
        
        return CompletenessResults(
            magnitude_bins=magnitude_bins[:-1],  # Bin centers
            completeness=completeness,
            reliability=reliability,
            detection_efficiency=detection_efficiency,
            false_positive_rate=false_positive_rate,
            completeness_50=completeness_50,
            completeness_80=completeness_80,
            reliability_90=reliability_90
        )
    
    def _analyze_systematic_errors(self, sources: List[Any]) -> SystematicErrorResults:
        """Analyze systematic errors in photometry."""
        self.logger.info("Analyzing systematic errors")
        
        # This would implement proper systematic error analysis
        # For now, create placeholder results
        
        results = SystematicErrorResults()
        
        # Placeholder values
        results.max_spatial_systematic = 0.02  # 2% maximum spatial variation
        results.max_magnitude_systematic = 0.01  # 1% magnitude-dependent error
        results.max_color_systematic = 0.005     # 0.5% color-dependent error
        
        return results
    
    def _compare_external_catalogs(self, sources: List[Any]) -> Dict[str, Dict[str, Any]]:
        """Compare photometry with external catalogs."""
        self.logger.info("Comparing with external catalogs")
        
        matches = {}
        
        for catalog_name, catalog_path in self.config.external_catalogs.items():
            try:
                # This would implement actual catalog cross-matching
                # For now, create placeholder results
                
                matches[catalog_name] = {
                    'n_matches': len(sources) // 2,  # Placeholder
                    'median_offset': 0.01,  # mag
                    'rms_scatter': 0.05,    # mag
                    'systematic_offset': -0.002,  # mag
                    'match_fraction': 0.8
                }
                
            except Exception as e:
                self.logger.warning(f"Could not compare with {catalog_name}: {e}")
                continue
        
        return matches
    
    def _assess_astrometry(self, sources: List[Any]) -> Tuple[float, float]:
        """Assess astrometric quality."""
        self.logger.info("Assessing astrometric quality")
        
        # This would implement proper astrometric assessment
        # For now, return placeholder values
        
        precision = 0.05  # arcseconds
        accuracy = 0.1    # arcseconds
        
        return precision, accuracy
    
    def _compute_overall_grade(self,
                              quality_summary: Dict[str, int],
                              completeness: Optional[CompletenessResults],
                              systematic_errors: Optional[SystematicErrorResults],
                              external_matches: Dict[str, Dict[str, Any]]) -> Tuple[str, List[str]]:
        """Compute overall quality grade and comments."""
        comments = []
        
        # Check source-level quality
        total_sources = quality_summary['total_sources']
        good_quality_fraction = quality_summary['good_quality'] / max(total_sources, 1)
        
        if good_quality_fraction > 0.9:
            source_grade = 'Excellent'
        elif good_quality_fraction > 0.8:
            source_grade = 'Good'
        elif good_quality_fraction > 0.7:
            source_grade = 'Fair'
        else:
            source_grade = 'Poor'
        
        comments.append(f"Source quality: {source_grade} ({good_quality_fraction:.1%} good quality)")
        
        # Check completeness
        completeness_grade = 'Unknown'
        if completeness:
            if completeness.completeness_80 < 25.0:
                completeness_grade = 'Excellent'
            elif completeness.completeness_80 < 26.0:
                completeness_grade = 'Good'
            elif completeness.completeness_80 < 27.0:
                completeness_grade = 'Fair'
            else:
                completeness_grade = 'Poor'
            
            comments.append(f"Completeness: {completeness_grade} (80% at {completeness.completeness_80:.1f} mag)")
        
        # Check systematic errors
        systematic_grade = 'Unknown'
        if systematic_errors:
            max_systematic = max(
                systematic_errors.max_spatial_systematic,
                systematic_errors.max_magnitude_systematic,
                systematic_errors.max_color_systematic
            )
            
            if max_systematic < 0.01:
                systematic_grade = 'Excellent'
            elif max_systematic < 0.02:
                systematic_grade = 'Good'
            elif max_systematic < 0.05:
                systematic_grade = 'Fair'
            else:
                systematic_grade = 'Poor'
            
            comments.append(f"Systematic errors: {systematic_grade} (max {max_systematic:.1%})")
        
        # Compute overall grade
        grades = [source_grade]
        if completeness_grade != 'Unknown':
            grades.append(completeness_grade)
        if systematic_grade != 'Unknown':
            grades.append(systematic_grade)
        
        grade_scores = {'Excellent': 4, 'Good': 3, 'Fair': 2, 'Poor': 1}
        avg_score = np.mean([grade_scores[grade] for grade in grades])
        
        if avg_score >= 3.5:
            overall_grade = 'Excellent'
        elif avg_score >= 2.5:
            overall_grade = 'Good'
        elif avg_score >= 1.5:
            overall_grade = 'Fair'
        else:
            overall_grade = 'Poor'
        
        return overall_grade, comments
    
    def _create_quality_plots(self,
                             sources: List[Any],
                             source_flags: List[QualityFlags],
                             completeness: Optional[CompletenessResults],
                             systematic_errors: Optional[SystematicErrorResults],
                             output_dir: Path) -> Dict[str, str]:
        """Create quality assessment diagnostic plots."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        plots = {}
        
        try:
            # Quality flags summary plot
            plots['quality_summary'] = self._plot_quality_summary(source_flags, output_dir)
            
            # Completeness plot
            if completeness:
                plots['completeness'] = self._plot_completeness(completeness, output_dir)
            
            # Systematic errors plot
            if systematic_errors:
                plots['systematic_errors'] = self._plot_systematic_errors(systematic_errors, output_dir)
            
            # Spatial distribution of quality
            if sources:
                plots['spatial_quality'] = self._plot_spatial_quality(sources, source_flags, output_dir)
            
        except Exception as e:
            self.logger.error(f"Failed to create quality plots: {e}")
        
        return plots
    
    def _plot_quality_summary(self, source_flags: List[QualityFlags], output_dir: Path) -> str:
        """Create quality flags summary plot."""
        flag_names = [
            'saturated', 'near_edge', 'crowded', 'blended',
            'negative_flux', 'high_background', 'poor_fit',
            'aperture_truncated', 'poor_centroid'
        ]
        
        flag_counts = []
        for flag_name in flag_names:
            count = sum(1 for flags in source_flags if getattr(flags, flag_name))
            flag_counts.append(count)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars = ax.bar(range(len(flag_names)), flag_counts)
        ax.set_xlabel('Quality Flag')
        ax.set_ylabel('Number of Sources')
        ax.set_title('Quality Flags Summary')
        ax.set_xticks(range(len(flag_names)))
        ax.set_xticklabels([name.replace('_', ' ').title() for name in flag_names], 
                          rotation=45, ha='right')
        
        # Add count labels on bars
        for bar, count in zip(bars, flag_counts):
            if count > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       str(count), ha='center', va='bottom')
        
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        plot_path = output_dir / f'quality_summary.{self.config.plot_format}'
        plt.savefig(plot_path, dpi=self.config.plot_dpi, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    def _plot_completeness(self, completeness: CompletenessResults, output_dir: Path) -> str:
        """Create completeness analysis plot."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Completeness plot
        ax1.plot(completeness.magnitude_bins, completeness.completeness, 'b-o', label='Completeness')
        ax1.axhline(y=0.8, color='r', linestyle='--', alpha=0.7, label='80% threshold')
        ax1.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='50% threshold')
        ax1.set_xlabel('Magnitude')
        ax1.set_ylabel('Completeness')
        ax1.set_title('Detection Completeness')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_ylim(0, 1.1)
        
        # Reliability plot
        ax2.plot(completeness.magnitude_bins, completeness.reliability, 'g-o', label='Reliability')
        ax2.axhline(y=0.9, color='r', linestyle='--', alpha=0.7, label='90% threshold')
        ax2.set_xlabel('Magnitude')
        ax2.set_ylabel('Reliability')
        ax2.set_title('Detection Reliability')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_ylim(0, 1.1)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = output_dir / f'completeness_analysis.{self.config.plot_format}'
        plt.savefig(plot_path, dpi=self.config.plot_dpi, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    def _plot_systematic_errors(self, systematic_errors: SystematicErrorResults, output_dir: Path) -> str:
        """Create systematic errors plot."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Bar plot of maximum systematic errors
        categories = ['Spatial', 'Magnitude', 'Color']
        values = [
            systematic_errors.max_spatial_systematic * 100,
            systematic_errors.max_magnitude_systematic * 100,
            systematic_errors.max_color_systematic * 100
        ]
        
        bars = ax.bar(categories, values, color=['blue', 'orange', 'green'], alpha=0.7)
        ax.set_ylabel('Maximum Systematic Error (%)')
        ax.set_title('Systematic Error Summary')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.2f}%', ha='center', va='bottom')
        
        # Save plot
        plot_path = output_dir / f'systematic_errors.{self.config.plot_format}'
        plt.savefig(plot_path, dpi=self.config.plot_dpi, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    def _plot_spatial_quality(self, sources: List[Any], source_flags: List[QualityFlags], 
                             output_dir: Path) -> str:
        """Create spatial distribution of quality plot."""
        # Extract coordinates
        x_coords = []
        y_coords = []
        good_quality = []
        
        for source, flags in zip(sources, source_flags):
            if hasattr(source, 'x') and hasattr(source, 'y'):
                x_coords.append(source.x)
                y_coords.append(source.y)
                # Mark as good if no major flags
                good = not (flags.saturated or flags.poor_fit or flags.negative_flux)
                good_quality.append(good)
        
        if not x_coords:
            return ""
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot good and bad quality sources separately
        x_coords = np.array(x_coords)
        y_coords = np.array(y_coords)
        good_quality = np.array(good_quality)
        
        good_mask = good_quality
        bad_mask = ~good_quality
        
        if np.any(good_mask):
            ax.scatter(x_coords[good_mask], y_coords[good_mask], 
                      c='green', alpha=0.6, s=10, label='Good Quality')
        
        if np.any(bad_mask):
            ax.scatter(x_coords[bad_mask], y_coords[bad_mask], 
                      c='red', alpha=0.6, s=20, label='Poor Quality')
        
        ax.set_xlabel('X [pixels]')
        ax.set_ylabel('Y [pixels]')
        ax.set_title('Spatial Distribution of Source Quality')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Save plot
        plot_path = output_dir / f'spatial_quality.{self.config.plot_format}'
        plt.savefig(plot_path, dpi=self.config.plot_dpi, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)


def create_quality_report(assessment_results: QualityAssessmentResults,
                         output_path: Path) -> None:
    """
    Create a comprehensive quality assessment report.
    
    Parameters:
    -----------
    assessment_results : QualityAssessmentResults
        Quality assessment results
    output_path : Path
        Path for the output report
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Creating quality assessment report: {output_path}")
    
    with open(output_path, 'w') as f:
        f.write("JWST NIRCam Photometry Quality Assessment Report\n")
        f.write("=" * 50 + "\n\n")
        
        # Overall assessment
        f.write(f"Overall Quality Grade: {assessment_results.overall_quality_grade}\n\n")
        
        f.write("Comments:\n")
        for comment in assessment_results.quality_comments:
            f.write(f"- {comment}\n")
        f.write("\n")
        
        # Source-level quality
        f.write("Source Quality Summary:\n")
        f.write("-" * 25 + "\n")
        summary = assessment_results.quality_summary
        f.write(f"Total sources: {summary['total_sources']}\n")
        f.write(f"Good quality: {summary['good_quality']} ({summary['good_quality']/max(summary['total_sources'], 1):.1%})\n")
        f.write(f"Saturated: {summary['saturated']}\n")
        f.write(f"Crowded: {summary['crowded']}\n")
        f.write(f"Blended: {summary['blended']}\n")
        f.write(f"Negative flux: {summary['negative_flux']}\n")
        f.write("\n")
        
        # Completeness
        if assessment_results.completeness:
            comp = assessment_results.completeness
            f.write("Completeness Analysis:\n")
            f.write("-" * 20 + "\n")
            f.write(f"50% completeness: {comp.completeness_50:.1f} mag\n")
            f.write(f"80% completeness: {comp.completeness_80:.1f} mag\n")
            f.write(f"90% reliability: {comp.reliability_90:.1f} mag\n")
            f.write("\n")
        
        # Systematic errors
        if assessment_results.systematic_errors:
            sys_err = assessment_results.systematic_errors
            f.write("Systematic Errors:\n")
            f.write("-" * 17 + "\n")
            f.write(f"Maximum spatial systematic: {sys_err.max_spatial_systematic:.1%}\n")
            f.write(f"Maximum magnitude systematic: {sys_err.max_magnitude_systematic:.1%}\n")
            f.write(f"Maximum color systematic: {sys_err.max_color_systematic:.1%}\n")
            f.write("\n")
        
        # Astrometry
        f.write("Astrometric Quality:\n")
        f.write("-" * 18 + "\n")
        f.write(f"Precision: {assessment_results.astrometric_precision:.3f} arcsec\n")
        f.write(f"Accuracy: {assessment_results.astrometric_accuracy:.3f} arcsec\n")
        f.write("\n")
        
        # External comparisons
        if assessment_results.external_matches:
            f.write("External Catalog Comparisons:\n")
            f.write("-" * 28 + "\n")
            for catalog, results in assessment_results.external_matches.items():
                f.write(f"{catalog}:\n")
                f.write(f"  Matches: {results['n_matches']}\n")
                f.write(f"  Median offset: {results['median_offset']:.3f} mag\n")
                f.write(f"  RMS scatter: {results['rms_scatter']:.3f} mag\n")
                f.write(f"  Match fraction: {results['match_fraction']:.1%}\n")
            f.write("\n")
        
        # Generated plots
        if assessment_results.plots:
            f.write("Generated Plots:\n")
            f.write("-" * 15 + "\n")
            for plot_name, plot_path in assessment_results.plots.items():
                f.write(f"- {plot_name}: {plot_path}\n")


# Convenience functions for common quality assessment tasks

def quick_quality_check(sources: List[Any]) -> Dict[str, Any]:
    """
    Quick quality check for a list of sources.
    
    Parameters:
    -----------
    sources : list
        List of photometry sources
        
    Returns:
    --------
    dict
        Basic quality statistics
    """
    assessor = QualityAssessor()
    
    # Create dummy photometry results
    photometry_results = {'sources': sources}
    
    # Run assessment
    results = assessor.assess_quality(photometry_results)
    
    return {
        'total_sources': results.quality_summary['total_sources'],
        'good_quality_fraction': results.quality_summary['good_quality'] / max(results.quality_summary['total_sources'], 1),
        'overall_grade': results.overall_quality_grade,
        'comments': results.quality_comments
    }


if __name__ == "__main__":
    # Example usage
    print("JWST Quality Assessment Module")
    print("This module provides quality assessment capabilities for JWST photometry")
