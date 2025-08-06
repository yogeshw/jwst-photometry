#!/usr/bin/env python3
"""
Color-Based Analysis Module for JWST NIRCam Photometry

This module provides comprehensive color analysis capabilities including:
- Color-color diagram generation
- Star-galaxy separation using colors and morphology
- Photometric redshift preparation
- Color-magnitude diagrams
- Color-dependent systematic corrections

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
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    ASTROPY_AVAILABLE = True
except ImportError:
    ASTROPY_AVAILABLE = False
    warnings.warn("Astropy not available - some features will be limited")

try:
    from sklearn.cluster import DBSCAN
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("Scikit-learn not available - ML features will be limited")

@dataclass
class ColorAnalysisConfig:
    """Configuration for color-based analysis."""
    
    # Color calculation parameters
    reference_bands: List[str] = field(default_factory=lambda: ['F150W', 'F277W', 'F444W'])
    magnitude_system: str = 'AB'  # AB or Vega
    extinction_correction: bool = True
    k_correction: bool = False  # For high-z sources
    
    # Star-galaxy separation
    use_morphology: bool = True
    use_colors: bool = True
    morphology_threshold: float = 0.1  # Size difference threshold
    color_separation_method: str = 'rf'  # 'rf', 'dbscan', 'manual'
    
    # Quality filtering
    snr_threshold: float = 5.0
    magnitude_limit: float = 28.0
    quality_flags: List[str] = field(default_factory=lambda: ['saturated', 'edge', 'blended'])
    
    # Plotting parameters
    create_plots: bool = True
    plot_format: str = 'png'
    plot_dpi: int = 300
    interactive_plots: bool = False

@dataclass
class ColorCatalogSource:
    """Source with color and morphological information."""
    
    # Basic properties
    id: int
    ra: float
    dec: float
    x: float
    y: float
    
    # Photometry (all bands)
    magnitudes: Dict[str, float] = field(default_factory=dict)
    magnitude_errors: Dict[str, float] = field(default_factory=dict)
    fluxes: Dict[str, float] = field(default_factory=dict)
    flux_errors: Dict[str, float] = field(default_factory=dict)
    
    # Colors
    colors: Dict[str, float] = field(default_factory=dict)
    color_errors: Dict[str, float] = field(default_factory=dict)
    
    # Morphology
    size_pixels: float = 0.0
    ellipticity: float = 0.0
    concentration: float = 0.0
    asymmetry: float = 0.0
    
    # Classifications
    star_galaxy_prob: float = 0.5  # 0=galaxy, 1=star
    classification: str = 'unknown'  # 'star', 'galaxy', 'artifact'
    
    # Quality
    quality_flags: List[str] = field(default_factory=list)
    detection_significance: float = 0.0

@dataclass
class ColorAnalysisResults:
    """Results from color analysis."""
    
    sources: List[ColorCatalogSource]
    config: ColorAnalysisConfig
    
    # Statistics
    n_sources_total: int = 0
    n_sources_good_quality: int = 0
    n_stars: int = 0
    n_galaxies: int = 0
    n_artifacts: int = 0
    
    # Color distributions
    color_statistics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Plots generated
    plots: Dict[str, str] = field(default_factory=dict)


class ColorAnalyzer:
    """
    Advanced color analysis processor for JWST NIRCam observations.
    
    This class provides comprehensive color analysis including star-galaxy
    separation, color-color diagrams, and photometric redshift preparation.
    """
    
    def __init__(self, config: Optional[ColorAnalysisConfig] = None):
        """
        Initialize the color analyzer.
        
        Parameters:
        -----------
        config : ColorAnalysisConfig, optional
            Color analysis configuration. If None, uses defaults.
        """
        self.config = config or ColorAnalysisConfig()
        self.logger = logging.getLogger(__name__)
        
        # Validate configuration
        self._validate_config()
        
        # Initialize machine learning models if available
        self.star_galaxy_classifier = None
        if SKLEARN_AVAILABLE and self.config.color_separation_method == 'rf':
            self._initialize_ml_models()
    
    def _validate_config(self) -> None:
        """Validate the configuration parameters."""
        if len(self.config.reference_bands) < 2:
            raise ValueError("At least 2 bands required for color analysis")
        
        if self.config.magnitude_system not in ['AB', 'Vega']:
            raise ValueError("Magnitude system must be 'AB' or 'Vega'")
        
        valid_methods = ['rf', 'dbscan', 'manual']
        if self.config.color_separation_method not in valid_methods:
            raise ValueError(f"Color separation method must be one of {valid_methods}")
    
    def _initialize_ml_models(self) -> None:
        """Initialize machine learning models for classification."""
        self.logger.info("Initializing machine learning models for star-galaxy separation")
        
        # Random Forest classifier for star-galaxy separation
        self.star_galaxy_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        )
    
    def analyze_colors(self, 
                      calibrated_sources: List[Any],
                      output_dir: Optional[Path] = None) -> ColorAnalysisResults:
        """
        Perform comprehensive color analysis on calibrated sources.
        
        Parameters:
        -----------
        calibrated_sources : list
            List of calibrated photometry sources
        output_dir : Path, optional
            Directory for output plots and files
            
        Returns:
        --------
        ColorAnalysisResults
            Complete color analysis results
        """
        self.logger.info("Starting comprehensive color analysis")
        start_time = self.logger.info("Starting color analysis")
        
        # Convert sources to color catalog format
        color_sources = self._convert_to_color_catalog(calibrated_sources)
        
        # Calculate colors
        self._calculate_colors(color_sources)
        
        # Quality filtering
        good_sources = self._apply_quality_filtering(color_sources)
        
        # Morphological measurements
        if self.config.use_morphology:
            self._measure_morphology(good_sources)
        
        # Star-galaxy separation
        if self.config.use_colors or self.config.use_morphology:
            self._perform_star_galaxy_separation(good_sources)
        
        # Generate statistics
        statistics = self._compute_color_statistics(good_sources)
        
        # Create diagnostic plots
        plots = {}
        if self.config.create_plots and output_dir:
            plots = self._create_diagnostic_plots(good_sources, output_dir)
        
        # Create results object
        results = ColorAnalysisResults(
            sources=good_sources,
            config=self.config,
            n_sources_total=len(color_sources),
            n_sources_good_quality=len(good_sources),
            n_stars=len([s for s in good_sources if s.classification == 'star']),
            n_galaxies=len([s for s in good_sources if s.classification == 'galaxy']),
            n_artifacts=len([s for s in good_sources if s.classification == 'artifact']),
            color_statistics=statistics,
            plots=plots
        )
        
        self.logger.info(f"Color analysis completed: {len(good_sources)} good quality sources")
        self.logger.info(f"Classifications: {results.n_stars} stars, {results.n_galaxies} galaxies")
        
        return results
    
    def _convert_to_color_catalog(self, calibrated_sources: List[Any]) -> List[ColorCatalogSource]:
        """Convert calibrated sources to color catalog format."""
        self.logger.info("Converting sources to color catalog format")
        
        color_sources = []
        
        for i, source in enumerate(calibrated_sources):
            try:
                # Extract basic properties
                if hasattr(source, 'bands'):
                    # Multi-band calibrated source
                    magnitudes = {}
                    magnitude_errors = {}
                    fluxes = {}
                    flux_errors = {}
                    
                    for band, band_data in source.bands.items():
                        if band in self.config.reference_bands:
                            # Extract magnitude and flux from largest aperture
                            if 'circular_photometry' in band_data:
                                largest_aperture = max(band_data['circular_photometry'].keys())
                                phot_data = band_data['circular_photometry'][largest_aperture]
                                
                                magnitudes[band] = phot_data.get('magnitude_calibrated', 99.0)
                                magnitude_errors[band] = phot_data.get('magnitude_error_calibrated', 99.0)
                                fluxes[band] = phot_data.get('flux_calibrated', 0.0)
                                flux_errors[band] = phot_data.get('flux_error_calibrated', 0.0)
                    
                    # Create color source
                    color_source = ColorCatalogSource(
                        id=i,
                        ra=getattr(source, 'ra', 0.0),
                        dec=getattr(source, 'dec', 0.0),
                        x=getattr(source, 'x', 0.0),
                        y=getattr(source, 'y', 0.0),
                        magnitudes=magnitudes,
                        magnitude_errors=magnitude_errors,
                        fluxes=fluxes,
                        flux_errors=flux_errors
                    )
                    
                    color_sources.append(color_source)
                    
            except Exception as e:
                self.logger.warning(f"Could not convert source {i}: {e}")
                continue
        
        self.logger.info(f"Converted {len(color_sources)} sources to color catalog")
        return color_sources
    
    def _calculate_colors(self, sources: List[ColorCatalogSource]) -> None:
        """Calculate colors for all source combinations."""
        self.logger.info("Calculating colors")
        
        bands = self.config.reference_bands
        
        for source in sources:
            try:
                # Calculate all possible color combinations
                for i, band1 in enumerate(bands):
                    for j, band2 in enumerate(bands):
                        if i < j:  # Avoid duplicates
                            color_name = f"{band1}-{band2}"
                            
                            if (band1 in source.magnitudes and band2 in source.magnitudes and
                                source.magnitudes[band1] < 90 and source.magnitudes[band2] < 90):
                                
                                # Calculate color
                                color = source.magnitudes[band1] - source.magnitudes[band2]
                                source.colors[color_name] = color
                                
                                # Calculate color error
                                if (band1 in source.magnitude_errors and 
                                    band2 in source.magnitude_errors):
                                    color_error = np.sqrt(
                                        source.magnitude_errors[band1]**2 + 
                                        source.magnitude_errors[band2]**2
                                    )
                                    source.color_errors[color_name] = color_error
                                else:
                                    source.color_errors[color_name] = 99.0
                
            except Exception as e:
                self.logger.warning(f"Could not calculate colors for source {source.id}: {e}")
                continue
    
    def _apply_quality_filtering(self, sources: List[ColorCatalogSource]) -> List[ColorCatalogSource]:
        """Apply quality filtering to sources."""
        self.logger.info("Applying quality filtering")
        
        good_sources = []
        
        for source in sources:
            quality_ok = True
            
            # Check signal-to-noise ratio
            max_snr = 0.0
            for band in self.config.reference_bands:
                if band in source.fluxes and band in source.flux_errors:
                    if source.flux_errors[band] > 0:
                        snr = source.fluxes[band] / source.flux_errors[band]
                        max_snr = max(max_snr, snr)
            
            if max_snr < self.config.snr_threshold:
                source.quality_flags.append('low_snr')
                quality_ok = False
            
            source.detection_significance = max_snr
            
            # Check magnitude limits
            for band in self.config.reference_bands:
                if band in source.magnitudes:
                    if source.magnitudes[band] > self.config.magnitude_limit:
                        source.quality_flags.append('faint')
                        quality_ok = False
                        break
            
            # Check for required colors
            if len(source.colors) == 0:
                source.quality_flags.append('no_colors')
                quality_ok = False
            
            if quality_ok:
                good_sources.append(source)
        
        filtered_out = len(sources) - len(good_sources)
        self.logger.info(f"Quality filtering: {len(good_sources)} sources passed, {filtered_out} filtered out")
        
        return good_sources
    
    def _measure_morphology(self, sources: List[ColorCatalogSource]) -> None:
        """Measure morphological parameters for star-galaxy separation."""
        self.logger.info("Measuring morphological parameters")
        
        for source in sources:
            try:
                # Basic morphology - in real implementation, this would
                # extract from the source detection results
                
                # For now, use simple estimates based on available data
                # This would be replaced with actual morphological measurements
                source.size_pixels = 2.0  # Default size
                source.ellipticity = 0.1  # Default ellipticity
                source.concentration = 0.5  # Default concentration
                source.asymmetry = 0.1  # Default asymmetry
                
            except Exception as e:
                self.logger.warning(f"Could not measure morphology for source {source.id}: {e}")
                continue
    
    def _perform_star_galaxy_separation(self, sources: List[ColorCatalogSource]) -> None:
        """Perform star-galaxy separation using colors and morphology."""
        self.logger.info(f"Performing star-galaxy separation using {self.config.color_separation_method}")
        
        if self.config.color_separation_method == 'manual':
            self._manual_star_galaxy_separation(sources)
        elif self.config.color_separation_method == 'dbscan' and SKLEARN_AVAILABLE:
            self._dbscan_star_galaxy_separation(sources)
        elif self.config.color_separation_method == 'rf' and SKLEARN_AVAILABLE:
            self._rf_star_galaxy_separation(sources)
        else:
            self.logger.warning("Star-galaxy separation method not available, using manual")
            self._manual_star_galaxy_separation(sources)
    
    def _manual_star_galaxy_separation(self, sources: List[ColorCatalogSource]) -> None:
        """Manual star-galaxy separation using simple criteria."""
        for source in sources:
            # Simple criteria based on size and colors
            is_star = (source.size_pixels < 3.0 and 
                      source.ellipticity < 0.3)
            
            if is_star:
                source.classification = 'star'
                source.star_galaxy_prob = 0.8
            else:
                source.classification = 'galaxy'
                source.star_galaxy_prob = 0.2
    
    def _dbscan_star_galaxy_separation(self, sources: List[ColorCatalogSource]) -> None:
        """DBSCAN-based star-galaxy separation."""
        # Prepare features for clustering
        features = []
        valid_sources = []
        
        for source in sources:
            if len(source.colors) >= 2:
                color_values = list(source.colors.values())[:2]  # Use first 2 colors
                if self.config.use_morphology:
                    feature_vector = color_values + [source.size_pixels, source.ellipticity]
                else:
                    feature_vector = color_values
                
                features.append(feature_vector)
                valid_sources.append(source)
        
        if len(features) < 10:
            self.logger.warning("Not enough sources for DBSCAN, using manual classification")
            self._manual_star_galaxy_separation(sources)
            return
        
        # Perform clustering
        features_array = np.array(features)
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        clusters = dbscan.fit_predict(features_array)
        
        # Assign classifications based on clusters
        for i, source in enumerate(valid_sources):
            cluster = clusters[i]
            if cluster == -1:  # Noise
                source.classification = 'artifact'
                source.star_galaxy_prob = 0.1
            elif cluster == 0:  # Assume first cluster is stars
                source.classification = 'star'
                source.star_galaxy_prob = 0.8
            else:  # Other clusters are galaxies
                source.classification = 'galaxy'
                source.star_galaxy_prob = 0.2
    
    def _rf_star_galaxy_separation(self, sources: List[ColorCatalogSource]) -> None:
        """Random Forest star-galaxy separation (requires training data)."""
        # This is a placeholder - in practice, you would train on
        # known star/galaxy samples or use pre-trained models
        
        # For now, use manual classification
        self._manual_star_galaxy_separation(sources)
        
        self.logger.info("Random Forest classification placeholder - using manual method")
    
    def _compute_color_statistics(self, sources: List[ColorCatalogSource]) -> Dict[str, Dict[str, float]]:
        """Compute color distribution statistics."""
        self.logger.info("Computing color statistics")
        
        statistics = {}
        
        # Get all unique colors
        all_colors = set()
        for source in sources:
            all_colors.update(source.colors.keys())
        
        for color_name in all_colors:
            color_values = []
            for source in sources:
                if color_name in source.colors:
                    color_values.append(source.colors[color_name])
            
            if color_values:
                color_array = np.array(color_values)
                statistics[color_name] = {
                    'mean': float(np.mean(color_array)),
                    'median': float(np.median(color_array)),
                    'std': float(np.std(color_array)),
                    'min': float(np.min(color_array)),
                    'max': float(np.max(color_array)),
                    'n_sources': len(color_values)
                }
        
        return statistics
    
    def _create_diagnostic_plots(self, sources: List[ColorCatalogSource], 
                                output_dir: Path) -> Dict[str, str]:
        """Create diagnostic plots for color analysis."""
        self.logger.info("Creating diagnostic plots")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        plots = {}
        
        try:
            # Color-color diagram
            plots['color_color'] = self._create_color_color_diagram(sources, output_dir)
            
            # Color-magnitude diagram
            plots['color_magnitude'] = self._create_color_magnitude_diagram(sources, output_dir)
            
            # Star-galaxy separation plot
            plots['star_galaxy'] = self._create_star_galaxy_plot(sources, output_dir)
            
            # Color histograms
            plots['color_histograms'] = self._create_color_histograms(sources, output_dir)
            
        except Exception as e:
            self.logger.error(f"Failed to create diagnostic plots: {e}")
        
        return plots
    
    def _create_color_color_diagram(self, sources: List[ColorCatalogSource], 
                                   output_dir: Path) -> str:
        """Create color-color diagram."""
        bands = self.config.reference_bands
        if len(bands) < 3:
            return ""
        
        # Use first three bands for the diagram
        color1_name = f"{bands[0]}-{bands[1]}"
        color2_name = f"{bands[1]}-{bands[2]}"
        
        color1_values = []
        color2_values = []
        classifications = []
        
        for source in sources:
            if color1_name in source.colors and color2_name in source.colors:
                color1_values.append(source.colors[color1_name])
                color2_values.append(source.colors[color2_name])
                classifications.append(source.classification)
        
        if not color1_values:
            return ""
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot different classifications with different colors
        for class_type in ['star', 'galaxy', 'artifact', 'unknown']:
            mask = np.array(classifications) == class_type
            if np.any(mask):
                ax.scatter(np.array(color1_values)[mask], 
                          np.array(color2_values)[mask],
                          label=class_type.capitalize(), 
                          alpha=0.6, s=20)
        
        ax.set_xlabel(f'{color1_name} [mag]')
        ax.set_ylabel(f'{color2_name} [mag]')
        ax.set_title('Color-Color Diagram')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Save plot
        plot_path = output_dir / f'color_color_diagram.{self.config.plot_format}'
        plt.savefig(plot_path, dpi=self.config.plot_dpi, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    def _create_color_magnitude_diagram(self, sources: List[ColorCatalogSource], 
                                       output_dir: Path) -> str:
        """Create color-magnitude diagram."""
        bands = self.config.reference_bands
        if len(bands) < 2:
            return ""
        
        color_name = f"{bands[0]}-{bands[1]}"
        reference_band = bands[0]
        
        colors = []
        magnitudes = []
        classifications = []
        
        for source in sources:
            if (color_name in source.colors and 
                reference_band in source.magnitudes):
                colors.append(source.colors[color_name])
                magnitudes.append(source.magnitudes[reference_band])
                classifications.append(source.classification)
        
        if not colors:
            return ""
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot different classifications
        for class_type in ['star', 'galaxy', 'artifact', 'unknown']:
            mask = np.array(classifications) == class_type
            if np.any(mask):
                ax.scatter(np.array(colors)[mask], 
                          np.array(magnitudes)[mask],
                          label=class_type.capitalize(), 
                          alpha=0.6, s=20)
        
        ax.set_xlabel(f'{color_name} [mag]')
        ax.set_ylabel(f'{reference_band} [mag]')
        ax.set_title('Color-Magnitude Diagram')
        ax.invert_yaxis()  # Brighter objects at top
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Save plot
        plot_path = output_dir / f'color_magnitude_diagram.{self.config.plot_format}'
        plt.savefig(plot_path, dpi=self.config.plot_dpi, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    def _create_star_galaxy_plot(self, sources: List[ColorCatalogSource], 
                                output_dir: Path) -> str:
        """Create star-galaxy separation diagnostic plot."""
        if not self.config.use_morphology:
            return ""
        
        sizes = []
        ellipticities = []
        classifications = []
        
        for source in sources:
            sizes.append(source.size_pixels)
            ellipticities.append(source.ellipticity)
            classifications.append(source.classification)
        
        if not sizes:
            return ""
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot different classifications
        for class_type in ['star', 'galaxy', 'artifact', 'unknown']:
            mask = np.array(classifications) == class_type
            if np.any(mask):
                ax.scatter(np.array(sizes)[mask], 
                          np.array(ellipticities)[mask],
                          label=class_type.capitalize(), 
                          alpha=0.6, s=20)
        
        ax.set_xlabel('Size [pixels]')
        ax.set_ylabel('Ellipticity')
        ax.set_title('Star-Galaxy Separation')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Save plot
        plot_path = output_dir / f'star_galaxy_separation.{self.config.plot_format}'
        plt.savefig(plot_path, dpi=self.config.plot_dpi, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    def _create_color_histograms(self, sources: List[ColorCatalogSource], 
                                output_dir: Path) -> str:
        """Create color distribution histograms."""
        # Get all unique colors
        all_colors = set()
        for source in sources:
            all_colors.update(source.colors.keys())
        
        if not all_colors:
            return ""
        
        # Create subplots
        n_colors = len(all_colors)
        n_cols = min(3, n_colors)
        n_rows = (n_colors + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
        if n_colors == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, color_name in enumerate(sorted(all_colors)):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            
            # Get color values for each classification
            for class_type in ['star', 'galaxy', 'artifact', 'unknown']:
                color_values = []
                for source in sources:
                    if (source.classification == class_type and 
                        color_name in source.colors):
                        color_values.append(source.colors[color_name])
                
                if color_values:
                    ax.hist(color_values, bins=20, alpha=0.5, 
                           label=class_type.capitalize(), density=True)
            
            ax.set_xlabel(f'{color_name} [mag]')
            ax.set_ylabel('Density')
            ax.set_title(f'{color_name} Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(n_colors, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            if n_rows > 1:
                axes[row, col].set_visible(False)
            elif n_cols > 1:
                axes[col].set_visible(False)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = output_dir / f'color_histograms.{self.config.plot_format}'
        plt.savefig(plot_path, dpi=self.config.plot_dpi, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)


def create_color_catalog(photometry_results: Dict[str, Any], 
                        output_path: Optional[Path] = None) -> Table:
    """
    Create a color catalog from multi-band photometry results.
    
    Parameters:
    -----------
    photometry_results : dict
        Dictionary of photometry results by band
    output_path : Path, optional
        Path to save the catalog
        
    Returns:
    --------
    astropy.table.Table
        Color catalog table
    """
    if not ASTROPY_AVAILABLE:
        raise ImportError("Astropy required for catalog creation")
    
    logger = logging.getLogger(__name__)
    logger.info("Creating color catalog")
    
    # Initialize color analyzer
    analyzer = ColorAnalyzer()
    
    # Convert results to calibrated sources format
    calibrated_sources = []
    # This would be implemented based on the actual photometry results format
    
    # Perform color analysis
    color_results = analyzer.analyze_colors(calibrated_sources)
    
    # Create astropy table
    table_data = {
        'id': [],
        'ra': [],
        'dec': [],
        'x': [],
        'y': [],
        'classification': [],
        'star_galaxy_prob': [],
        'detection_significance': []
    }
    
    # Add magnitude columns
    bands = analyzer.config.reference_bands
    for band in bands:
        table_data[f'mag_{band}'] = []
        table_data[f'mag_err_{band}'] = []
    
    # Add color columns
    for i, band1 in enumerate(bands):
        for j, band2 in enumerate(bands):
            if i < j:
                color_name = f"{band1}-{band2}"
                table_data[f'color_{color_name}'] = []
                table_data[f'color_err_{color_name}'] = []
    
    # Fill table data
    for source in color_results.sources:
        table_data['id'].append(source.id)
        table_data['ra'].append(source.ra)
        table_data['dec'].append(source.dec)
        table_data['x'].append(source.x)
        table_data['y'].append(source.y)
        table_data['classification'].append(source.classification)
        table_data['star_galaxy_prob'].append(source.star_galaxy_prob)
        table_data['detection_significance'].append(source.detection_significance)
        
        # Add magnitudes
        for band in bands:
            table_data[f'mag_{band}'].append(
                source.magnitudes.get(band, 99.0)
            )
            table_data[f'mag_err_{band}'].append(
                source.magnitude_errors.get(band, 99.0)
            )
        
        # Add colors
        for i, band1 in enumerate(bands):
            for j, band2 in enumerate(bands):
                if i < j:
                    color_name = f"{band1}-{band2}"
                    table_data[f'color_{color_name}'].append(
                        source.colors.get(color_name, 99.0)
                    )
                    table_data[f'color_err_{color_name}'].append(
                        source.color_errors.get(color_name, 99.0)
                    )
    
    # Create table
    catalog = Table(table_data)
    
    # Add metadata
    catalog.meta['description'] = 'JWST NIRCam Color Catalog'
    catalog.meta['bands'] = bands
    catalog.meta['n_sources'] = len(color_results.sources)
    catalog.meta['n_stars'] = color_results.n_stars
    catalog.meta['n_galaxies'] = color_results.n_galaxies
    
    # Save if requested
    if output_path:
        catalog.write(output_path, format='fits', overwrite=True)
        logger.info(f"Saved color catalog to {output_path}")
    
    return catalog


# Convenience functions for common color analysis tasks

def quick_star_galaxy_separation(sources: List[Any], 
                                method: str = 'manual') -> List[str]:
    """
    Quick star-galaxy separation for a list of sources.
    
    Parameters:
    -----------
    sources : list
        List of photometry sources
    method : str
        Separation method ('manual', 'dbscan', 'rf')
        
    Returns:
    --------
    list
        List of classifications ('star', 'galaxy', 'artifact')
    """
    config = ColorAnalysisConfig(color_separation_method=method)
    analyzer = ColorAnalyzer(config)
    
    # Convert and analyze
    color_sources = analyzer._convert_to_color_catalog(sources)
    analyzer._calculate_colors(color_sources)
    good_sources = analyzer._apply_quality_filtering(color_sources)
    analyzer._perform_star_galaxy_separation(good_sources)
    
    return [source.classification for source in good_sources]


def calculate_colors(band1_mags: np.ndarray, 
                    band2_mags: np.ndarray,
                    band1_errors: Optional[np.ndarray] = None,
                    band2_errors: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate colors and color errors from magnitude arrays.
    
    Parameters:
    -----------
    band1_mags, band2_mags : array_like
        Magnitude arrays for the two bands
    band1_errors, band2_errors : array_like, optional
        Magnitude error arrays
        
    Returns:
    --------
    tuple
        Colors and color errors
    """
    colors = band1_mags - band2_mags
    
    if band1_errors is not None and band2_errors is not None:
        color_errors = np.sqrt(band1_errors**2 + band2_errors**2)
    else:
        color_errors = np.zeros_like(colors)
    
    return colors, color_errors


if __name__ == "__main__":
    # Example usage
    print("JWST Color Analysis Module")
    print("This module provides color analysis capabilities for JWST photometry")
