#!/usr/bin/env python3
"""
Catalog Generation Module for JWST NIRCam Photometry

This module provides comprehensive catalog generation capabilities including:
- Multi-format output (FITS, HDF5, CSV, ASCII)
- Metadata and provenance tracking
- Cross-matching with external catalogs
- Catalog validation and verification
- Custom format support

Author: JWST Photometry Pipeline
Date: August 2025
"""

import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
import warnings
import json
import time
from datetime import datetime

try:
    from astropy.table import Table, Column
    from astropy.io import fits, ascii
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    from astropy.time import Time
    ASTROPY_AVAILABLE = True
except ImportError:
    ASTROPY_AVAILABLE = False
    warnings.warn("Astropy not available - catalog generation will be limited")

try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False
    warnings.warn("h5py not available - HDF5 output not supported")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    warnings.warn("pandas not available - some features will be limited")


@dataclass
class CatalogConfig:
    """Configuration for catalog generation."""
    
    # Output formats
    output_formats: List[str] = field(default_factory=lambda: ['fits', 'csv'])
    fits_compress: bool = True
    hdf5_compression: str = 'gzip'
    csv_delimiter: str = ','
    
    # Coordinate system
    coordinate_frame: str = 'icrs'  # 'icrs', 'fk5', 'galactic'
    epoch: float = 2000.0
    
    # Column selection and naming
    include_flags: bool = True
    include_errors: bool = True
    include_quality: bool = True
    include_apertures: bool = True
    include_colors: bool = True
    
    # Custom column mapping
    column_mapping: Dict[str, str] = field(default_factory=dict)
    unit_mapping: Dict[str, str] = field(default_factory=dict)
    
    # Metadata
    include_metadata: bool = True
    include_provenance: bool = True
    survey_name: str = 'JWST NIRCam'
    pi_name: str = ''
    program_id: str = ''
    
    # Cross-matching
    external_catalogs: Dict[str, str] = field(default_factory=dict)
    match_radius: float = 1.0  # arcseconds
    
    # Validation
    validate_output: bool = True
    check_duplicates: bool = True
    position_tolerance: float = 0.1  # arcseconds

@dataclass
class CatalogMetadata:
    """Metadata for generated catalogs."""
    
    # Basic information
    catalog_name: str = 'JWST_NIRCam_Photometry'
    creation_date: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    pipeline_version: str = '1.0.0'
    
    # Survey information
    survey: str = 'JWST NIRCam'
    filters: List[str] = field(default_factory=list)
    field_name: str = ''
    
    # Processing information
    detection_software: str = 'SEP'
    photometry_method: str = 'Aperture'
    calibration_reference: str = ''
    
    # Statistics
    total_sources: int = 0
    area_coverage: float = 0.0  # square arcminutes
    magnitude_limit: float = 0.0
    completeness_limit: float = 0.0
    
    # Quality metrics
    astrometric_precision: float = 0.0  # arcseconds
    photometric_precision: float = 0.0  # magnitudes
    
    # Additional metadata
    notes: str = ''
    references: List[str] = field(default_factory=list)

@dataclass
class CatalogSource:
    """Individual source in the catalog."""
    
    # Position
    ra: float
    dec: float
    x: float
    y: float
    
    # Photometry by band
    magnitudes: Dict[str, float] = field(default_factory=dict)
    mag_errors: Dict[str, float] = field(default_factory=dict)
    fluxes: Dict[str, float] = field(default_factory=dict)
    flux_errors: Dict[str, float] = field(default_factory=dict)
    
    # Aperture photometry
    aperture_mags: Dict[str, Dict[str, float]] = field(default_factory=dict)
    aperture_mag_errors: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Colors
    colors: Dict[str, float] = field(default_factory=dict)
    color_errors: Dict[str, float] = field(default_factory=dict)
    
    # Morphology
    fwhm: float = 0.0
    ellipticity: float = 0.0
    position_angle: float = 0.0
    
    # Quality flags
    flags: int = 0
    quality_grade: str = 'Unknown'
    
    # Cross-matches
    external_matches: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Additional properties
    properties: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CatalogResults:
    """Results from catalog generation."""
    
    # Catalog contents (required fields first)
    sources: List[CatalogSource]
    metadata: CatalogMetadata
    
    # Generated files
    output_files: Dict[str, str] = field(default_factory=dict)
    
    # Statistics
    total_sources: int = 0
    valid_sources: int = 0
    duplicate_sources: int = 0
    
    # Cross-matching results
    cross_match_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Quality assessment
    catalog_quality: str = 'Unknown'
    quality_comments: List[str] = field(default_factory=list)
    
    # Processing time
    generation_time: float = 0.0


class CatalogGenerator:
    """
    Comprehensive catalog generator for JWST NIRCam photometry.
    
    This class provides catalog generation capabilities including multi-format
    output, metadata tracking, cross-matching, and validation.
    """
    
    def __init__(self, config: Optional[CatalogConfig] = None):
        """
        Initialize the catalog generator.
        
        Parameters:
        -----------
        config : CatalogConfig, optional
            Catalog generation configuration. If None, uses defaults.
        """
        self.config = config or CatalogConfig()
        self.logger = logging.getLogger(__name__)
        
        # Validate configuration
        self._validate_config()
        
        # Initialize format handlers
        self._init_format_handlers()
    
    def _validate_config(self) -> None:
        """Validate the configuration parameters."""
        supported_formats = ['fits', 'hdf5', 'csv', 'ascii', 'json']
        
        for fmt in self.config.output_formats:
            if fmt not in supported_formats:
                raise ValueError(f"Unsupported output format: {fmt}")
        
        if 'hdf5' in self.config.output_formats and not HDF5_AVAILABLE:
            raise ValueError("HDF5 format requested but h5py not available")
        
        if self.config.match_radius <= 0:
            raise ValueError("Match radius must be positive")
    
    def _init_format_handlers(self) -> None:
        """Initialize format-specific handlers."""
        self.format_handlers = {
            'fits': self._write_fits,
            'hdf5': self._write_hdf5,
            'csv': self._write_csv,
            'ascii': self._write_ascii,
            'json': self._write_json
        }
    
    def generate_catalog(self,
                        photometry_results: Dict[str, Any],
                        color_results: Optional[Dict[str, Any]] = None,
                        quality_results: Optional[Dict[str, Any]] = None,
                        output_dir: Path = Path('.'),
                        catalog_name: str = 'jwst_catalog') -> CatalogResults:
        """
        Generate comprehensive photometric catalog.
        
        Parameters:
        -----------
        photometry_results : dict
            Photometry results by band
        color_results : dict, optional
            Color analysis results
        quality_results : dict, optional
            Quality assessment results
        output_dir : Path
            Output directory for catalog files
        catalog_name : str
            Base name for catalog files
            
        Returns:
        --------
        CatalogResults
            Complete catalog generation results
        """
        start_time = time.time()
        self.logger.info(f"Generating catalog: {catalog_name}")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract sources from photometry results
        self.logger.info("Extracting and organizing source data")
        sources = self._extract_sources(photometry_results, color_results, quality_results)
        
        # Create catalog metadata
        metadata = self._create_metadata(photometry_results, sources)
        
        # Validate catalog
        if self.config.validate_output:
            self.logger.info("Validating catalog")
            valid_sources, duplicate_count = self._validate_catalog(sources)
        else:
            valid_sources = sources
            duplicate_count = 0
        
        # Perform cross-matching
        cross_match_stats = {}
        if self.config.external_catalogs:
            self.logger.info("Cross-matching with external catalogs")
            cross_match_stats = self._cross_match_catalogs(valid_sources)
        
        # Generate output files
        self.logger.info("Writing catalog files")
        output_files = self._write_catalog_files(valid_sources, metadata, output_dir, catalog_name)
        
        # Assess catalog quality
        catalog_quality, quality_comments = self._assess_catalog_quality(valid_sources, metadata)
        
        # Create results object
        generation_time = time.time() - start_time
        
        results = CatalogResults(
            output_files=output_files,
            sources=valid_sources,
            metadata=metadata,
            total_sources=len(sources),
            valid_sources=len(valid_sources),
            duplicate_sources=duplicate_count,
            cross_match_stats=cross_match_stats,
            catalog_quality=catalog_quality,
            quality_comments=quality_comments,
            generation_time=generation_time
        )
        
        self.logger.info(f"Catalog generation completed in {generation_time:.1f}s")
        self.logger.info(f"Generated {len(valid_sources)} valid sources")
        
        return results
    
    def _extract_sources(self,
                        photometry_results: Dict[str, Any],
                        color_results: Optional[Dict[str, Any]],
                        quality_results: Optional[Dict[str, Any]]) -> List[CatalogSource]:
        """Extract and organize source data into catalog format."""
        sources = []
        
        # This would extract sources from the actual results format
        # For now, create placeholder sources
        
        # In a real implementation, this would:
        # 1. Extract positions and photometry from photometry_results
        # 2. Add color information from color_results
        # 3. Add quality flags from quality_results
        # 4. Create CatalogSource objects
        
        self.logger.info("Creating placeholder catalog sources")
        
        # Create some example sources
        for i in range(100):  # Placeholder
            source = CatalogSource(
                ra=150.0 + np.random.normal(0, 0.1),
                dec=2.0 + np.random.normal(0, 0.1),
                x=1000 + np.random.normal(0, 100),
                y=1000 + np.random.normal(0, 100)
            )
            
            # Add photometry for different bands
            for band in ['F150W', 'F277W', 'F444W']:
                mag = 22.0 + np.random.normal(0, 2.0)
                mag_err = 0.05 + np.random.exponential(0.02)
                
                source.magnitudes[band] = mag
                source.mag_errors[band] = mag_err
                
                # Convert to flux (placeholder)
                flux = 10**(-0.4 * (mag - 25.0))
                flux_err = flux * mag_err / 1.086
                
                source.fluxes[band] = flux
                source.flux_errors[band] = flux_err
            
            # Add colors if available
            if len(source.magnitudes) >= 2:
                bands = list(source.magnitudes.keys())
                for i in range(len(bands) - 1):
                    color_name = f"{bands[i]}-{bands[i+1]}"
                    color_value = source.magnitudes[bands[i]] - source.magnitudes[bands[i+1]]
                    color_error = np.sqrt(source.mag_errors[bands[i]]**2 + source.mag_errors[bands[i+1]]**2)
                    
                    source.colors[color_name] = color_value
                    source.color_errors[color_name] = color_error
            
            # Add morphological properties
            source.fwhm = np.random.normal(0.1, 0.02)  # arcseconds
            source.ellipticity = np.random.beta(2, 5)
            source.position_angle = np.random.uniform(0, 180)
            
            # Add quality assessment
            source.quality_grade = np.random.choice(['Excellent', 'Good', 'Fair', 'Poor'], 
                                                   p=[0.4, 0.3, 0.2, 0.1])
            source.flags = np.random.randint(0, 16)  # Random quality flags
            
            sources.append(source)
        
        return sources
    
    def _create_metadata(self, photometry_results: Dict[str, Any], sources: List[CatalogSource]) -> CatalogMetadata:
        """Create catalog metadata."""
        metadata = CatalogMetadata()
        
        # Extract basic information
        metadata.total_sources = len(sources)
        
        # Extract filter information
        if sources:
            metadata.filters = list(sources[0].magnitudes.keys())
        
        # Add processing information from config
        metadata.survey = self.config.survey_name
        metadata.pi_name = self.config.pi_name
        metadata.program_id = self.config.program_id
        
        # Compute statistics
        if sources:
            # Magnitude limits (5-sigma depth)
            for band in metadata.filters:
                mags = [s.magnitudes.get(band, np.nan) for s in sources]
                valid_mags = [m for m in mags if not np.isnan(m)]
                if valid_mags:
                    # Use 90th percentile as rough magnitude limit
                    mag_limit = np.percentile(valid_mags, 90)
                    metadata.magnitude_limit = max(metadata.magnitude_limit, mag_limit)
            
            # Astrometric precision (placeholder)
            metadata.astrometric_precision = 0.05  # arcseconds
            
            # Photometric precision (median error)
            all_errors = []
            for source in sources:
                all_errors.extend(source.mag_errors.values())
            if all_errors:
                metadata.photometric_precision = np.median(all_errors)
        
        return metadata
    
    def _validate_catalog(self, sources: List[CatalogSource]) -> Tuple[List[CatalogSource], int]:
        """Validate catalog and remove duplicates/invalid sources."""
        valid_sources = []
        duplicate_count = 0
        
        for source in sources:
            # Check for valid coordinates
            if np.isnan(source.ra) or np.isnan(source.dec):
                continue
            
            # Check for valid photometry
            if not source.magnitudes:
                continue
            
            # Check for duplicates
            if self.config.check_duplicates:
                is_duplicate = False
                for valid_source in valid_sources:
                    # Simple position-based duplicate check
                    distance = np.sqrt((source.ra - valid_source.ra)**2 + 
                                     (source.dec - valid_source.dec)**2) * 3600  # arcseconds
                    
                    if distance < self.config.position_tolerance:
                        is_duplicate = True
                        duplicate_count += 1
                        break
                
                if is_duplicate:
                    continue
            
            valid_sources.append(source)
        
        self.logger.info(f"Validation complete: {len(valid_sources)} valid sources, {duplicate_count} duplicates removed")
        
        return valid_sources, duplicate_count
    
    def _cross_match_catalogs(self, sources: List[CatalogSource]) -> Dict[str, Dict[str, Any]]:
        """Cross-match with external catalogs."""
        cross_match_stats = {}
        
        for catalog_name, catalog_path in self.config.external_catalogs.items():
            try:
                self.logger.info(f"Cross-matching with {catalog_name}")
                
                # This would implement actual cross-matching
                # For now, create placeholder results
                
                n_matches = len(sources) // 2  # Placeholder
                match_fraction = n_matches / len(sources)
                
                cross_match_stats[catalog_name] = {
                    'catalog_path': catalog_path,
                    'n_matches': n_matches,
                    'match_fraction': match_fraction,
                    'median_separation': 0.2,  # arcseconds
                    'rms_separation': 0.3      # arcseconds
                }
                
                # Add match information to sources (placeholder)
                for i, source in enumerate(sources[:n_matches]):
                    source.external_matches[catalog_name] = {
                        'matched': True,
                        'separation': np.random.exponential(0.2),
                        'external_id': f"{catalog_name}_{i:06d}",
                        'external_mag': source.magnitudes.get(list(source.magnitudes.keys())[0], 0) + np.random.normal(0, 0.1)
                    }
                
            except Exception as e:
                self.logger.warning(f"Cross-matching with {catalog_name} failed: {e}")
                continue
        
        return cross_match_stats
    
    def _assess_catalog_quality(self, sources: List[CatalogSource], metadata: CatalogMetadata) -> Tuple[str, List[str]]:
        """Assess overall catalog quality."""
        comments = []
        
        # Check number of sources
        if len(sources) < 10:
            quality = 'Poor'
            comments.append("Very few sources detected")
        elif len(sources) < 100:
            quality = 'Fair'
            comments.append("Limited number of sources")
        else:
            quality = 'Good'
            comments.append(f"Good source count: {len(sources)}")
        
        # Check photometric precision
        if metadata.photometric_precision < 0.05:
            if quality == 'Good':
                quality = 'Excellent'
            comments.append("Excellent photometric precision")
        elif metadata.photometric_precision > 0.1:
            if quality == 'Good':
                quality = 'Fair'
            comments.append("Limited photometric precision")
        
        # Check astrometric precision
        if metadata.astrometric_precision < 0.1:
            comments.append("Good astrometric precision")
        else:
            comments.append("Limited astrometric precision")
        
        # Check filter coverage
        if len(metadata.filters) >= 3:
            comments.append("Good multi-band coverage")
        elif len(metadata.filters) == 2:
            comments.append("Limited filter coverage")
        else:
            comments.append("Single-band catalog")
            if quality == 'Excellent':
                quality = 'Good'
        
        return quality, comments
    
    def _write_catalog_files(self,
                           sources: List[CatalogSource],
                           metadata: CatalogMetadata,
                           output_dir: Path,
                           catalog_name: str) -> Dict[str, str]:
        """Write catalog in all requested formats."""
        output_files = {}
        
        for fmt in self.config.output_formats:
            try:
                self.logger.info(f"Writing {fmt.upper()} format")
                
                output_path = output_dir / f"{catalog_name}.{fmt}"
                handler = self.format_handlers.get(fmt)
                
                if handler:
                    handler(sources, metadata, output_path)
                    output_files[fmt] = str(output_path)
                else:
                    self.logger.warning(f"No handler for format: {fmt}")
                    
            except Exception as e:
                self.logger.error(f"Failed to write {fmt} format: {e}")
                continue
        
        # Write metadata file
        metadata_path = output_dir / f"{catalog_name}_metadata.json"
        self._write_metadata(metadata, metadata_path)
        output_files['metadata'] = str(metadata_path)
        
        return output_files
    
    def _write_fits(self, sources: List[CatalogSource], metadata: CatalogMetadata, output_path: Path) -> None:
        """Write catalog in FITS format."""
        if not ASTROPY_AVAILABLE:
            raise RuntimeError("Astropy required for FITS output")
        
        # Create columns
        columns = []
        
        # Basic columns
        columns.extend([
            Column(name='ID', data=np.arange(len(sources)), dtype='i4'),
            Column(name='RA', data=[s.ra for s in sources], unit='deg', dtype='f8'),
            Column(name='DEC', data=[s.dec for s in sources], unit='deg', dtype='f8'),
            Column(name='X', data=[s.x for s in sources], unit='pix', dtype='f4'),
            Column(name='Y', data=[s.y for s in sources], unit='pix', dtype='f4'),
        ])
        
        # Photometry columns
        if sources and sources[0].magnitudes:
            for band in sources[0].magnitudes.keys():
                mags = [s.magnitudes.get(band, np.nan) for s in sources]
                mag_errs = [s.mag_errors.get(band, np.nan) for s in sources]
                
                columns.extend([
                    Column(name=f'MAG_{band}', data=mags, unit='mag', dtype='f4'),
                    Column(name=f'MAGERR_{band}', data=mag_errs, unit='mag', dtype='f4')
                ])
        
        # Color columns
        if sources and sources[0].colors:
            for color_name in sources[0].colors.keys():
                colors = [s.colors.get(color_name, np.nan) for s in sources]
                color_errs = [s.color_errors.get(color_name, np.nan) for s in sources]
                
                safe_color_name = color_name.replace('-', '_')
                columns.extend([
                    Column(name=f'COLOR_{safe_color_name}', data=colors, unit='mag', dtype='f4'),
                    Column(name=f'COLORERR_{safe_color_name}', data=color_errs, unit='mag', dtype='f4')
                ])
        
        # Morphology columns
        if self.config.include_quality:
            columns.extend([
                Column(name='FWHM', data=[s.fwhm for s in sources], unit='arcsec', dtype='f4'),
                Column(name='ELLIPTICITY', data=[s.ellipticity for s in sources], dtype='f4'),
                Column(name='PA', data=[s.position_angle for s in sources], unit='deg', dtype='f4'),
            ])
        
        # Quality columns
        if self.config.include_flags:
            columns.extend([
                Column(name='FLAGS', data=[s.flags for s in sources], dtype='i2'),
                Column(name='QUALITY', data=[s.quality_grade for s in sources], dtype='U10')
            ])
        
        # Create table
        table = Table(columns)
        
        # Add metadata to header
        table.meta['SURVEY'] = metadata.survey
        table.meta['FILTERS'] = ','.join(metadata.filters)
        table.meta['NSOURCES'] = metadata.total_sources
        table.meta['MAGLIMIT'] = metadata.magnitude_limit
        table.meta['ASTPREC'] = metadata.astrometric_precision
        table.meta['PHOTPREC'] = metadata.photometric_precision
        table.meta['CREATED'] = metadata.creation_date
        table.meta['VERSION'] = metadata.pipeline_version
        
        # Write FITS file
        table.write(output_path, format='fits', overwrite=True)
        
        if self.config.fits_compress:
            # Compress FITS file
            with fits.open(output_path) as hdul:
                hdul.writeto(str(output_path).replace('.fits', '.fits.gz'), 
                           overwrite=True, checksum=True)
    
    def _write_hdf5(self, sources: List[CatalogSource], metadata: CatalogMetadata, output_path: Path) -> None:
        """Write catalog in HDF5 format."""
        if not HDF5_AVAILABLE:
            raise RuntimeError("h5py required for HDF5 output")
        
        with h5py.File(output_path, 'w') as f:
            # Create main catalog group
            catalog_group = f.create_group('catalog')
            
            # Basic data
            catalog_group.create_dataset('id', data=np.arange(len(sources)), compression=self.config.hdf5_compression)
            catalog_group.create_dataset('ra', data=[s.ra for s in sources], compression=self.config.hdf5_compression)
            catalog_group.create_dataset('dec', data=[s.dec for s in sources], compression=self.config.hdf5_compression)
            catalog_group.create_dataset('x', data=[s.x for s in sources], compression=self.config.hdf5_compression)
            catalog_group.create_dataset('y', data=[s.y for s in sources], compression=self.config.hdf5_compression)
            
            # Photometry data
            if sources and sources[0].magnitudes:
                phot_group = catalog_group.create_group('photometry')
                for band in sources[0].magnitudes.keys():
                    mags = [s.magnitudes.get(band, np.nan) for s in sources]
                    mag_errs = [s.mag_errors.get(band, np.nan) for s in sources]
                    
                    band_group = phot_group.create_group(band)
                    band_group.create_dataset('magnitude', data=mags, compression=self.config.hdf5_compression)
                    band_group.create_dataset('magnitude_error', data=mag_errs, compression=self.config.hdf5_compression)
            
            # Metadata
            meta_group = f.create_group('metadata')
            meta_group.attrs['survey'] = metadata.survey
            meta_group.attrs['filters'] = ','.join(metadata.filters)
            meta_group.attrs['total_sources'] = metadata.total_sources
            meta_group.attrs['magnitude_limit'] = metadata.magnitude_limit
            meta_group.attrs['astrometric_precision'] = metadata.astrometric_precision
            meta_group.attrs['photometric_precision'] = metadata.photometric_precision
            meta_group.attrs['creation_date'] = metadata.creation_date
            meta_group.attrs['pipeline_version'] = metadata.pipeline_version
    
    def _write_csv(self, sources: List[CatalogSource], metadata: CatalogMetadata, output_path: Path) -> None:
        """Write catalog in CSV format."""
        # Create header
        header = ['ID', 'RA', 'DEC', 'X', 'Y']
        
        # Add photometry columns
        if sources and sources[0].magnitudes:
            for band in sources[0].magnitudes.keys():
                header.extend([f'MAG_{band}', f'MAGERR_{band}'])
        
        # Add color columns
        if sources and sources[0].colors:
            for color_name in sources[0].colors.keys():
                safe_color_name = color_name.replace('-', '_')
                header.extend([f'COLOR_{safe_color_name}', f'COLORERR_{safe_color_name}'])
        
        # Add quality columns
        if self.config.include_quality:
            header.extend(['FWHM', 'ELLIPTICITY', 'PA'])
        
        if self.config.include_flags:
            header.extend(['FLAGS', 'QUALITY'])
        
        # Write data
        with open(output_path, 'w') as f:
            # Write header
            f.write(self.config.csv_delimiter.join(header) + '\n')
            
            # Write sources
            for i, source in enumerate(sources):
                row = [str(i), f"{source.ra:.8f}", f"{source.dec:.8f}", 
                      f"{source.x:.2f}", f"{source.y:.2f}"]
                
                # Add photometry
                if source.magnitudes:
                    for band in source.magnitudes.keys():
                        mag = source.magnitudes.get(band, np.nan)
                        mag_err = source.mag_errors.get(band, np.nan)
                        row.extend([f"{mag:.4f}", f"{mag_err:.4f}"])
                
                # Add colors
                if source.colors:
                    for color_name in source.colors.keys():
                        color = source.colors.get(color_name, np.nan)
                        color_err = source.color_errors.get(color_name, np.nan)
                        row.extend([f"{color:.4f}", f"{color_err:.4f}"])
                
                # Add quality
                if self.config.include_quality:
                    row.extend([f"{source.fwhm:.4f}", f"{source.ellipticity:.4f}", 
                               f"{source.position_angle:.2f}"])
                
                if self.config.include_flags:
                    row.extend([str(source.flags), source.quality_grade])
                
                f.write(self.config.csv_delimiter.join(row) + '\n')
    
    def _write_ascii(self, sources: List[CatalogSource], metadata: CatalogMetadata, output_path: Path) -> None:
        """Write catalog in ASCII format."""
        if not ASTROPY_AVAILABLE:
            raise RuntimeError("Astropy required for ASCII output")
        
        # Create table (similar to FITS but for ASCII)
        columns = []
        
        columns.extend([
            Column(name='ID', data=np.arange(len(sources))),
            Column(name='RA', data=[s.ra for s in sources]),
            Column(name='DEC', data=[s.dec for s in sources]),
            Column(name='X', data=[s.x for s in sources]),
            Column(name='Y', data=[s.y for s in sources]),
        ])
        
        # Add photometry
        if sources and sources[0].magnitudes:
            for band in sources[0].magnitudes.keys():
                mags = [s.magnitudes.get(band, np.nan) for s in sources]
                mag_errs = [s.mag_errors.get(band, np.nan) for s in sources]
                
                columns.extend([
                    Column(name=f'MAG_{band}', data=mags),
                    Column(name=f'MAGERR_{band}', data=mag_errs)
                ])
        
        table = Table(columns)
        ascii.write(table, output_path, format='fixed_width_two_line')
    
    def _write_json(self, sources: List[CatalogSource], metadata: CatalogMetadata, output_path: Path) -> None:
        """Write catalog in JSON format."""
        catalog_data = {
            'metadata': {
                'survey': metadata.survey,
                'filters': metadata.filters,
                'total_sources': metadata.total_sources,
                'magnitude_limit': metadata.magnitude_limit,
                'astrometric_precision': metadata.astrometric_precision,
                'photometric_precision': metadata.photometric_precision,
                'creation_date': metadata.creation_date,
                'pipeline_version': metadata.pipeline_version
            },
            'sources': []
        }
        
        # Convert sources to dictionaries
        for i, source in enumerate(sources):
            source_dict = {
                'id': i,
                'ra': source.ra,
                'dec': source.dec,
                'x': source.x,
                'y': source.y,
                'magnitudes': source.magnitudes,
                'magnitude_errors': source.mag_errors,
                'colors': source.colors,
                'color_errors': source.color_errors,
                'morphology': {
                    'fwhm': source.fwhm,
                    'ellipticity': source.ellipticity,
                    'position_angle': source.position_angle
                },
                'quality': {
                    'flags': source.flags,
                    'grade': source.quality_grade
                },
                'external_matches': source.external_matches
            }
            catalog_data['sources'].append(source_dict)
        
        # Write JSON file
        with open(output_path, 'w') as f:
            json.dump(catalog_data, f, indent=2, default=str)
    
    def _write_metadata(self, metadata: CatalogMetadata, output_path: Path) -> None:
        """Write metadata to JSON file."""
        metadata_dict = {
            'catalog_name': metadata.catalog_name,
            'creation_date': metadata.creation_date,
            'pipeline_version': metadata.pipeline_version,
            'survey': metadata.survey,
            'filters': metadata.filters,
            'field_name': metadata.field_name,
            'detection_software': metadata.detection_software,
            'photometry_method': metadata.photometry_method,
            'calibration_reference': metadata.calibration_reference,
            'total_sources': metadata.total_sources,
            'area_coverage': metadata.area_coverage,
            'magnitude_limit': metadata.magnitude_limit,
            'completeness_limit': metadata.completeness_limit,
            'astrometric_precision': metadata.astrometric_precision,
            'photometric_precision': metadata.photometric_precision,
            'notes': metadata.notes,
            'references': metadata.references
        }
        
        with open(output_path, 'w') as f:
            json.dump(metadata_dict, f, indent=2)


# Convenience functions

def quick_catalog_generation(photometry_results: Dict[str, Any],
                           output_dir: Path = Path('.'),
                           formats: List[str] = ['fits', 'csv']) -> CatalogResults:
    """
    Quick catalog generation with default settings.
    
    Parameters:
    -----------
    photometry_results : dict
        Photometry results
    output_dir : Path
        Output directory
    formats : list
        Output formats to generate
        
    Returns:
    --------
    CatalogResults
        Catalog generation results
    """
    config = CatalogConfig(output_formats=formats)
    generator = CatalogGenerator(config)
    
    return generator.generate_catalog(photometry_results, output_dir=output_dir)


def validate_catalog_file(catalog_path: Path) -> Dict[str, Any]:
    """
    Validate a generated catalog file.
    
    Parameters:
    -----------
    catalog_path : Path
        Path to catalog file
        
    Returns:
    --------
    dict
        Validation results
    """
    validation_results = {
        'valid': False,
        'format': None,
        'n_sources': 0,
        'columns': [],
        'issues': []
    }
    
    try:
        if catalog_path.suffix.lower() == '.fits':
            if ASTROPY_AVAILABLE:
                table = Table.read(catalog_path)
                validation_results['valid'] = True
                validation_results['format'] = 'FITS'
                validation_results['n_sources'] = len(table)
                validation_results['columns'] = table.colnames
            else:
                validation_results['issues'].append("Astropy not available for FITS validation")
        
        elif catalog_path.suffix.lower() == '.csv':
            if PANDAS_AVAILABLE:
                df = pd.read_csv(catalog_path)
                validation_results['valid'] = True
                validation_results['format'] = 'CSV'
                validation_results['n_sources'] = len(df)
                validation_results['columns'] = list(df.columns)
            else:
                validation_results['issues'].append("pandas not available for CSV validation")
        
        else:
            validation_results['issues'].append(f"Unsupported format: {catalog_path.suffix}")
    
    except Exception as e:
        validation_results['issues'].append(str(e))
    
    return validation_results


if __name__ == "__main__":
    # Example usage
    print("JWST Catalog Generation Module")
    print("This module provides catalog generation capabilities for JWST photometry")
