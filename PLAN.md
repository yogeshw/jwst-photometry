# JWST NIRCam Photometry Enhancement Plan

## Overview
This document outlines a comprehensive plan to enhance the existing JWST photometry codebase to achieve accurate and robust photometry for JWST NIRCam observations using the SEP (Source Extractor Python) library. The plan addresses current limitations and provides a roadmap for implementing industry-standard photometric practices.

## Current State Assessment

### Existing Strengths
- Basic framework for multi-band JWST photometry (7 bands: F115W, F150W, F200W, F277W, F356W, F410M, F444W)
- Configuration-driven approach using YAML
- SEP integration for source detection
- PSF homogenization framework using Pypher
- Aperture photometry capabilities
- Modular code structure with separate files for different functionalities

### Current Limitations
1. **Incomplete SEP integration**: Limited use of SEP's full capabilities
2. **Missing critical photometric steps**: No proper background estimation, flux calibration, or error propagation
3. **Incomplete PSF handling**: Basic PSF matching without proper empirical PSF modeling
4. **Insufficient error handling**: Limited robustness for real-world data
5. **Missing JWST-specific considerations**: No handling of detector effects, distortion, or calibration
6. **Incomplete catalog output**: Missing essential photometric parameters
7. **No data validation**: Missing input data quality checks

## Detailed Enhancement Plan

### Phase 1: Core Infrastructure Improvements (Weeks 1-2)

#### 1.1 Enhanced Configuration Management
- **File**: `src/config_manager.py` (new)
- **Purpose**: Robust configuration validation and management
- **Tasks**:
  - Create comprehensive configuration schema validation
  - Add support for filter-specific parameters
  - Implement configuration inheritance and overrides
  - Add validation for file paths and data integrity
  - Support for observation-specific parameters (exposure time, zero points, etc.)

#### 1.2 JWST-Specific Data Handling
- **File**: `src/jwst_data_handler.py` (new)
- **Purpose**: Handle JWST-specific data formats and metadata
- **Tasks**:
  - Implement JWST FITS header parsing
  - Extract and validate WCS information
  - Handle pixel scale variations across detectors
  - Extract exposure times, gain values, and calibration factors
  - Implement proper unit handling (MJy/sr to flux densities)

#### 1.3 Enhanced Error Handling and Logging
- **Files**: Update all existing files
- **Purpose**: Robust error handling and comprehensive logging
- **Tasks**:
  - Implement structured logging with different verbosity levels
  - Add comprehensive error handling with meaningful error messages
  - Create data validation routines
  - Add progress tracking for long-running operations
  - Implement memory usage monitoring

### Phase 2: Advanced Source Detection and SEP Integration (Weeks 3-4)

#### 2.1 Comprehensive Background Estimation
- **File**: `src/background.py` (new)
- **Purpose**: Implement sophisticated background estimation using SEP
- **Tasks**:
  - Multi-scale background modeling using SEP's Background class
  - Spatially varying background estimation
  - Background mesh optimization for different field complexities
  - Bad pixel masking and interpolation
  - Background gradient correction for wide-field observations

#### 2.2 Advanced Source Detection
- **File**: `src/detection.py` (enhance existing)
- **Purpose**: Implement state-of-the-art source detection
- **Tasks**:
  - Multi-threshold detection for completeness optimization
  - Adaptive detection parameters based on local noise properties
  - Deblending optimization for crowded fields
  - Source classification (point sources vs. extended objects)
  - Detection significance assessment
  - Spurious source filtering

#### 2.3 Segmentation and Masking
- **File**: `src/segmentation.py` (new)
- **Purpose**: Advanced segmentation for accurate photometry
- **Tasks**:
  - Implement watershed segmentation using SEP
  - Create adaptive apertures based on source morphology
  - Generate contamination masks for neighboring sources
  - Implement Kron apertures with proper elliptical fitting
  - Bad pixel and cosmic ray masking

### Phase 3: PSF Modeling and Homogenization (Weeks 5-6)

#### 3.1 Empirical PSF Generation
- **File**: `src/psf.py` (enhance existing)
- **Purpose**: Generate high-quality empirical PSFs
- **Tasks**:
  - Implement robust star selection using magnitude, morphology, and color criteria
  - Generate spatially varying PSF models across the field
  - PSF quality assessment and outlier rejection
  - Sub-pixel PSF centering and normalization
  - PSF FWHM and ellipticity measurements

#### 3.2 PSF Homogenization
- **File**: `src/psf_matching.py` (new)
- **Purpose**: Accurate PSF matching across bands
- **Tasks**:
  - Implement Pypher-based PSF matching with optimization
  - Quality assessment of PSF matching results
  - Spatially varying PSF matching for wide fields
  - PSF matching validation using stellar photometry
  - Preserve flux calibration during PSF matching

#### 3.3 PSF Photometry
- **File**: `src/psf_photometry.py` (new)
- **Purpose**: Implement PSF photometry for crowded fields
- **Tasks**:
  - PSF fitting photometry using empirical PSFs
  - Multi-band simultaneous PSF fitting
  - Crowded field deblending
  - PSF photometry error estimation

### Phase 4: Advanced Photometry (Weeks 7-8)

#### 4.1 Aperture Photometry Enhancement
- **File**: `src/photometry.py` (enhance existing)
- **Purpose**: Implement comprehensive aperture photometry
- **Tasks**:
  - Multiple aperture sizes with proper scaling
  - Elliptical apertures based on source morphology
  - Kron apertures with optimized parameters
  - Aperture corrections for finite aperture sizes
  - Local background estimation for each source
  - Contamination correction from nearby sources

#### 4.2 Flux Calibration
- **File**: `src/calibration.py` (new)
- **Purpose**: Proper flux calibration to physical units
- **Tasks**:
  - Convert from detector units to physical flux densities
  - Apply zero-point calibrations
  - Aperture corrections to total magnitudes
  - Color-dependent calibration corrections
  - Galactic extinction corrections
  - Systematic calibration uncertainty propagation

#### 4.3 Error Estimation
- **File**: `src/uncertainties.py` (new)
- **Purpose**: Comprehensive photometric error estimation
- **Tasks**:
  - Poisson noise calculation from source and background
  - Systematic uncertainties from calibration
  - Correlated noise estimation
  - PSF matching uncertainties
  - Background subtraction uncertainties
  - Total error budget calculation

### Phase 5: Advanced Features (Weeks 9-10)

#### 5.1 Color-Based Analysis
- **File**: `src/colors.py` (new)
- **Purpose**: Multi-band color analysis and quality assessment
- **Tasks**:
  - Color-color diagram generation
  - Star-galaxy separation using colors and morphology
  - Photometric redshift preparation
  - Color-magnitude diagrams
  - Color-dependent systematic corrections

#### 5.2 Quality Assessment
- **File**: `src/quality.py` (new)
- **Purpose**: Comprehensive photometry quality assessment
- **Tasks**:
  - Photometric quality flags
  - Completeness and reliability assessment
  - Systematic error identification
  - Comparison with external catalogs
  - Astrometric quality assessment

#### 5.3 Catalog Generation
- **File**: `src/catalog.py` (new)
- **Purpose**: Generate comprehensive photometric catalogs
- **Tasks**:
  - Multi-format output (FITS, HDF5, CSV)
  - Comprehensive metadata inclusion
  - Catalog cross-matching capabilities
  - Virtual Observatory compatibility
  - Catalog validation and statistics

### Phase 6: JWST-Specific Enhancements (Weeks 11-12)

#### 6.1 Detector-Specific Corrections
- **File**: `src/jwst_corrections.py` (new)
- **Purpose**: Handle JWST NIRCam specific effects
- **Tasks**:
  - Detector nonlinearity corrections
  - Persistence effect modeling
  - Cross-talk corrections between detectors
  - Saturation handling and flag propagation
  - Detector-to-detector photometric calibration

#### 6.2 Astrometric Enhancement
- **File**: `src/astrometry.py` (new)
- **Purpose**: High-precision astrometry for JWST data
- **Tasks**:
  - WCS validation and refinement
  - Proper motion corrections
  - Parallax corrections for nearby sources
  - Systematic astrometric error modeling
  - Multi-epoch astrometry

#### 6.3 Multi-Epoch Handling
- **File**: `src/multiepoch.py` (new)
- **Purpose**: Handle time-series photometry
- **Tasks**:
  - Multi-epoch source matching
  - Variability detection and characterization
  - Proper motion measurement
  - Long-term systematic monitoring

### Phase 7: Performance and Scalability (Weeks 13-14)

#### 7.1 Performance Optimization
- **Files**: All modules
- **Purpose**: Optimize for large datasets
- **Tasks**:
  - Memory-efficient processing for large images
  - Parallel processing for multi-band operations
  - Chunked processing for memory management
  - GPU acceleration where applicable
  - Caching strategies for repeated operations

#### 7.2 Pipeline Integration
- **File**: `src/pipeline.py` (new)
- **Purpose**: End-to-end pipeline management
- **Tasks**:
  - Workflow management and dependency tracking
  - Checkpoint and resume capabilities
  - Batch processing for multiple fields
  - Pipeline monitoring and diagnostics
  - Integration with JWST calibration pipeline

### Phase 8: Testing and Validation (Weeks 15-16)

#### 8.1 Comprehensive Testing
- **Files**: Enhance all test files
- **Purpose**: Ensure code reliability and accuracy
- **Tasks**:
  - Unit tests for all functions
  - Integration tests for complete workflows
  - Regression tests against known results
  - Performance benchmarking
  - Edge case testing

#### 8.2 Scientific Validation
- **File**: `validation/` (new directory)
- **Purpose**: Validate scientific accuracy
- **Tasks**:
  - Comparison with HST photometry where available
  - Validation against theoretical predictions
  - Cross-validation with other JWST photometry tools
  - Systematic error characterization
  - Completeness and reliability assessment

## Technical Implementation Details

### Dependencies Enhancement
Update `requirements.txt` to include:
```
numpy>=1.21.0
scipy>=1.7.0
astropy>=5.0
sep>=1.2.1
photutils>=1.5.0
pypher>=0.4.0
pyyaml>=6.0
matplotlib>=3.5.0
h5py>=3.6.0
numba>=0.56.0
scikit-image>=0.19.0
tqdm>=4.62.0
```

### Configuration Schema Enhancement
Extend `config.yaml` with:
- Detailed SEP parameters for different field types
- PSF modeling parameters
- Calibration reference data paths
- Quality control thresholds
- Output format specifications
- Performance tuning parameters

### Error Handling Strategy
- Implement graceful degradation for missing data
- Comprehensive input validation
- Recovery strategies for common failure modes
- Detailed error reporting with suggested solutions

### Documentation Requirements
- Comprehensive API documentation
- Tutorial notebooks for common use cases
- Performance optimization guidelines
- Troubleshooting guide
- Scientific validation reports

## Expected Outcomes

### Performance Metrics
- **Accuracy**: <2% photometric accuracy for sources brighter than 25th magnitude
- **Completeness**: >90% completeness to 26th magnitude in crowded fields
- **Reliability**: <5% false detection rate
- **Processing Speed**: <1 hour for typical JWST pointing with all bands

### Scientific Capabilities
- Multi-band photometry with proper error propagation
- Star-galaxy separation to 26th magnitude
- Color-magnitude diagrams for stellar population analysis
- Photometric redshift-quality measurements
- Time-domain photometry capabilities

### Code Quality
- >95% test coverage
- Comprehensive documentation
- Modular, maintainable architecture
- Performance optimization for large datasets
- Scientific validation against established benchmarks

## Risk Mitigation

### Technical Risks
- **Memory limitations**: Implement chunked processing and streaming
- **Performance bottlenecks**: Profile and optimize critical paths
- **Dependency conflicts**: Use virtual environments and version pinning

### Scientific Risks
- **Systematic errors**: Implement comprehensive validation suite
- **Calibration uncertainties**: Use multiple calibration approaches
- **Algorithm limitations**: Provide multiple algorithm options

### Timeline Risks
- **Scope creep**: Maintain strict prioritization
- **Dependency delays**: Identify critical path early
- **Testing complexity**: Implement continuous integration

## Success Criteria

### Minimum Viable Product (MVP)
- Accurate multi-band photometry using SEP
- Proper PSF homogenization
- Calibrated magnitudes and colors
- Comprehensive error estimation
- Quality flags and assessments

### Full Implementation
- All planned features implemented
- Comprehensive testing and validation
- Performance optimization completed
- Scientific validation published
- Community adoption metrics achieved

This plan provides a roadmap for transforming the existing codebase into a production-ready tool for accurate JWST NIRCam photometry. The phased approach ensures steady progress while maintaining scientific rigor and code quality throughout the development process.
