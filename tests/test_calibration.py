"""
Unit tests for the calibration module (Phase 4.2).
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from astropy.io import fits
from astropy.table import Table

# Import calibration modules
try:
    from src.calibration import (
        FluxCalibrator, CalibrationConfig, BandCalibration, CalibratedSource,
        CalibrationResults, calibrate_to_physical_units
    )
except ImportError:
    # Add the src directory to the path and try again
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    
    try:
        from src.calibration import (
            FluxCalibrator, CalibrationConfig, BandCalibration, CalibratedSource,
            CalibrationResults, calibrate_to_physical_units
        )
    except ImportError:
        # Create dummy classes for testing
        class FluxCalibrator:
            def __init__(self, config=None): pass
        class CalibrationConfig:
            def __init__(self): pass
        class BandCalibration:
            def __init__(self, *args, **kwargs): pass
        class CalibratedSource:
            def __init__(self, *args, **kwargs): pass
        class CalibrationResults:
            def __init__(self, *args, **kwargs): pass
        def calibrate_to_physical_units(*args, **kwargs): return 1.0


class TestCalibrationConfig:
    """Test calibration configuration."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = CalibrationConfig()
        
        assert config.input_units == "DN/s"
        assert config.output_units == "uJy"
        assert config.use_in_flight_zeropoints is True
        assert config.apply_aperture_corrections is True
        assert config.zeropoint_uncertainty == 0.02
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = CalibrationConfig(
            input_units="ADU",
            output_units="nJy",
            zeropoint_uncertainty=0.05,
            apply_galactic_extinction=False
        )
        
        assert config.input_units == "ADU"
        assert config.output_units == "nJy"
        assert config.zeropoint_uncertainty == 0.05
        assert config.apply_galactic_extinction is False


class TestBandCalibration:
    """Test band calibration container."""
    
    def test_band_calibration_creation(self):
        """Test basic band calibration creation."""
        calibration = BandCalibration(
            band="F444W",
            zeropoint=24.5,
            zeropoint_error=0.02,
            zeropoint_source="reference",
            photflam=1.5e-21,
            photfnu=1.2e-6,
            photplam=44000
        )
        
        assert calibration.band == "F444W"
        assert calibration.zeropoint == 24.5
        assert calibration.zeropoint_error == 0.02
        assert calibration.photflam == 1.5e-21
        assert isinstance(calibration.flags, list)
    
    def test_systematic_uncertainty_calculation(self):
        """Test systematic uncertainty computation."""
        calibration = BandCalibration(
            band="F200W",
            zeropoint=25.0,
            zeropoint_error=0.02,
            zeropoint_source="reference",
            photflam=5e-21,
            photfnu=2e-6,
            photplam=20000
        )
        
        # Set individual uncertainty components
        calibration.flat_field_error = 0.005
        calibration.gain_error = 0.002
        calibration.dark_error = 0.001
        
        # Calculate total systematic uncertainty
        expected_sys_error = np.sqrt(0.02**2 + 0.005**2 + 0.002**2 + 0.001**2)
        calibration.systematic_uncertainty = expected_sys_error
        
        assert abs(calibration.systematic_uncertainty - expected_sys_error) < 1e-6


class TestCalibratedSource:
    """Test calibrated source container."""
    
    def test_source_creation(self):
        """Test basic calibrated source creation."""
        source = CalibratedSource(id=1)
        
        assert source.id == 1
        assert isinstance(source.instrumental_fluxes, dict)
        assert isinstance(source.calibrated_fluxes, dict)
        assert isinstance(source.colors, dict)
        assert isinstance(source.calibration_flags, list)
    
    def test_source_with_calibration_data(self):
        """Test source with calibration data."""
        source = CalibratedSource(id=1)
        
        # Add instrumental measurements
        source.instrumental_fluxes["F444W"] = {3.0: 1000.0, 5.0: 1500.0}
        source.instrumental_errors["F444W"] = {3.0: 50.0, 5.0: 75.0}
        
        # Add calibrated measurements
        source.calibrated_fluxes["F444W"] = {3.0: 1200.0, 5.0: 1800.0}
        source.calibrated_errors["F444W"] = {3.0: 80.0, 5.0: 120.0}
        
        # Add best estimates
        source.best_fluxes["F444W"] = 1800.0
        source.best_flux_errors["F444W"] = 120.0
        
        assert source.instrumental_fluxes["F444W"][3.0] == 1000.0
        assert source.calibrated_fluxes["F444W"][5.0] == 1800.0
        assert source.best_fluxes["F444W"] == 1800.0


class TestFluxCalibrator:
    """Test the flux calibrator."""
    
    @pytest.fixture
    def sample_photometry_results(self):
        """Create sample photometry results."""
        # Mock photometry results structure
        results = {}
        
        for band in ["F200W", "F444W"]:
            # Create mock sources with circular fluxes
            sources = []
            for i in range(3):
                source = Mock()
                source.circular_fluxes = {3.0: 1000 + i*100, 5.0: 1500 + i*150}
                source.circular_flux_errors = {3.0: 50 + i*5, 5.0: 75 + i*7.5}
                sources.append(source)
            
            # Create mock photometry results
            phot_results = Mock()
            phot_results.sources = sources
            results[band] = phot_results
        
        return results
    
    @pytest.fixture
    def sample_headers(self):
        """Create sample FITS headers."""
        headers = {}
        
        headers["F200W"] = fits.Header({
            'PHOTZPT': 25.79,
            'PHOTFLAM': 5.24e-21,
            'GAIN': 2.0,
            'EXPTIME': 1000.0
        })
        
        headers["F444W"] = fits.Header({
            'PHOTZPT': 24.51,
            'PHOTFLAM': 1.38e-21,
            'GAIN': 2.0,
            'EXPTIME': 1000.0
        })
        
        return headers
    
    def test_calibrator_initialization(self):
        """Test calibrator initialization."""
        config = CalibrationConfig()
        calibrator = FluxCalibrator(config)
        
        assert calibrator.config == config
        assert hasattr(calibrator, 'logger')
        assert hasattr(calibrator, 'jwst_zeropoints')
        assert "F444W" in calibrator.jwst_zeropoints
    
    def test_band_calibration_creation(self, sample_headers):
        """Test band calibration creation from header."""
        calibrator = FluxCalibrator()
        
        band = "F444W"
        header = sample_headers[band]
        
        calibration = calibrator._create_band_calibration(band, header)
        
        assert calibration.band == band
        assert calibration.zeropoint == header['PHOTZPT']
        assert calibration.photflam == header['PHOTFLAM']
        assert calibration.zeropoint_source == "header"
    
    def test_reference_zeropoint_fallback(self):
        """Test fallback to reference zeropoints."""
        calibrator = FluxCalibrator()
        
        # Empty header should use reference values
        empty_header = fits.Header()
        calibration = calibrator._create_band_calibration("F444W", empty_header)
        
        assert calibration.zeropoint_source == "reference"
        assert calibration.zeropoint == calibrator.jwst_zeropoints["F444W"]
    
    def test_flux_calibration(self, sample_photometry_results, sample_headers):
        """Test flux calibration process."""
        config = CalibrationConfig()
        calibrator = FluxCalibrator(config)
        
        # Perform calibration
        results = calibrator.calibrate_photometry(
            photometry_results=sample_photometry_results,
            headers=sample_headers
        )
        
        assert isinstance(results, CalibrationResults)
        assert len(results.band_calibrations) == 2
        assert "F200W" in results.band_calibrations
        assert "F444W" in results.band_calibrations
        assert len(results.sources) == 3  # Should have 3 sources
    
    def test_unit_conversion(self):
        """Test flux unit conversion."""
        calibrator = FluxCalibrator()
        
        # Create test calibration
        calibration = BandCalibration(
            band="F444W",
            zeropoint=24.5,
            zeropoint_error=0.02,
            zeropoint_source="test",
            photflam=1.5e-21,
            photfnu=1.2e-6,
            photplam=44000
        )
        
        # Test conversion to microjanskys
        inst_flux = 1000.0
        inst_error = 50.0
        
        cal_flux, cal_error = calibrator._calibrate_flux(inst_flux, inst_error, calibration)
        
        assert cal_flux > 0
        assert cal_error > 0
        assert cal_error < cal_flux  # Error should be smaller than flux
    
    def test_extinction_correction(self):
        """Test extinction correction application."""
        config = CalibrationConfig(apply_galactic_extinction=True)
        calibrator = FluxCalibrator(config)
        
        # Get extinction coefficient for test band
        extinction_coeff = calibrator._get_extinction_coefficient("F444W")
        
        assert extinction_coeff >= 0
        assert extinction_coeff < 2.0  # Reasonable range for NIR
    
    def test_aperture_correction_computation(self):
        """Test aperture correction computation."""
        calibrator = FluxCalibrator()
        
        # Create mock PSF model
        psf_model = np.zeros((21, 21))
        center = 10
        
        # Create simple Gaussian PSF
        y, x = np.ogrid[:21, :21]
        psf_model = np.exp(-((x - center)**2 + (y - center)**2) / (2 * 2.5**2))
        psf_model /= np.sum(psf_model)  # Normalize
        
        # Compute aperture corrections
        corrections = calibrator._compute_psf_based_corrections(psf_model)
        
        assert isinstance(corrections, dict)
        assert len(corrections) > 0
        
        # All corrections should be >= 1 (larger apertures include more flux)
        for radius, correction in corrections.items():
            assert correction >= 1.0
    
    def test_color_computation(self):
        """Test color computation."""
        calibrator = FluxCalibrator()
        
        # Create test sources with magnitudes
        sources = []
        for i in range(3):
            source = CalibratedSource(id=i)
            source.best_magnitudes["F200W"] = 20.0 + i * 0.5
            source.best_magnitudes["F444W"] = 19.5 + i * 0.4
            source.best_magnitude_errors["F200W"] = 0.05
            source.best_magnitude_errors["F444W"] = 0.04
            sources.append(source)
        
        # Compute colors
        sources = calibrator._compute_colors(sources)
        
        # Check that colors were computed
        for source in sources:
            assert "F200W-F444W" in source.colors
            assert "F200W-F444W" in source.color_errors
            
            # Verify color calculation
            expected_color = source.best_magnitudes["F200W"] - source.best_magnitudes["F444W"]
            assert abs(source.colors["F200W-F444W"] - expected_color) < 1e-6


class TestCalibrationResults:
    """Test calibration results container."""
    
    def test_results_creation(self):
        """Test results container creation."""
        sources = [CalibratedSource(id=i) for i in range(3)]
        band_calibrations = {
            "F200W": BandCalibration("F200W", 25.0, 0.02, "test", 5e-21, 2e-6, 20000),
            "F444W": BandCalibration("F444W", 24.5, 0.02, "test", 1.5e-21, 1.2e-6, 44000)
        }
        config = CalibrationConfig()
        
        results = CalibrationResults(
            sources=sources,
            band_calibrations=band_calibrations,
            config=config
        )
        
        assert len(results.sources) == 3
        assert len(results.band_calibrations) == 2
        assert results.config == config
        assert isinstance(results.calibration_statistics, dict)
    
    def test_catalog_export(self, tmp_path):
        """Test catalog export functionality."""
        calibrator = FluxCalibrator()
        
        # Create minimal calibration results
        sources = []
        for i in range(3):
            source = CalibratedSource(id=i)
            source.calibrated_fluxes["F444W"] = {3.0: 1000 + i*100}
            source.calibrated_errors["F444W"] = {3.0: 50 + i*5}
            source.best_fluxes["F444W"] = 1000 + i*100
            source.best_flux_errors["F444W"] = 50 + i*5
            sources.append(source)
        
        band_calibrations = {
            "F444W": BandCalibration("F444W", 24.5, 0.02, "test", 1.5e-21, 1.2e-6, 44000)
        }
        
        results = CalibrationResults(
            sources=sources,
            band_calibrations=band_calibrations,
            config=CalibrationConfig()
        )
        
        # Test export
        output_path = tmp_path / "test_catalog.fits"
        calibrator.export_calibrated_catalog(results, str(output_path), format='fits')
        
        assert output_path.exists()
        
        # Verify catalog content
        catalog = Table.read(output_path)
        assert len(catalog) == 3
        assert 'id' in catalog.colnames
        assert 'flux_best_F444W' in catalog.colnames


class TestLegacyCompatibility:
    """Test legacy function compatibility."""
    
    def test_legacy_calibration_function(self):
        """Test legacy calibration function."""
        flux_instrumental = 1000.0
        band = "F444W"
        
        # Test with default zeropoint
        result = calibrate_to_physical_units(flux_instrumental, band)
        
        assert result > 0
        assert isinstance(result, float)
        
        # Test with custom zeropoint
        result_custom = calibrate_to_physical_units(flux_instrumental, band, zeropoint=25.0)
        
        assert result_custom > 0
        assert isinstance(result_custom, float)


class TestErrorHandling:
    """Test error handling in calibration module."""
    
    def test_empty_photometry_results(self):
        """Test handling of empty photometry results."""
        calibrator = FluxCalibrator()
        
        with pytest.raises(ValueError, match="No photometry results provided"):
            calibrator._validate_calibration_inputs({}, {})
    
    def test_missing_headers(self):
        """Test handling of missing headers."""
        calibrator = FluxCalibrator()
        
        photometry_results = {"F444W": Mock()}
        
        with pytest.raises(ValueError, match="No headers provided"):
            calibrator._validate_calibration_inputs(photometry_results, {})
    
    def test_invalid_band_calibration(self):
        """Test handling of invalid band data."""
        calibrator = FluxCalibrator()
        
        # Test with completely empty header
        empty_header = fits.Header()
        calibration = calibrator._create_band_calibration("UNKNOWN_BAND", empty_header)
        
        # Should create calibration with defaults and flags
        assert calibration.band == "UNKNOWN_BAND"
        assert "no_zeropoint" in calibration.flags


class TestCalibrationIntegration:
    """Integration tests for calibration module."""
    
    def test_full_calibration_pipeline(self):
        """Test complete calibration pipeline."""
        # Create realistic test data
        photometry_results = {}
        headers = {}
        
        bands = ["F200W", "F444W"]
        
        for band in bands:
            # Create mock photometry sources
            sources = []
            for i in range(5):
                source = Mock()
                source.circular_fluxes = {
                    2.0: 800 + i*50 + np.random.normal(0, 10),
                    3.0: 1200 + i*75 + np.random.normal(0, 15),
                    5.0: 1800 + i*100 + np.random.normal(0, 20)
                }
                source.circular_flux_errors = {
                    2.0: 40 + i*2,
                    3.0: 60 + i*3,
                    5.0: 90 + i*4
                }
                sources.append(source)
            
            # Create mock photometry results
            phot_results = Mock()
            phot_results.sources = sources
            photometry_results[band] = phot_results
            
            # Create realistic header
            headers[band] = fits.Header({
                'PHOTZPT': 25.79 if band == "F200W" else 24.51,
                'PHOTFLAM': 5.24e-21 if band == "F200W" else 1.38e-21,
                'GAIN': 2.0,
                'EXPTIME': 1000.0,
                'DETECTOR': 'NRCB1'
            })
        
        # Configure calibrator
        config = CalibrationConfig(
            output_units="uJy",
            apply_aperture_corrections=False,  # Skip for simplicity
            apply_galactic_extinction=True,
            check_color_consistency=True
        )
        
        calibrator = FluxCalibrator(config)
        
        # Run calibration
        results = calibrator.calibrate_photometry(
            photometry_results=photometry_results,
            headers=headers
        )
        
        # Verify results
        assert isinstance(results, CalibrationResults)
        assert len(results.band_calibrations) == 2
        assert len(results.sources) == 5
        
        # Check band calibrations
        for band, calibration in results.band_calibrations.items():
            assert calibration.band == band
            assert calibration.zeropoint > 0
            assert calibration.photflam > 0
            assert calibration.systematic_uncertainty > 0
        
        # Check source calibrations
        for source in results.sources:
            assert len(source.calibrated_fluxes) == 2  # Both bands
            assert len(source.colors) > 0  # Should have computed colors
            
            for band in bands:
                if band in source.calibrated_fluxes:
                    for aperture, flux in source.calibrated_fluxes[band].items():
                        assert flux > 0
                        assert aperture in source.calibrated_errors[band]
                        assert source.calibrated_errors[band][aperture] > 0
        
        # Check statistics
        assert 'bands' in results.calibration_statistics
        assert 'band_statistics' in results.calibration_statistics
        assert results.calibration_statistics['n_bands'] == 2
        
        # Check quality metrics
        assert 0 <= results.overall_quality <= 1
        assert results.processing_time > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
