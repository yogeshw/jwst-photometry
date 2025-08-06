"""
Unit tests for the main module with comprehensive Phase 4 integration testing.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from astropy.io import fits
from astropy.table import Table

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from main import (
        JWSTPhotometryPipeline, process_single_image, process_multiple_images,
        create_comprehensive_pipeline, run_advanced_photometry
    )
    from detection import Source
except ImportError:
    # Handle import fallback
    from src.main import (
        JWSTPhotometryPipeline, process_single_image, process_multiple_images,
        create_comprehensive_pipeline, run_advanced_photometry
    )
    from src.detection import Source


class TestJWSTPhotometryPipeline:
    """Test the main JWST photometry pipeline."""
    
    @pytest.fixture
    def pipeline(self):
        """Create a test pipeline instance."""
        return JWSTPhotometryPipeline()
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample image for testing."""
        image = np.random.normal(100, 10, (100, 100))  # Background
        
        # Add some point sources
        sources = [(30, 40, 1000), (70, 60, 1500), (50, 25, 800)]
        for x, y, flux in sources:
            xx, yy = np.meshgrid(range(x-5, x+6), range(y-5, y+6))
            source = flux * np.exp(-((xx-x)**2 + (yy-y)**2)/(2*2**2))
            image[y-5:y+6, x-5:x+6] += source
        
        return image
    
    @pytest.fixture
    def sample_header(self):
        """Create a sample FITS header."""
        header = fits.Header({
            'INSTRUME': 'NIRCAM',
            'DETECTOR': 'NRCB1',
            'FILTER': 'F444W',
            'PHOTZPT': 24.51,
            'PHOTFLAM': 1.38e-21,
            'GAIN': 2.0,
            'READNOIS': 10.0,
            'EXPTIME': 1000.0,
            'PIXAR_SR': 1.21e-13
        })
        return header
    
    def test_pipeline_initialization(self, pipeline):
        """Test pipeline initialization."""
        assert pipeline is not None
        assert hasattr(pipeline, 'config')
        assert hasattr(pipeline, 'logger')
    
    @patch('main.fits.open')
    def test_load_image(self, mock_fits_open, pipeline, sample_image, sample_header):
        """Test image loading functionality."""
        # Mock FITS file
        mock_hdul = Mock()
        mock_hdu = Mock()
        mock_hdu.data = sample_image
        mock_hdu.header = sample_header
        mock_hdul.__getitem__.return_value = mock_hdu
        mock_hdul.__enter__.return_value = mock_hdul
        mock_hdul.__exit__.return_value = None
        mock_fits_open.return_value = mock_hdul
        
        image_path = "test_image.fits"
        image_data, header = pipeline.load_image(image_path)
        
        assert image_data is not None
        assert image_data.shape == (100, 100)
        assert header is not None
        assert header['FILTER'] == 'F444W'
    
    @patch('main.sep.extract')
    def test_run_detection(self, mock_sep_extract, pipeline, sample_image):
        """Test source detection."""
        # Mock SEP extraction results
        mock_sep_extract.return_value = np.array([
            (30.0, 40.0, 1000.0, 0.1, 0.1, 2.0, 2.0, 0.0),
            (70.0, 60.0, 1500.0, 0.15, 0.12, 2.2, 2.1, 0.1)
        ], dtype=[('x', 'f8'), ('y', 'f8'), ('flux', 'f8'), 
                  ('a', 'f8'), ('b', 'f8'), ('cxx', 'f8'), ('cyy', 'f8'), ('cxy', 'f8')])
        
        sources = pipeline.run_detection(sample_image)
        
        assert sources is not None
        assert len(sources) == 2
        mock_sep_extract.assert_called_once()
    
    def test_run_basic_photometry(self, pipeline, sample_image):
        """Test basic photometry execution."""
        # Create mock sources
        sources = [
            Source(1, 30.0, 40.0, 1000.0, 0.1, 0.1),
            Source(2, 70.0, 60.0, 1500.0, 0.15, 0.12)
        ]
        
        with patch('main.EnhancedAperturePhotometry') as mock_photometry:
            mock_phot_instance = Mock()
            mock_phot_instance.run_photometry.return_value = Mock()
            mock_photometry.return_value = mock_phot_instance
            
            result = pipeline.run_photometry(sample_image, sources, 'F444W')
            
            assert result is not None
            mock_photometry.assert_called_once()
            mock_phot_instance.run_photometry.assert_called_once()
    
    def test_process_single_image_integration(self, pipeline):
        """Test integration of single image processing."""
        with patch.object(pipeline, 'load_image') as mock_load, \
             patch.object(pipeline, 'run_detection') as mock_detect, \
             patch.object(pipeline, 'run_photometry') as mock_phot:
            
            # Setup mocks
            mock_load.return_value = (np.random.random((100, 100)), fits.Header())
            mock_detect.return_value = [Mock(), Mock()]
            mock_phot.return_value = Mock()
            
            result = pipeline.process_single_image("test.fits")
            
            assert result is not None
            mock_load.assert_called_once_with("test.fits")
            mock_detect.assert_called_once()
            mock_phot.assert_called_once()


class TestPhase4Integration:
    """Test Phase 4 enhanced photometry integration."""
    
    @pytest.fixture
    def multi_band_data(self):
        """Create multi-band test data."""
        bands = ['F200W', 'F444W']
        images = {}
        headers = {}
        
        for band in bands:
            # Create image
            image = np.random.normal(100, 10, (100, 100))
            
            # Add sources with band-dependent fluxes
            flux_scale = 1.0 if band == 'F200W' else 0.8
            sources = [(30, 40, 1000*flux_scale), (70, 60, 1500*flux_scale)]
            
            for x, y, flux in sources:
                xx, yy = np.meshgrid(range(x-5, x+6), range(y-5, y+6))
                source = flux * np.exp(-((xx-x)**2 + (yy-y)**2)/(2*2**2))
                image[y-5:y+6, x-5:x+6] += source
            
            images[band] = image
            
            # Create header
            headers[band] = fits.Header({
                'INSTRUME': 'NIRCAM',
                'DETECTOR': 'NRCB1',
                'FILTER': band,
                'PHOTZPT': 25.79 if band == 'F200W' else 24.51,
                'PHOTFLAM': 5.24e-21 if band == 'F200W' else 1.38e-21,
                'GAIN': 2.0,
                'READNOIS': 10.0,
                'EXPTIME': 1000.0
            })
        
        return images, headers
    
    @patch('main.EnhancedAperturePhotometry')
    @patch('main.FluxCalibrator')
    @patch('main.ComprehensiveErrorEstimator')
    def test_enhanced_aperture_photometry_integration(self, mock_error_est, mock_calibrator, mock_photometry, multi_band_data):
        """Test enhanced aperture photometry integration."""
        images, headers = multi_band_data
        
        # Mock photometry results
        mock_phot_instance = Mock()
        mock_phot_results = Mock()
        mock_phot_results.sources = [Mock(), Mock()]
        mock_phot_instance.run_photometry.return_value = mock_phot_results
        mock_photometry.return_value = mock_phot_instance
        
        # Mock calibration results
        mock_cal_instance = Mock()
        mock_cal_results = Mock()
        mock_cal_instance.calibrate_photometry.return_value = mock_cal_results
        mock_calibrator.return_value = mock_cal_instance
        
        # Mock uncertainty estimation
        mock_error_instance = Mock()
        mock_error_results = Mock()
        mock_error_instance.estimate_uncertainties.return_value = mock_error_results
        mock_error_est.return_value = mock_error_instance
        
        # Create enhanced pipeline
        pipeline = create_comprehensive_pipeline()
        
        # Test multi-band processing
        results = pipeline.process_multi_band_images(images, headers)
        
        assert results is not None
        assert 'photometry_results' in results
        assert 'calibration_results' in results
        assert 'uncertainty_results' in results
    
    def test_psf_photometry_integration(self, multi_band_data):
        """Test PSF photometry integration."""
        images, headers = multi_band_data
        
        with patch('main.AdvancedPSFPhotometry') as mock_psf_phot:
            mock_psf_instance = Mock()
            mock_psf_results = Mock()
            mock_psf_instance.run_psf_photometry.return_value = mock_psf_results
            mock_psf_phot.return_value = mock_psf_instance
            
            pipeline = create_comprehensive_pipeline(enable_psf_photometry=True)
            
            # Mock PSF model
            with patch.object(pipeline, '_create_psf_model') as mock_psf_model:
                mock_psf_model.return_value = np.ones((21, 21)) / (21*21)
                
                results = pipeline.process_multi_band_images(images, headers)
                
                assert results is not None
                assert 'psf_photometry_results' in results
    
    def test_calibration_integration(self, multi_band_data):
        """Test flux calibration integration."""
        images, headers = multi_band_data
        
        with patch('main.FluxCalibrator') as mock_calibrator:
            mock_cal_instance = Mock()
            
            # Create realistic calibration results
            from calibration import CalibrationResults, CalibrationConfig, CalibratedSource, BandCalibration
            
            sources = [CalibratedSource(i) for i in range(2)]
            band_calibrations = {
                'F200W': BandCalibration('F200W', 25.79, 0.02, 'header', 5.24e-21, 2e-6, 20000),
                'F444W': BandCalibration('F444W', 24.51, 0.02, 'header', 1.38e-21, 1.2e-6, 44000)
            }
            
            mock_cal_results = CalibrationResults(sources, band_calibrations, CalibrationConfig())
            mock_cal_instance.calibrate_photometry.return_value = mock_cal_results
            mock_calibrator.return_value = mock_cal_instance
            
            pipeline = create_comprehensive_pipeline()
            
            # Mock photometry results
            with patch.object(pipeline, 'run_photometry') as mock_phot:
                mock_phot.return_value = Mock()
                
                results = pipeline.process_multi_band_images(images, headers)
                
                assert results is not None
                assert 'calibration_results' in results
                mock_calibrator.assert_called()
    
    def test_uncertainty_estimation_integration(self, multi_band_data):
        """Test comprehensive uncertainty estimation integration."""
        images, headers = multi_band_data
        
        with patch('main.ComprehensiveErrorEstimator') as mock_error_est:
            mock_error_instance = Mock()
            
            # Create realistic uncertainty results
            from uncertainties import UncertaintyResults, UncertaintyConfig, ErrorBudget, NoiseModel, CorrelationModel
            
            source_uncertainties = {i: ErrorBudget(i, 'F444W', 3.0, []) for i in range(2)}
            noise_models = {'F444W': NoiseModel('NRCB1', 10.0, 0.005, 2.0)}
            correlation_model = CorrelationModel()
            
            mock_uncertainty_results = UncertaintyResults(
                source_uncertainties, noise_models, correlation_model, UncertaintyConfig()
            )
            mock_error_instance.estimate_uncertainties.return_value = mock_uncertainty_results
            mock_error_est.return_value = mock_error_instance
            
            pipeline = create_comprehensive_pipeline()
            
            # Mock earlier processing steps
            with patch.object(pipeline, 'run_photometry') as mock_phot, \
                 patch('main.FluxCalibrator') as mock_cal:
                
                mock_phot.return_value = Mock()
                mock_cal_instance = Mock()
                mock_cal_instance.calibrate_photometry.return_value = Mock()
                mock_cal.return_value = mock_cal_instance
                
                results = pipeline.process_multi_band_images(images, headers)
                
                assert results is not None
                assert 'uncertainty_results' in results
                mock_error_est.assert_called()


class TestAdvancedProcessingFunctions:
    """Test advanced processing functions."""
    
    def test_run_advanced_photometry_function(self):
        """Test run_advanced_photometry function."""
        image_paths = {'F200W': 'test_f200w.fits', 'F444W': 'test_f444w.fits'}
        
        with patch('main.create_comprehensive_pipeline') as mock_create_pipeline:
            mock_pipeline = Mock()
            mock_pipeline.process_multi_band_images.return_value = {'results': 'test'}
            mock_create_pipeline.return_value = mock_pipeline
            
            with patch('main.fits.open') as mock_fits:
                # Mock FITS file opening
                mock_hdul = Mock()
                mock_hdu = Mock()
                mock_hdu.data = np.random.normal(100, 10, (100, 100))
                mock_hdu.header = fits.Header({'FILTER': 'F444W'})
                mock_hdul.__getitem__.return_value = mock_hdu
                mock_hdul.__enter__.return_value = mock_hdul
                mock_hdul.__exit__.return_value = None
                mock_fits.return_value = mock_hdul
                
                results = run_advanced_photometry(image_paths)
                
                assert results is not None
                mock_create_pipeline.assert_called()
    
    def test_comprehensive_pipeline_creation(self):
        """Test comprehensive pipeline creation."""
        pipeline = create_comprehensive_pipeline(
            enable_psf_photometry=True,
            enable_advanced_calibration=True,
            enable_uncertainty_estimation=True
        )
        
        assert pipeline is not None
        assert hasattr(pipeline, 'config')
        assert hasattr(pipeline, 'process_multi_band_images')


class TestLegacyCompatibility:
    """Test legacy function compatibility."""
    
    @patch('main.JWSTPhotometryPipeline')
    def test_process_single_image_function(self, mock_pipeline_class):
        """Test process_single_image function."""
        mock_pipeline = Mock()
        mock_pipeline.process_single_image.return_value = Mock()
        mock_pipeline_class.return_value = mock_pipeline
        
        result = process_single_image("test.fits")
        
        assert result is not None
        mock_pipeline.process_single_image.assert_called_once_with("test.fits")
    
    @patch('main.JWSTPhotometryPipeline')
    def test_process_multiple_images_function(self, mock_pipeline_class):
        """Test process_multiple_images function."""
        mock_pipeline = Mock()
        mock_pipeline.process_single_image.return_value = Mock()
        mock_pipeline_class.return_value = mock_pipeline
        
        image_paths = ["test1.fits", "test2.fits"]
        results = process_multiple_images(image_paths)
        
        assert results is not None
        assert len(results) == 2


class TestErrorHandling:
    """Test error handling in main module."""
    
    def test_missing_image_file(self):
        """Test handling of missing image files."""
        pipeline = JWSTPhotometryPipeline()
        
        with pytest.raises(FileNotFoundError):
            pipeline.load_image("nonexistent_file.fits")
    
    def test_invalid_image_data(self):
        """Test handling of invalid image data."""
        pipeline = JWSTPhotometryPipeline()
        
        # Test with non-numeric data
        with pytest.raises(ValueError):
            pipeline.run_detection("not_an_array")
    
    def test_empty_source_list(self):
        """Test handling of empty source lists."""
        pipeline = JWSTPhotometryPipeline()
        image = np.random.normal(100, 10, (100, 100))
        
        with patch('main.sep.extract') as mock_extract:
            mock_extract.return_value = np.array([], dtype=[
                ('x', 'f8'), ('y', 'f8'), ('flux', 'f8'),
                ('a', 'f8'), ('b', 'f8'), ('cxx', 'f8'), ('cyy', 'f8'), ('cxy', 'f8')
            ])
            
            sources = pipeline.run_detection(image)
            assert len(sources) == 0


class TestConfigurationManagement:
    """Test configuration management."""
    
    def test_default_configuration(self):
        """Test default configuration loading."""
        pipeline = JWSTPhotometryPipeline()
        
        assert pipeline.config is not None
        assert hasattr(pipeline.config, 'detection')
        assert hasattr(pipeline.config, 'photometry')
    
    def test_custom_configuration(self):
        """Test custom configuration."""
        custom_config = {
            'detection': {'threshold': 3.0},
            'photometry': {'aperture_radii': [2.0, 4.0, 6.0]}
        }
        
        pipeline = JWSTPhotometryPipeline(config=custom_config)
        
        assert pipeline.config['detection']['threshold'] == 3.0
        assert pipeline.config['photometry']['aperture_radii'] == [2.0, 4.0, 6.0]


class TestMultiBandProcessing:
    """Test multi-band processing capabilities."""
    
    @pytest.fixture
    def multi_band_images(self):
        """Create multi-band test images."""
        bands = ['F200W', 'F356W', 'F444W']
        images = {}
        
        for band in bands:
            image = np.random.normal(100, 10, (100, 100))
            # Add band-specific sources
            flux_scale = {'F200W': 1.2, 'F356W': 1.0, 'F444W': 0.8}[band]
            sources = [(30, 40, 1000*flux_scale), (70, 60, 1500*flux_scale)]
            
            for x, y, flux in sources:
                xx, yy = np.meshgrid(range(x-5, x+6), range(y-5, y+6))
                source = flux * np.exp(-((xx-x)**2 + (yy-y)**2)/(2*2**2))
                image[y-5:y+6, x-5:x+6] += source
            
            images[band] = image
        
        return images
    
    def test_multi_band_coordination(self, multi_band_images):
        """Test multi-band processing coordination."""
        pipeline = create_comprehensive_pipeline()
        
        # Mock all processing steps
        with patch.object(pipeline, 'run_detection') as mock_detect, \
             patch.object(pipeline, 'run_photometry') as mock_phot, \
             patch('main.FluxCalibrator') as mock_cal, \
             patch('main.ComprehensiveErrorEstimator') as mock_error:
            
            # Setup mocks
            mock_detect.return_value = [Mock(), Mock()]
            mock_phot.return_value = Mock()
            
            mock_cal_instance = Mock()
            mock_cal_instance.calibrate_photometry.return_value = Mock()
            mock_cal.return_value = mock_cal_instance
            
            mock_error_instance = Mock()
            mock_error_instance.estimate_uncertainties.return_value = Mock()
            mock_error.return_value = mock_error_instance
            
            # Create headers
            headers = {}
            for band in multi_band_images.keys():
                headers[band] = fits.Header({'FILTER': band, 'PHOTZPT': 25.0})
            
            results = pipeline.process_multi_band_images(multi_band_images, headers)
            
            assert results is not None
            assert len(results) > 0
            
            # Verify that detection was called for each band
            assert mock_detect.call_count == len(multi_band_images)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
