"""
Unit tests for the psf_photometry module (Phase 3.3).
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from astropy.io import fits
from astropy.table import Table

# Import PSF photometry modules
try:
    from src.psf_photometry import (
        PSFPhotometryConfig, PSFPhotometryResults, AdvancedPSFPhotometry,
        PSFSource, BlendGroup, ModelComponent
    )
except ImportError:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
    
    from psf_photometry import (
        PSFPhotometryConfig, PSFPhotometryResults, AdvancedPSFPhotometry,
        PSFSource, BlendGroup, ModelComponent
    )


class TestPSFPhotometryConfig:
    """Test PSF photometry configuration."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = PSFPhotometryConfig()
        
        assert config.fit_shape is True
        assert config.simultaneous_fitting is True
        assert config.blend_detection is True
        assert config.crowding_threshold == 3.0
        assert config.background_fitting is True
        assert config.min_separation == 2.0
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = PSFPhotometryConfig(
            fit_shape=False,
            simultaneous_fitting=False,
            crowding_threshold=5.0,
            max_iterations=200,
            convergence_tolerance=1e-6
        )
        
        assert config.fit_shape is False
        assert config.simultaneous_fitting is False
        assert config.crowding_threshold == 5.0
        assert config.max_iterations == 200
        assert config.convergence_tolerance == 1e-6


class TestPSFSource:
    """Test PSF source container."""
    
    def test_source_creation(self):
        """Test basic PSF source creation."""
        source = PSFSource(
            id=1,
            x=50.0,
            y=30.0,
            flux=1000.0
        )
        
        assert source.id == 1
        assert source.x == 50.0
        assert source.y == 30.0
        assert source.flux == 1000.0
        assert source.flux_error is None
        assert isinstance(source.fitted_parameters, dict)
        assert isinstance(source.covariance_matrix, np.ndarray)
    
    def test_source_with_uncertainties(self):
        """Test source with uncertainty estimates."""
        source = PSFSource(
            id=1,
            x=50.0,
            y=30.0,
            flux=1000.0,
            flux_error=50.0,
            x_error=0.1,
            y_error=0.1
        )
        
        assert source.flux_error == 50.0
        assert source.x_error == 0.1
        assert source.y_error == 0.1
        
        # Test signal-to-noise calculation
        snr = source.signal_to_noise()
        expected_snr = 1000.0 / 50.0
        assert abs(snr - expected_snr) < 1e-6
    
    def test_magnitude_conversion(self):
        """Test magnitude conversion."""
        source = PSFSource(
            id=1,
            x=50.0,
            y=30.0,
            flux=1000.0,
            flux_error=50.0
        )
        
        zeropoint = 25.0
        mag, mag_error = source.magnitude(zeropoint)
        
        expected_mag = -2.5 * np.log10(1000.0) + zeropoint
        expected_mag_error = 2.5 / np.log(10) * 50.0 / 1000.0
        
        assert abs(mag - expected_mag) < 1e-6
        assert abs(mag_error - expected_mag_error) < 1e-6


class TestModelComponent:
    """Test PSF model component."""
    
    def test_component_creation(self):
        """Test model component creation."""
        component = ModelComponent(
            source_id=1,
            component_type="psf",
            parameters={'x': 50.0, 'y': 30.0, 'flux': 1000.0}
        )
        
        assert component.source_id == 1
        assert component.component_type == "psf"
        assert component.parameters['flux'] == 1000.0
        assert isinstance(component.parameter_errors, dict)
    
    def test_component_evaluation(self):
        """Test model component evaluation."""
        component = ModelComponent(
            source_id=1,
            component_type="psf",
            parameters={'x': 10.0, 'y': 10.0, 'flux': 1000.0, 'sigma': 2.0}
        )
        
        # Create coordinate grids
        x = np.arange(21)
        y = np.arange(21)
        xx, yy = np.meshgrid(x, y)
        
        # Evaluate Gaussian model
        model = component.evaluate_gaussian(xx, yy)
        
        assert model.shape == (21, 21)
        assert model[10, 10] > model[0, 0]  # Peak at center
        assert np.sum(model) > 0


class TestBlendGroup:
    """Test blend group container."""
    
    def test_blend_group_creation(self):
        """Test blend group creation."""
        sources = [
            PSFSource(1, 50.0, 30.0, 1000.0),
            PSFSource(2, 52.0, 31.0, 800.0),
            PSFSource(3, 49.0, 32.0, 600.0)
        ]
        
        blend = BlendGroup(
            group_id=1,
            sources=sources,
            bounding_box=(45, 55, 25, 35)
        )
        
        assert blend.group_id == 1
        assert len(blend.sources) == 3
        assert blend.bounding_box == (45, 55, 25, 35)
        assert blend.n_sources == 3
    
    def test_blend_analysis(self):
        """Test blend analysis methods."""
        sources = [
            PSFSource(1, 50.0, 30.0, 1000.0),
            PSFSource(2, 52.0, 31.0, 800.0)
        ]
        
        blend = BlendGroup(1, sources, (45, 55, 25, 35))
        
        # Test separation calculation
        separation = blend.calculate_separations()
        expected_sep = np.sqrt((52.0 - 50.0)**2 + (31.0 - 30.0)**2)
        assert abs(separation[0][1] - expected_sep) < 1e-6
        
        # Test flux ratio
        flux_ratios = blend.calculate_flux_ratios()
        expected_ratio = 800.0 / 1000.0
        assert abs(flux_ratios[0][1] - expected_ratio) < 1e-6


class TestAdvancedPSFPhotometry:
    """Test the advanced PSF photometry class."""
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample image with known sources."""
        image = np.random.normal(100, 5, (100, 100))  # Background
        
        # Add point sources
        sources = [
            (30, 40, 5000),  # x, y, flux
            (70, 60, 3000),
            (50, 25, 2000)
        ]
        
        for x, y, flux in sources:
            # Create simple Gaussian PSF
            xx, yy = np.meshgrid(range(x-10, x+11), range(y-10, y+11))
            psf = flux * np.exp(-((xx-x)**2 + (yy-y)**2)/(2*2.5**2))
            
            # Add to image with bounds checking
            x1, x2 = max(0, x-10), min(100, x+11)
            y1, y2 = max(0, y-10), min(100, y+11)
            px1, px2 = max(0, 10-(x-x1)), min(21, 10+(x2-x))
            py1, py2 = max(0, 10-(y-y1)), min(21, 10+(y2-y))
            
            image[y1:y2, x1:x2] += psf[py1:py2, px1:px2]
        
        return image, sources
    
    @pytest.fixture
    def sample_psf_model(self):
        """Create a sample PSF model."""
        psf = np.zeros((21, 21))
        center = 10
        
        # Create normalized Gaussian PSF
        y, x = np.ogrid[:21, :21]
        psf = np.exp(-((x - center)**2 + (y - center)**2) / (2 * 2.5**2))
        psf /= np.sum(psf)
        
        return psf
    
    @pytest.fixture
    def sample_detection_table(self):
        """Create sample detection table."""
        detections = Table()
        detections['x'] = [30.0, 70.0, 50.0]
        detections['y'] = [40.0, 60.0, 25.0]
        detections['flux'] = [5000.0, 3000.0, 2000.0]
        detections['peak'] = [500.0, 300.0, 200.0]
        detections['a'] = [2.5, 2.3, 2.4]
        detections['b'] = [2.3, 2.1, 2.2]
        detections['theta'] = [0.1, 0.2, 0.0]
        
        return detections
    
    def test_photometry_initialization(self):
        """Test PSF photometry initialization."""
        config = PSFPhotometryConfig()
        photometry = AdvancedPSFPhotometry(config)
        
        assert photometry.config == config
        assert hasattr(photometry, 'logger')
        assert hasattr(photometry, 'fitted_models')
        assert isinstance(photometry.fitted_models, dict)
    
    def test_detection_preprocessing(self, sample_detection_table):
        """Test detection table preprocessing."""
        photometry = AdvancedPSFPhotometry()
        
        processed = photometry._preprocess_detections(sample_detection_table)
        
        assert len(processed) == len(sample_detection_table)
        assert 'x' in processed.colnames
        assert 'y' in processed.colnames
        assert 'flux' in processed.colnames
        
        # Should have computed additional parameters
        assert np.all(processed['x'] >= 0)
        assert np.all(processed['y'] >= 0)
    
    def test_blend_detection(self, sample_detection_table):
        """Test blend detection algorithm."""
        config = PSFPhotometryConfig(crowding_threshold=5.0)
        photometry = AdvancedPSFPhotometry(config)
        
        blend_groups = photometry._detect_blends(sample_detection_table)
        
        assert isinstance(blend_groups, list)
        
        # Check that nearby sources are grouped
        for group in blend_groups:
            assert isinstance(group, BlendGroup)
            assert group.n_sources >= 1
    
    def test_psf_model_preparation(self, sample_psf_model):
        """Test PSF model preparation."""
        photometry = AdvancedPSFPhotometry()
        
        # Test PSF normalization
        normalized_psf = photometry._prepare_psf_model(sample_psf_model)
        
        assert normalized_psf.shape == sample_psf_model.shape
        assert abs(np.sum(normalized_psf) - 1.0) < 1e-6  # Should be normalized
        assert np.all(normalized_psf >= 0)  # Should be non-negative
    
    def test_single_source_fitting(self, sample_image, sample_psf_model):
        """Test single source PSF fitting."""
        photometry = AdvancedPSFPhotometry()
        
        image, true_sources = sample_image
        psf_model = sample_psf_model
        
        # Fit first source
        x_init, y_init, flux_init = true_sources[0]
        
        result = photometry._fit_single_source(
            image, psf_model, x_init, y_init, flux_init
        )
        
        assert isinstance(result, PSFSource)
        assert result.x is not None
        assert result.y is not None
        assert result.flux > 0
        assert result.flux_error > 0
        
        # Should be close to true position
        assert abs(result.x - x_init) < 2.0
        assert abs(result.y - y_init) < 2.0
    
    @patch('src.psf_photometry.minimize')
    def test_simultaneous_fitting(self, mock_minimize, sample_image, sample_psf_model):
        """Test simultaneous fitting of multiple sources."""
        # Configure mock optimization result
        mock_result = Mock()
        mock_result.success = True
        mock_result.x = [30.0, 40.0, 5000.0, 70.0, 60.0, 3000.0]  # x1, y1, f1, x2, y2, f2
        mock_result.fun = 100.0
        mock_result.hess_inv = np.eye(6) * 0.01  # Mock covariance
        mock_minimize.return_value = mock_result
        
        config = PSFPhotometryConfig(simultaneous_fitting=True)
        photometry = AdvancedPSFPhotometry(config)
        
        image, true_sources = sample_image
        psf_model = sample_psf_model
        
        # Create blend group with first two sources
        sources = [
            PSFSource(1, true_sources[0][0], true_sources[0][1], true_sources[0][2]),
            PSFSource(2, true_sources[1][0], true_sources[1][1], true_sources[1][2])
        ]
        blend_group = BlendGroup(1, sources, (20, 80, 30, 70))
        
        fitted_sources = photometry._fit_blend_group(image, psf_model, blend_group)
        
        assert len(fitted_sources) == 2
        assert all(isinstance(src, PSFSource) for src in fitted_sources)
        assert all(src.fitted for src in fitted_sources)
    
    def test_background_estimation(self, sample_image):
        """Test background estimation for PSF fitting."""
        photometry = AdvancedPSFPhotometry()
        
        image, _ = sample_image
        
        # Estimate background around a source region
        x_center, y_center = 50, 50
        region_size = 20
        
        background = photometry._estimate_local_background(
            image, x_center, y_center, region_size
        )
        
        assert isinstance(background, dict)
        assert 'median' in background
        assert 'std' in background
        assert 'mesh_size' in background
        assert background['median'] > 0
        assert background['std'] > 0
    
    def test_quality_assessment(self, sample_image, sample_psf_model):
        """Test fit quality assessment."""
        photometry = AdvancedPSFPhotometry()
        
        image, true_sources = sample_image
        psf_model = sample_psf_model
        
        # Create a fitted source
        source = PSFSource(
            id=1,
            x=true_sources[0][0],
            y=true_sources[0][1],
            flux=true_sources[0][2],
            flux_error=100.0
        )
        source.fitted = True
        source.chi_squared = 1.2
        source.degrees_freedom = 100
        
        quality = photometry._assess_fit_quality(source, image, psf_model)
        
        assert isinstance(quality, dict)
        assert 'chi_squared' in quality
        assert 'reduced_chi_squared' in quality
        assert 'residual_rms' in quality
        assert quality['chi_squared'] > 0
        assert quality['reduced_chi_squared'] > 0
    
    def test_full_psf_photometry(self, sample_image, sample_psf_model, sample_detection_table):
        """Test complete PSF photometry pipeline."""
        config = PSFPhotometryConfig(
            simultaneous_fitting=False,  # Simpler for testing
            blend_detection=True,
            background_fitting=True
        )
        photometry = AdvancedPSFPhotometry(config)
        
        image, true_sources = sample_image
        
        # Run PSF photometry
        results = photometry.run_psf_photometry(
            image=image,
            psf_model=sample_psf_model,
            detections=sample_detection_table
        )
        
        assert isinstance(results, PSFPhotometryResults)
        assert len(results.sources) == len(sample_detection_table)
        assert len(results.blend_groups) >= 0
        
        # Check source results
        for source in results.sources:
            assert isinstance(source, PSFSource)
            assert source.x is not None
            assert source.y is not None
            assert source.flux > 0
            assert source.flux_error > 0
        
        # Check processing metadata
        assert 'n_sources' in results.processing_metadata
        assert 'n_blends' in results.processing_metadata
        assert 'processing_time' in results.processing_metadata
        assert results.processing_metadata['n_sources'] == len(sample_detection_table)


class TestPSFPhotometryResults:
    """Test PSF photometry results container."""
    
    def test_results_creation(self):
        """Test results container creation."""
        sources = [
            PSFSource(1, 30.0, 40.0, 1000.0, 50.0),
            PSFSource(2, 70.0, 60.0, 800.0, 40.0)
        ]
        
        blend_groups = [
            BlendGroup(1, sources[:1], (25, 35, 35, 45))
        ]
        
        config = PSFPhotometryConfig()
        
        results = PSFPhotometryResults(
            sources=sources,
            blend_groups=blend_groups,
            config=config
        )
        
        assert len(results.sources) == 2
        assert len(results.blend_groups) == 1
        assert results.config == config
        assert isinstance(results.processing_metadata, dict)
    
    def test_catalog_export(self, tmp_path):
        """Test catalog export functionality."""
        sources = [
            PSFSource(1, 30.0, 40.0, 1000.0, 50.0),
            PSFSource(2, 70.0, 60.0, 800.0, 40.0)
        ]
        
        for source in sources:
            source.fitted = True
            source.chi_squared = 1.1
            source.degrees_freedom = 50
        
        results = PSFPhotometryResults(
            sources=sources,
            blend_groups=[],
            config=PSFPhotometryConfig()
        )
        
        # Test export to FITS
        output_path = tmp_path / "psf_catalog.fits"
        results.export_catalog(str(output_path), format='fits')
        
        assert output_path.exists()
        
        # Verify catalog content
        catalog = Table.read(output_path)
        assert len(catalog) == 2
        assert 'id' in catalog.colnames
        assert 'x' in catalog.colnames
        assert 'y' in catalog.colnames
        assert 'flux' in catalog.colnames
        assert 'flux_error' in catalog.colnames
    
    def test_statistics_computation(self):
        """Test photometry statistics computation."""
        sources = []
        for i in range(10):
            source = PSFSource(i, 30.0 + i, 40.0 + i, 1000.0 + i*100, 50.0 + i*5)
            source.fitted = True
            source.chi_squared = 1.0 + i*0.1
            sources.append(source)
        
        results = PSFPhotometryResults(
            sources=sources,
            blend_groups=[],
            config=PSFPhotometryConfig()
        )
        
        stats = results.compute_statistics()
        
        assert 'n_sources' in stats
        assert 'flux_statistics' in stats
        assert 'fit_quality' in stats
        assert stats['n_sources'] == 10
        
        flux_stats = stats['flux_statistics']
        assert 'mean' in flux_stats
        assert 'median' in flux_stats
        assert 'std' in flux_stats
        assert flux_stats['mean'] > 0


class TestMultiBandPSFPhotometry:
    """Test multi-band PSF photometry capabilities."""
    
    @pytest.fixture
    def multi_band_data(self):
        """Create multi-band test data."""
        bands = ['F200W', 'F444W']
        images = {}
        psf_models = {}
        
        for band in bands:
            # Create image with background
            image = np.random.normal(100, 5, (100, 100))
            
            # Add sources with band-dependent fluxes
            sources = [
                (30, 40, 5000 if band == 'F200W' else 3000),
                (70, 60, 3000 if band == 'F200W' else 4000)
            ]
            
            for x, y, flux in sources:
                xx, yy = np.meshgrid(range(x-10, x+11), range(y-10, y+11))
                psf = flux * np.exp(-((xx-x)**2 + (yy-y)**2)/(2*2.5**2))
                
                x1, x2 = max(0, x-10), min(100, x+11)
                y1, y2 = max(0, y-10), min(100, y+11)
                px1, px2 = max(0, 10-(x-x1)), min(21, 10+(x2-x))
                py1, py2 = max(0, 10-(y-y1)), min(21, 10+(y2-y))
                
                image[y1:y2, x1:x2] += psf[py1:py2, px1:px2]
            
            images[band] = image
            
            # Create PSF model
            psf = np.zeros((21, 21))
            center = 10
            y, x = np.ogrid[:21, :21]
            psf = np.exp(-((x - center)**2 + (y - center)**2) / (2 * 2.5**2))
            psf /= np.sum(psf)
            psf_models[band] = psf
        
        return images, psf_models, sources
    
    def test_multi_band_simultaneous_fitting(self, multi_band_data):
        """Test simultaneous fitting across multiple bands."""
        images, psf_models, true_sources = multi_band_data
        
        config = PSFPhotometryConfig(
            simultaneous_fitting=True,
            multi_band_fitting=True
        )
        photometry = AdvancedPSFPhotometry(config)
        
        # Create detection table
        detections = Table()
        detections['x'] = [30.0, 70.0]
        detections['y'] = [40.0, 60.0]
        detections['flux'] = [4000.0, 3500.0]  # Average flux
        
        # Run multi-band photometry
        results = photometry.run_multi_band_photometry(
            images=images,
            psf_models=psf_models,
            detections=detections
        )
        
        assert isinstance(results, dict)
        assert len(results) == 2  # Two bands
        
        for band, band_results in results.items():
            assert isinstance(band_results, PSFPhotometryResults)
            assert len(band_results.sources) == 2
            
            # Check that sources have consistent positions across bands
            for i, source in enumerate(band_results.sources):
                assert abs(source.x - true_sources[i][0]) < 3.0
                assert abs(source.y - true_sources[i][1]) < 3.0


class TestErrorHandling:
    """Test error handling in PSF photometry module."""
    
    def test_invalid_psf_model(self):
        """Test handling of invalid PSF model."""
        photometry = AdvancedPSFPhotometry()
        
        # Test with negative PSF values
        invalid_psf = np.array([[-1, 0, 1], [0, 2, 0], [1, 0, -1]])
        
        with pytest.raises(ValueError, match="PSF model contains negative values"):
            photometry._prepare_psf_model(invalid_psf)
    
    def test_empty_detection_table(self):
        """Test handling of empty detection table."""
        photometry = AdvancedPSFPhotometry()
        
        empty_table = Table()
        
        with pytest.raises(ValueError, match="No detections provided"):
            photometry._preprocess_detections(empty_table)
    
    def test_mismatched_image_psf_dimensions(self):
        """Test handling of mismatched image and PSF dimensions."""
        photometry = AdvancedPSFPhotometry()
        
        image = np.zeros((100, 100))
        psf = np.zeros((25, 25))  # Different from expected 21x21
        
        detections = Table()
        detections['x'] = [50.0]
        detections['y'] = [50.0]
        detections['flux'] = [1000.0]
        
        # Should handle PSF size gracefully
        results = photometry.run_psf_photometry(image, psf, detections)
        assert isinstance(results, PSFPhotometryResults)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
