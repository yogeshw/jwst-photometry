"""
Unit tests for the uncertainties module (Phase 4.3).
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from astropy.io import fits
from astropy.table import Table

# Import uncertainties modules
try:
    from src.uncertainties import (
        UncertaintyConfig, NoiseModel, ErrorContribution, CorrelationModel,
        ComprehensiveErrorEstimator, UncertaintyResults, ErrorBudget
    )
except ImportError:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
    
    from uncertainties import (
        UncertaintyConfig, NoiseModel, ErrorContribution, CorrelationModel,
        ComprehensiveErrorEstimator, UncertaintyResults, ErrorBudget
    )


class TestUncertaintyConfig:
    """Test uncertainty estimation configuration."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = UncertaintyConfig()
        
        assert config.include_poisson_noise is True
        assert config.include_read_noise is True
        assert config.include_background_uncertainty is True
        assert config.monte_carlo_iterations == 1000
        assert config.confidence_level == 0.68
        assert config.readout_noise == 10.0
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = UncertaintyConfig(
            monte_carlo_iterations=5000,
            confidence_level=0.95,
            include_correlated_noise=False,
            spatial_correlation_scale=2.5
        )
        
        assert config.monte_carlo_iterations == 5000
        assert config.confidence_level == 0.95
        assert config.include_correlated_noise is False
        assert config.spatial_correlation_scale == 2.5


class TestNoiseModel:
    """Test noise model container."""
    
    def test_noise_model_creation(self):
        """Test basic noise model creation."""
        model = NoiseModel(
            detector="NRCB1",
            readout_noise=10.0,
            dark_current=0.005,
            gain=2.0
        )
        
        assert model.detector == "NRCB1"
        assert model.readout_noise == 10.0
        assert model.dark_current == 0.005
        assert model.gain == 2.0
        assert isinstance(model.components, dict)
    
    def test_total_noise_calculation(self):
        """Test total noise computation."""
        model = NoiseModel(
            detector="NRCB1",
            readout_noise=10.0,
            dark_current=0.005,
            gain=2.0
        )
        
        # Add noise components
        model.components['poisson'] = 15.0
        model.components['background'] = 5.0
        model.components['systematic'] = 3.0
        
        # Calculate total noise
        total_noise = np.sqrt(
            model.readout_noise**2 + 
            model.components['poisson']**2 +
            model.components['background']**2 +
            model.components['systematic']**2
        )
        
        assert abs(model.calculate_total_noise() - total_noise) < 1e-6


class TestErrorContribution:
    """Test error contribution tracking."""
    
    def test_error_contribution_creation(self):
        """Test error contribution creation."""
        contribution = ErrorContribution(
            source="poisson",
            magnitude=15.0,
            uncertainty=0.5,
            correlation_scale=0.0
        )
        
        assert contribution.source == "poisson"
        assert contribution.magnitude == 15.0
        assert contribution.uncertainty == 0.5
        assert contribution.correlation_scale == 0.0
        assert isinstance(contribution.metadata, dict)
    
    def test_fractional_uncertainty(self):
        """Test fractional uncertainty calculation."""
        contribution = ErrorContribution(
            source="background",
            magnitude=1000.0,
            uncertainty=50.0,
            correlation_scale=1.0
        )
        
        expected_fraction = 50.0 / 1000.0
        assert abs(contribution.fractional_uncertainty() - expected_fraction) < 1e-6


class TestCorrelationModel:
    """Test correlation model for error propagation."""
    
    def test_correlation_model_creation(self):
        """Test correlation model creation."""
        model = CorrelationModel(
            spatial_scale=2.0,
            temporal_scale=0.5,
            band_correlations={'F200W-F444W': 0.3}
        )
        
        assert model.spatial_scale == 2.0
        assert model.temporal_scale == 0.5
        assert model.band_correlations['F200W-F444W'] == 0.3
        assert isinstance(model.correlation_matrix, np.ndarray)
    
    def test_spatial_correlation_calculation(self):
        """Test spatial correlation computation."""
        model = CorrelationModel(spatial_scale=2.0)
        
        # Test correlation between nearby pixels
        distance = 1.0
        correlation = model.spatial_correlation(distance)
        
        assert 0 <= correlation <= 1
        assert correlation > 0.5  # Should be correlated at short distances
        
        # Test correlation at large distances
        large_distance = 10.0
        large_correlation = model.spatial_correlation(large_distance)
        
        assert large_correlation < correlation  # Should decrease with distance
    
    def test_band_correlation_matrix(self):
        """Test band correlation matrix construction."""
        bands = ['F200W', 'F444W', 'F356W']
        correlations = {
            'F200W-F444W': 0.3,
            'F200W-F356W': 0.5,
            'F444W-F356W': 0.8
        }
        
        model = CorrelationModel(band_correlations=correlations)
        matrix = model.build_band_correlation_matrix(bands)
        
        assert matrix.shape == (3, 3)
        assert matrix[0, 0] == 1.0  # Self-correlation
        assert matrix[1, 1] == 1.0
        assert matrix[2, 2] == 1.0
        assert matrix[0, 1] == matrix[1, 0] == 0.3  # Symmetric
        assert matrix[0, 2] == matrix[2, 0] == 0.5
        assert matrix[1, 2] == matrix[2, 1] == 0.8


class TestComprehensiveErrorEstimator:
    """Test the comprehensive error estimator."""
    
    @pytest.fixture
    def sample_calibrated_results(self):
        """Create sample calibrated results."""
        # Import calibration module for creating test data
        try:
            from src.calibration import CalibratedSource, CalibrationResults, CalibrationConfig, BandCalibration
        except ImportError:
            from calibration import CalibratedSource, CalibrationResults, CalibrationConfig, BandCalibration
        
        sources = []
        for i in range(5):
            source = CalibratedSource(id=i)
            
            # Add calibrated fluxes for multiple bands
            for band in ['F200W', 'F444W']:
                source.calibrated_fluxes[band] = {3.0: 1000 + i*100, 5.0: 1500 + i*150}
                source.calibrated_errors[band] = {3.0: 50 + i*5, 5.0: 75 + i*7.5}
                source.best_fluxes[band] = 1500 + i*150
                source.best_flux_errors[band] = 75 + i*7.5
                source.best_magnitudes[band] = 20.0 + i*0.1
                source.best_magnitude_errors[band] = 0.05 + i*0.001
            
            sources.append(source)
        
        # Create band calibrations
        band_calibrations = {}
        for band in ['F200W', 'F444W']:
            band_calibrations[band] = BandCalibration(
                band=band,
                zeropoint=25.0 if band == 'F200W' else 24.5,
                zeropoint_error=0.02,
                zeropoint_source="test",
                photflam=5e-21 if band == 'F200W' else 1.5e-21,
                photfnu=2e-6 if band == 'F200W' else 1.2e-6,
                photplam=20000 if band == 'F200W' else 44000
            )
        
        results = CalibrationResults(
            sources=sources,
            band_calibrations=band_calibrations,
            config=CalibrationConfig()
        )
        
        return results
    
    @pytest.fixture
    def sample_images(self):
        """Create sample images."""
        images = {}
        for band in ['F200W', 'F444W']:
            # Create realistic noise image
            image = np.random.normal(100, 10, (100, 100))  # Background ~ 100
            
            # Add some sources
            for i in range(5):
                x, y = 20 + i*15, 30 + i*10
                if x < 95 and y < 95:
                    xx, yy = np.meshgrid(range(x-5, x+6), range(y-5, y+6))
                    source = 1000 * np.exp(-((xx-x)**2 + (yy-y)**2)/(2*2**2))
                    image[y-5:y+6, x-5:x+6] += source
            
            images[band] = image
        
        return images
    
    @pytest.fixture
    def sample_headers(self):
        """Create sample FITS headers."""
        headers = {}
        for band in ['F200W', 'F444W']:
            headers[band] = fits.Header({
                'GAIN': 2.0,
                'READNOIS': 10.0,
                'DARK': 0.005,
                'EXPTIME': 1000.0,
                'DETECTOR': 'NRCB1'
            })
        return headers
    
    def test_estimator_initialization(self):
        """Test error estimator initialization."""
        config = UncertaintyConfig()
        estimator = ComprehensiveErrorEstimator(config)
        
        assert estimator.config == config
        assert hasattr(estimator, 'logger')
        assert hasattr(estimator, 'noise_models')
        assert hasattr(estimator, 'correlation_model')
    
    def test_noise_model_creation(self, sample_headers):
        """Test noise model creation from headers."""
        estimator = ComprehensiveErrorEstimator()
        
        band = 'F444W'
        header = sample_headers[band]
        
        noise_model = estimator._create_noise_model(band, header)
        
        assert noise_model.detector == header['DETECTOR']
        assert noise_model.readout_noise == header['READNOIS']
        assert noise_model.dark_current == header['DARK']
        assert noise_model.gain == header['GAIN']
    
    def test_poisson_noise_estimation(self, sample_images):
        """Test Poisson noise estimation."""
        estimator = ComprehensiveErrorEstimator()
        
        image = sample_images['F444W']
        gain = 2.0
        
        poisson_noise = estimator._estimate_poisson_noise(image, gain)
        
        assert poisson_noise.shape == image.shape
        assert np.all(poisson_noise > 0)
        
        # Verify Poisson scaling: noise ~ sqrt(signal)
        high_signal_region = image > 500
        low_signal_region = image < 200
        
        if np.any(high_signal_region) and np.any(low_signal_region):
            high_noise = np.mean(poisson_noise[high_signal_region])
            low_noise = np.mean(poisson_noise[low_signal_region])
            assert high_noise > low_noise
    
    def test_background_uncertainty_estimation(self, sample_images):
        """Test background uncertainty estimation."""
        estimator = ComprehensiveErrorEstimator()
        
        image = sample_images['F444W']
        
        bg_uncertainty = estimator._estimate_background_uncertainty(image)
        
        assert isinstance(bg_uncertainty, dict)
        assert 'local_rms' in bg_uncertainty
        assert 'global_rms' in bg_uncertainty
        assert 'mesh_size' in bg_uncertainty
        assert bg_uncertainty['local_rms'] > 0
        assert bg_uncertainty['global_rms'] > 0
    
    def test_crowding_uncertainty_estimation(self, sample_calibrated_results):
        """Test crowding uncertainty estimation."""
        estimator = ComprehensiveErrorEstimator()
        
        results = sample_calibrated_results
        
        crowding_uncertainties = estimator._estimate_crowding_uncertainty(results.sources)
        
        assert len(crowding_uncertainties) == len(results.sources)
        
        for source_id, uncertainty in crowding_uncertainties.items():
            assert 'neighbor_contamination' in uncertainty
            assert 'flux_bias' in uncertainty
            assert 'magnitude_bias' in uncertainty
            assert uncertainty['neighbor_contamination'] >= 0
    
    def test_systematic_uncertainty_modeling(self, sample_calibrated_results):
        """Test systematic uncertainty modeling."""
        estimator = ComprehensiveErrorEstimator()
        
        results = sample_calibrated_results
        
        systematic_errors = estimator._model_systematic_uncertainties(results)
        
        assert isinstance(systematic_errors, dict)
        
        # Should have systematic components for each band
        for band in ['F200W', 'F444W']:
            assert band in systematic_errors
            assert 'calibration' in systematic_errors[band]
            assert 'psf' in systematic_errors[band]
            assert 'aperture' in systematic_errors[band]
            
            for component, error in systematic_errors[band].items():
                assert error >= 0
    
    def test_monte_carlo_error_propagation(self, sample_calibrated_results):
        """Test Monte Carlo error propagation."""
        config = UncertaintyConfig(monte_carlo_iterations=100)  # Reduced for testing
        estimator = ComprehensiveErrorEstimator(config)
        
        results = sample_calibrated_results
        
        # Create simple correlation model
        bands = ['F200W', 'F444W']
        correlations = {'F200W-F444W': 0.3}
        correlation_model = CorrelationModel(band_correlations=correlations)
        
        mc_results = estimator._monte_carlo_error_propagation(
            results.sources[:2],  # Test with subset for speed
            bands,
            correlation_model
        )
        
        assert isinstance(mc_results, dict)
        assert len(mc_results) == 2  # Two sources
        
        for source_id, mc_data in mc_results.items():
            assert 'flux_samples' in mc_data
            assert 'magnitude_samples' in mc_data
            assert 'color_samples' in mc_data
            
            # Check sample dimensions
            for band in bands:
                if band in mc_data['flux_samples']:
                    assert len(mc_data['flux_samples'][band]) == config.monte_carlo_iterations
    
    def test_correlated_noise_modeling(self, sample_images):
        """Test correlated noise modeling."""
        estimator = ComprehensiveErrorEstimator()
        
        image = sample_images['F444W']
        spatial_scale = 2.0
        
        correlation_map = estimator._model_correlated_noise(image, spatial_scale)
        
        assert correlation_map.shape == image.shape
        assert np.all(correlation_map >= 0)
        assert np.all(correlation_map <= 1)
        
        # Central region should have higher correlation
        center = image.shape[0] // 2
        center_correlation = correlation_map[center, center]
        edge_correlation = correlation_map[0, 0]
        assert center_correlation >= edge_correlation
    
    def test_full_uncertainty_estimation(self, sample_calibrated_results, sample_images, sample_headers):
        """Test complete uncertainty estimation."""
        config = UncertaintyConfig(
            monte_carlo_iterations=50,  # Reduced for testing
            include_correlated_noise=True,
            include_crowding_effects=True
        )
        estimator = ComprehensiveErrorEstimator(config)
        
        # Run full uncertainty estimation
        uncertainty_results = estimator.estimate_uncertainties(
            calibrated_results=sample_calibrated_results,
            images=sample_images,
            headers=sample_headers
        )
        
        assert isinstance(uncertainty_results, UncertaintyResults)
        assert len(uncertainty_results.source_uncertainties) == len(sample_calibrated_results.sources)
        assert len(uncertainty_results.noise_models) == 2  # Two bands
        
        # Check individual source uncertainties
        for source_id, uncertainties in uncertainty_results.source_uncertainties.items():
            assert isinstance(uncertainties, ErrorBudget)
            assert len(uncertainties.components) > 0
            
            # Should have various error components
            component_names = [comp.source for comp in uncertainties.components]
            assert 'poisson' in component_names
            assert 'background' in component_names
            
            # Check total uncertainty is computed
            assert uncertainties.total_uncertainty > 0
            assert uncertainties.systematic_uncertainty >= 0
            assert uncertainties.random_uncertainty > 0


class TestErrorBudget:
    """Test error budget calculations."""
    
    def test_error_budget_creation(self):
        """Test error budget creation."""
        components = [
            ErrorContribution('poisson', 1000, 31.6, 0.0),
            ErrorContribution('background', 100, 10.0, 1.0),
            ErrorContribution('systematic', 50, 5.0, 0.0)
        ]
        
        budget = ErrorBudget(
            source_id=1,
            band='F444W',
            aperture=3.0,
            components=components
        )
        
        assert budget.source_id == 1
        assert budget.band == 'F444W'
        assert budget.aperture == 3.0
        assert len(budget.components) == 3
    
    def test_total_uncertainty_calculation(self):
        """Test total uncertainty calculation."""
        components = [
            ErrorContribution('poisson', 1000, 30.0, 0.0),
            ErrorContribution('background', 100, 10.0, 0.0),
            ErrorContribution('calibration', 50, 5.0, 0.0)
        ]
        
        budget = ErrorBudget(1, 'F444W', 3.0, components)
        
        # Calculate expected total (quadrature sum for uncorrelated)
        expected_total = np.sqrt(30.0**2 + 10.0**2 + 5.0**2)
        
        assert abs(budget.calculate_total_uncertainty() - expected_total) < 1e-6
        assert abs(budget.total_uncertainty - expected_total) < 1e-6
    
    def test_random_vs_systematic_separation(self):
        """Test separation of random and systematic uncertainties."""
        components = [
            ErrorContribution('poisson', 1000, 30.0, 0.0),  # Random
            ErrorContribution('background', 100, 10.0, 1.0),  # Random with correlation
            ErrorContribution('calibration', 50, 5.0, 0.0),  # Systematic
            ErrorContribution('flat_field', 20, 2.0, 0.0)   # Systematic
        ]
        
        # Mark systematic components
        components[2].metadata['systematic'] = True
        components[3].metadata['systematic'] = True
        
        budget = ErrorBudget(1, 'F444W', 3.0, components)
        
        assert budget.random_uncertainty > 0
        assert budget.systematic_uncertainty > 0
        assert budget.total_uncertainty > max(budget.random_uncertainty, budget.systematic_uncertainty)


class TestUncertaintyResults:
    """Test uncertainty results container."""
    
    def test_results_creation(self):
        """Test uncertainty results creation."""
        # Create sample error budgets
        source_uncertainties = {}
        for i in range(3):
            budget = ErrorBudget(
                source_id=i,
                band='F444W',
                aperture=3.0,
                components=[
                    ErrorContribution('poisson', 1000, 30.0, 0.0),
                    ErrorContribution('background', 100, 10.0, 0.0)
                ]
            )
            source_uncertainties[i] = budget
        
        # Create noise models
        noise_models = {
            'F444W': NoiseModel('NRCB1', 10.0, 0.005, 2.0)
        }
        
        # Create correlation model
        correlation_model = CorrelationModel(spatial_scale=2.0)
        
        config = UncertaintyConfig()
        
        results = UncertaintyResults(
            source_uncertainties=source_uncertainties,
            noise_models=noise_models,
            correlation_model=correlation_model,
            config=config
        )
        
        assert len(results.source_uncertainties) == 3
        assert len(results.noise_models) == 1
        assert results.correlation_model == correlation_model
        assert results.config == config
    
    def test_summary_statistics(self):
        """Test uncertainty summary statistics."""
        # Create test results with varying uncertainties
        source_uncertainties = {}
        for i in range(5):
            budget = ErrorBudget(
                source_id=i,
                band='F444W',
                aperture=3.0,
                components=[
                    ErrorContribution('poisson', 1000, 20.0 + i*5, 0.0),
                    ErrorContribution('background', 100, 5.0 + i, 0.0)
                ]
            )
            source_uncertainties[i] = budget
        
        results = UncertaintyResults(
            source_uncertainties=source_uncertainties,
            noise_models={'F444W': NoiseModel('NRCB1', 10.0, 0.005, 2.0)},
            correlation_model=CorrelationModel(),
            config=UncertaintyConfig()
        )
        
        # Check statistics computation
        assert 'uncertainty_statistics' in results.processing_metadata
        stats = results.processing_metadata['uncertainty_statistics']
        
        assert 'mean_uncertainty' in stats
        assert 'median_uncertainty' in stats
        assert 'uncertainty_range' in stats
        assert stats['mean_uncertainty'] > 0


class TestErrorHandling:
    """Test error handling in uncertainties module."""
    
    def test_missing_calibration_data(self):
        """Test handling of missing calibration data."""
        estimator = ComprehensiveErrorEstimator()
        
        with pytest.raises(ValueError, match="No calibrated results provided"):
            estimator.estimate_uncertainties(None, {}, {})
    
    def test_mismatched_band_data(self):
        """Test handling of mismatched band data."""
        estimator = ComprehensiveErrorEstimator()
        
        # Create minimal calibrated results
        try:
            from src.calibration import CalibrationResults, CalibrationConfig
        except ImportError:
            from calibration import CalibrationResults, CalibrationConfig
        
        results = CalibrationResults([], {}, CalibrationConfig())
        images = {'F444W': np.zeros((10, 10))}
        headers = {'F200W': fits.Header()}  # Different band
        
        with pytest.raises(ValueError, match="Mismatch between available bands"):
            estimator.estimate_uncertainties(results, images, headers)
    
    def test_invalid_monte_carlo_parameters(self):
        """Test handling of invalid Monte Carlo parameters."""
        config = UncertaintyConfig(monte_carlo_iterations=0)
        
        with pytest.raises(ValueError, match="Monte Carlo iterations must be positive"):
            ComprehensiveErrorEstimator(config)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
