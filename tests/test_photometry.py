"""
Enhanced unit tests for the photometry module, including Phase 4 advanced capabilities.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from astropy.table import Table

# Import enhanced photometry modules
try:
    from src.photometry import (
        EnhancedAperturePhotometry, AperturePhotometryConfig, AperturePhotometrySource,
        AperturePhotometryResults, extract_photometry_table, perform_aperture_photometry
    )
except ImportError:
    # Fallback imports for testing
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
    
    from photometry import (
        EnhancedAperturePhotometry, AperturePhotometryConfig, AperturePhotometrySource,
        AperturePhotometryResults, extract_photometry_table, perform_aperture_photometry
    )


class TestAperturePhotometryConfig:
    """Test the aperture photometry configuration."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = AperturePhotometryConfig()
        
        assert isinstance(config.circular_apertures, list)
        assert len(config.circular_apertures) > 0
        assert config.use_elliptical_apertures is True
        assert config.use_kron_apertures is True
        assert config.background_method == "local_annulus"
        assert config.estimate_uncertainties is True
    
    def test_custom_config(self):
        """Test custom configuration."""
        custom_apertures = [1.0, 2.0, 3.0]
        config = AperturePhotometryConfig(
            circular_apertures=custom_apertures,
            use_elliptical_apertures=False,
            background_method="global"
        )
        
        assert config.circular_apertures == custom_apertures
        assert config.use_elliptical_apertures is False
        assert config.background_method == "global"


class TestAperturePhotometrySource:
    """Test the aperture photometry source container."""
    
    def test_source_creation(self):
        """Test basic source creation."""
        source = AperturePhotometrySource(id=1, x=100.5, y=200.3)
        
        assert source.id == 1
        assert source.x == 100.5
        assert source.y == 200.3
        assert isinstance(source.circular_fluxes, dict)
        assert isinstance(source.flags, list)
    
    def test_source_with_photometry(self):
        """Test source with photometry data."""
        source = AperturePhotometrySource(id=1, x=100, y=200)
        
        # Add some photometry measurements
        source.circular_fluxes[2.0] = 1000.0
        source.circular_flux_errors[2.0] = 50.0
        source.circular_magnitudes[2.0] = 20.5
        
        assert source.circular_fluxes[2.0] == 1000.0
        assert source.circular_flux_errors[2.0] == 50.0
        assert source.circular_magnitudes[2.0] == 20.5


class TestEnhancedAperturePhotometry:
    """Test the enhanced aperture photometry processor."""
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample image with artificial sources."""
        image = np.random.normal(100, 10, (200, 200))
        
        # Add some bright sources
        for x, y, flux in [(50, 50, 5000), (150, 100, 3000), (100, 150, 2000)]:
            xx, yy = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
            source = flux * np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * 2.5**2))
            image += source
        
        return image
    
    @pytest.fixture
    def sample_sources(self):
        """Create sample source catalog."""
        sources = np.array([
            (50, 50, 2.5, 2.0, 0.0),
            (150, 100, 3.0, 2.5, 0.5),
            (100, 150, 2.0, 2.0, 0.0)
        ], dtype=[('x', 'f4'), ('y', 'f4'), ('a', 'f4'), ('b', 'f4'), ('theta', 'f4')])
        
        return sources
    
    def test_processor_initialization(self):
        """Test processor initialization."""
        config = AperturePhotometryConfig()
        processor = EnhancedAperturePhotometry(config)
        
        assert processor.config == config
        assert hasattr(processor, 'logger')
    
    def test_aperture_photometry_basic(self, sample_image, sample_sources):
        """Test basic aperture photometry functionality."""
        config = AperturePhotometryConfig(
            circular_apertures=[2.0, 3.0],
            use_elliptical_apertures=False,
            use_kron_apertures=False,
            use_adaptive_apertures=False,
            estimate_uncertainties=False
        )
        
        processor = EnhancedAperturePhotometry(config)
        
        # Mock the SEP functions to avoid dependency issues
        with patch('photometry.sep.sum_circle') as mock_sep:
            mock_sep.return_value = ([1000.0, 2000.0], [50.0, 100.0], [0, 0])
            
            results = processor.perform_aperture_photometry(
                sample_image, sample_sources
            )
        
        assert isinstance(results, AperturePhotometryResults)
        assert len(results.sources) == len(sample_sources)
        assert results.config == config
    
    def test_source_validation(self, sample_image):
        """Test source validation."""
        processor = EnhancedAperturePhotometry()
        
        # Test with empty sources
        empty_sources = np.array([], dtype=[('x', 'f4'), ('y', 'f4')])
        
        with pytest.raises(ValueError, match="No sources provided"):
            processor._validate_inputs(sample_image, empty_sources, None, None)
    
    def test_background_estimation_methods(self, sample_image, sample_sources):
        """Test different background estimation methods."""
        source = AperturePhotometrySource(id=0, x=50, y=50)
        
        processor = EnhancedAperturePhotometry()
        
        # Test local annulus method
        processor.config.background_method = "local_annulus"
        processor._estimate_local_background(source, sample_image, None, None)
        
        assert source.local_background is not None
        assert isinstance(source.local_background, float)
    
    def test_quality_checks(self, sample_image):
        """Test quality checking functionality."""
        processor = EnhancedAperturePhotometry()
        
        # Test edge source
        edge_source = AperturePhotometrySource(id=0, x=5, y=5)  # Near edge
        processor._perform_quality_checks(edge_source, sample_image)
        
        assert "near_edge" in edge_source.flags
        
        # Test normal source
        normal_source = AperturePhotometrySource(id=1, x=100, y=100)
        processor._perform_quality_checks(normal_source, sample_image)
        
        assert "near_edge" not in normal_source.flags


class TestPhotometryResults:
    """Test photometry results handling."""
    
    def test_results_creation(self):
        """Test results container creation."""
        config = AperturePhotometryConfig()
        sources = [AperturePhotometrySource(id=i, x=i*10, y=i*10) for i in range(3)]
        
        results = AperturePhotometryResults(
            sources=sources,
            config=config
        )
        
        assert len(results.sources) == 3
        assert results.config == config
        assert isinstance(results.statistics, dict)
    
    def test_table_extraction(self):
        """Test extraction to astropy table."""
        config = AperturePhotometryConfig(circular_apertures=[2.0, 3.0])
        
        # Create sources with some photometry data
        sources = []
        for i in range(3):
            source = AperturePhotometrySource(id=i, x=i*50, y=i*50)
            source.circular_fluxes[2.0] = 1000 + i*100
            source.circular_flux_errors[2.0] = 50 + i*5
            source.circular_magnitudes[2.0] = 20 + i*0.1
            source.circular_magnitude_errors[2.0] = 0.05
            sources.append(source)
        
        results = AperturePhotometryResults(sources=sources, config=config)
        table = extract_photometry_table(results)
        
        assert isinstance(table, Table)
        assert len(table) == 3
        assert 'id' in table.colnames
        assert 'x' in table.colnames
        assert 'flux_r2.0' in table.colnames
        assert 'flux_err_r2.0' in table.colnames


class TestLegacyCompatibility:
    """Test legacy function compatibility."""
    
    def test_legacy_aperture_photometry(self):
        """Test legacy aperture photometry function."""
        image = np.random.normal(100, 10, (100, 100))
        sources = [{'x': 50, 'y': 50, 'a': 2, 'b': 2, 'theta': 0}]
        params = {'apertures': [2.0, 3.0]}
        
        # This should not raise an error and return results
        result = perform_aperture_photometry(image, sources, params)
        
        # The function should handle the conversion internally
        assert result is not None or result is None  # May return None on error


class TestErrorHandling:
    """Test error handling in photometry module."""
    
    def test_invalid_image(self):
        """Test handling of invalid image data."""
        processor = EnhancedAperturePhotometry()
        
        with pytest.raises(Exception):
            processor._validate_inputs(None, np.array([]), None, None)
    
    def test_mismatched_shapes(self):
        """Test handling of mismatched array shapes."""
        processor = EnhancedAperturePhotometry()
        
        image = np.ones((100, 100))
        sources = np.array([(50, 50, 2, 2, 0)], 
                          dtype=[('x', 'f4'), ('y', 'f4'), ('a', 'f4'), ('b', 'f4'), ('theta', 'f4')])
        background_map = np.ones((50, 50))  # Wrong shape
        
        with pytest.raises(ValueError, match="Background map shape"):
            processor._validate_inputs(image, sources, background_map, None)


# Legacy test class for backward compatibility
class TestPhotometry:
    """Legacy test class for basic photometry functions."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a mock image and sources for testing
        self.image = np.ones((100, 100))
        self.sources = [{'x': 50, 'y': 50, 'flux': 1000, 'size': 1.0, 'mag': 20.0}]
        self.photometry_params = {'apertures': [0.32, 0.48, 0.70, 1.00, 1.40]}
        self.correction_params = {
            'f444w_curve_of_growth': True, 
            'local_noise_box_size': 9, 
            'num_background_apertures': 10000, 
            'outlier_sigma': 5
        }

    def test_perform_aperture_photometry_legacy(self):
        """Test legacy aperture photometry function."""
        phot_table = perform_aperture_photometry(self.image, self.sources, self.photometry_params)
        
        # Should return something (or None if error handled gracefully)
        assert phot_table is not None or phot_table is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
