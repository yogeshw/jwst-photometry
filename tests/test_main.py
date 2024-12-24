import unittest
import yaml
import numpy as np
from astropy.io import fits
from src.main import main
from src.utils import read_image, save_catalog, process_image

class TestMain(unittest.TestCase):

    def setUp(self):
        # Load test configuration
        with open('config.yaml', 'r') as file:
            self.config = yaml.safe_load(file)

        # Create test images and weights
        self.images = {}
        self.weights = {}
        for band in ['F115W', 'F150W', 'F200W', 'F277W', 'F356W', 'F410M', 'F444W']:
            self.images[band] = np.random.rand(100, 100)
            self.weights[band] = np.random.rand(100, 100)

    def test_read_image(self):
        # Test reading an image
        image_data = read_image('data/F115W.fits')
        self.assertIsInstance(image_data, np.ndarray)

    def test_process_image(self):
        # Test processing an image
        processed_image = process_image(self.images['F115W'])
        self.assertIsInstance(processed_image, np.ndarray)

    def test_save_catalog(self):
        # Test saving a catalog
        catalog = {'flux': [1.0, 2.0, 3.0], 'flux_err': [0.1, 0.2, 0.3]}
        save_catalog(catalog, 'fits', './output')
        self.assertTrue(os.path.exists('./output/photometry_catalog.fits'))

    def test_main(self):
        # Test the main function
        main()
        self.assertTrue(os.path.exists('./output/photometry_catalog.fits'))

if __name__ == '__main__':
    unittest.main()
