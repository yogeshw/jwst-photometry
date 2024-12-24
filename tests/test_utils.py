import unittest
import numpy as np
from astropy.io import fits
from utils import read_image, save_catalog, load_config, process_image, save_results

class TestUtils(unittest.TestCase):

    def test_read_image(self):
        # Create a dummy FITS file for testing
        data = np.random.rand(100, 100)
        hdu = fits.PrimaryHDU(data)
        hdul = fits.HDUList([hdu])
        hdul.writeto('test_image.fits', overwrite=True)

        # Test read_image function
        image_data = read_image('test_image.fits')
        self.assertTrue(np.array_equal(image_data, data))

    def test_save_catalog(self):
        # Create a dummy catalog for testing
        catalog = np.array([(1, 2.5), (2, 3.5)], dtype=[('id', 'i4'), ('flux', 'f4')])
        output_directory = './output'
        save_catalog(catalog, 'fits', output_directory)
        saved_catalog = fits.open(f"{output_directory}/photometry_catalog.fits")[1].data
        self.assertTrue(np.array_equal(saved_catalog['id'], catalog['id']))
        self.assertTrue(np.array_equal(saved_catalog['flux'], catalog['flux']))

    def test_load_config(self):
        # Create a dummy YAML config file for testing
        config_data = {
            'source_detection': {
                'kernel': 3.5,
                'minarea': 3,
                'thresh': 1.2,
                'deblend_nthresh': 32,
                'deblend_cont': 0.0001,
                'clean': 'N'
            }
        }
        with open('test_config.yaml', 'w') as file:
            yaml.dump(config_data, file)

        # Test load_config function
        config = load_config('test_config.yaml')
        self.assertEqual(config, config_data)

    def test_process_image(self):
        # Create a dummy image for testing
        data = np.random.rand(100, 100)
        background = np.median(data)
        image_data = data + background

        # Test process_image function
        processed_image = process_image(image_data)
        self.assertTrue(np.allclose(processed_image, data - background))

    def test_save_results(self):
        # Create dummy results for testing
        results = {'key': 'value'}
        output_file = 'test_results.yaml'

        # Test save_results function
        save_results(results, output_file)
        with open(output_file, 'r') as file:
            saved_results = yaml.safe_load(file)
        self.assertEqual(saved_results, results)

if __name__ == '__main__':
    unittest.main()
