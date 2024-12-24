import unittest
import numpy as np
from astropy.io import fits
from photometry import perform_aperture_photometry, correct_photometry, derive_photometric_uncertainties

class TestPhotometry(unittest.TestCase):

    def setUp(self):
        # Create a mock image and sources for testing
        self.image = np.ones((100, 100))
        self.sources = [{'x': 50, 'y': 50, 'flux': 1000, 'size': 1.0, 'mag': 20.0, 'weight': np.ones((100, 100))}]
        self.photometry_params = {'apertures': [0.32, 0.48, 0.70, 1.00, 1.40]}
        self.correction_params = {'f444w_curve_of_growth': True, 'local_noise_box_size': 9, 'num_background_apertures': 10000, 'outlier_sigma': 5}

    def test_perform_aperture_photometry(self):
        phot_table = perform_aperture_photometry(self.image, self.sources, self.photometry_params)
        self.assertIsNotNone(phot_table, "Aperture photometry failed to return a result.")
        self.assertGreater(len(phot_table), 0, "Aperture photometry returned an empty result.")

    def test_correct_photometry(self):
        phot_table = perform_aperture_photometry(self.image, self.sources, self.photometry_params)
        corrected_photometry = correct_photometry({'test_band': phot_table}, self.correction_params)
        self.assertIn('test_band', corrected_photometry, "Corrected photometry does not contain the test band.")
        self.assertIsNotNone(corrected_photometry['test_band'], "Corrected photometry for the test band is None.")

    def test_derive_photometric_uncertainties(self):
        sources_with_uncertainties = derive_photometric_uncertainties(self.image, self.sources, self.correction_params)
        self.assertIn('flux_err', sources_with_uncertainties[0], "Photometric uncertainties were not derived correctly.")
        self.assertGreater(sources_with_uncertainties[0]['flux_err'], 0, "Photometric uncertainties are not greater than zero.")

if __name__ == '__main__':
    unittest.main()
