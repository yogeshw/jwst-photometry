import unittest
import numpy as np
import sep
from astropy.io import fits
from detection import detect_sources, generate_segmentation_map, identify_star_candidates, identify_star_candidates_advanced

class TestDetection(unittest.TestCase):

    def setUp(self):
        # Create a mock image for testing
        self.image = np.random.rand(100, 100)
        self.detection_params = {
            'thresh': 1.2,
            'minarea': 3,
            'deblend_nthresh': 32,
            'deblend_cont': 0.0001,
            'clean': False
        }

    def test_detect_sources(self):
        objects, bkg = detect_sources(self.image, self.detection_params)
        self.assertIsInstance(objects, np.ndarray)
        self.assertIsInstance(bkg, sep.Background)
        self.assertGreater(len(objects), 0)

    def test_generate_segmentation_map(self):
        objects, _ = detect_sources(self.image, self.detection_params)
        segmap = generate_segmentation_map(self.image, objects)
        self.assertIsInstance(segmap, np.ndarray)
        self.assertEqual(segmap.shape, self.image.shape)

    def test_identify_star_candidates(self):
        objects, _ = detect_sources(self.image, self.detection_params)
        star_candidates = identify_star_candidates(objects)
        self.assertIsInstance(star_candidates, list)

    def test_identify_star_candidates_advanced(self):
        objects, _ = detect_sources(self.image, self.detection_params)
        f200w_image = np.random.rand(100, 100)
        f160w_image = np.random.rand(100, 100)
        star_candidates = identify_star_candidates_advanced(objects, f200w_image, f160w_image)
        self.assertIsInstance(star_candidates, list)

    def test_detect_sources_with_gaussian(self):
        # Create a mock image with Gaussian sources
        self.image = np.random.normal(loc=0.0, scale=1.0, size=(100, 100))
        for _ in range(10):
            x, y = np.random.randint(0, 100, size=2)
            width = np.random.uniform(1, 3)
            amplitude = np.random.uniform(5, 10)
            self.image += amplitude * np.exp(-((np.arange(100)[:, None] - x)**2 + (np.arange(100)[None, :] - y)**2) / (2 * width**2))
        
        objects, bkg = detect_sources(self.image, self.detection_params)
        self.assertIsInstance(objects, np.ndarray)
        self.assertIsInstance(bkg, sep.Background)
        self.assertGreater(len(objects), 0)

if __name__ == '__main__':
    unittest.main()
