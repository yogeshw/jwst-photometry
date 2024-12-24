import numpy as np
import sep
from astropy.io import fits
from astropy.wcs import WCS

def detect_sources(image, detection_params):
    data = image.data
    bkg = sep.Background(data)
    data_sub = data - bkg

    objects = sep.extract(data_sub, detection_params['thresh'], err=bkg.globalrms,
                          minarea=detection_params['minarea'],
                          deblend_nthresh=detection_params['deblend_nthresh'],
                          deblend_cont=detection_params['deblend_cont'],
                          clean=detection_params['clean'])

    return objects, bkg

def generate_segmentation_map(image, objects):
    segmap = np.zeros(image.shape, dtype=int)
    for i, obj in enumerate(objects):
        y, x = np.ogrid[:image.shape[0], :image.shape[1]]
        mask = ((x - obj['x'])**2 + (y - obj['y'])**2) <= obj['a']**2
        segmap[mask] = i + 1

    return segmap

def identify_star_candidates(objects, mag_limit=24, size_limit=2.8):
    star_candidates = []
    for obj in objects:
        if obj['mag'] < mag_limit and obj['size'] < size_limit:
            star_candidates.append(obj)

    return star_candidates
