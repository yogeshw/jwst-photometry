"""
JWST-Specific Data Handling Module

This module provides specialized functions for handling JWST NIRCam data,
including FITS header parsing, WCS validation, unit conversions, and
detector-specific metadata extraction.
"""

import os
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List, Union
from dataclasses import dataclass

from astropy.io import fits
from astropy.wcs import WCS, utils as wcs_utils
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.table import Table
import warnings

# Suppress some common astropy warnings
warnings.filterwarnings('ignore', category=fits.verify.VerifyWarning)
warnings.filterwarnings('ignore', category=wcs_utils.FITSFixedWarning)


@dataclass
class NIRCamMetadata:
    """Container for NIRCam-specific metadata extracted from FITS headers."""
    
    # Basic observation information
    filename: str
    filter_name: str
    detector: str
    module: str
    channel: str
    
    # Exposure parameters
    exposure_time: float
    effective_exposure_time: float
    nframes: int
    ngroups: int
    nints: int
    
    # Calibration information
    gain: float
    readnoise: float
    zero_point: Optional[float] = None
    photometric_zero_point: Optional[float] = None
    
    # Pixel and detector properties
    pixel_scale_x: float
    pixel_scale_y: float
    detector_xsize: int
    detector_ysize: int
    
    # World coordinate system
    wcs: Optional[WCS] = None
    ra_center: Optional[float] = None
    dec_center: Optional[float] = None
    
    # Data units and conversion factors
    data_units: str = 'MJy/sr'
    flux_conversion_factor: Optional[float] = None
    
    # Observation metadata
    observation_date: Optional[str] = None
    program_id: Optional[str] = None
    observation_id: Optional[str] = None
    visit_id: Optional[str] = None
    
    # Data quality
    data_quality_flags: Optional[np.ndarray] = None
    bad_pixel_mask: Optional[np.ndarray] = None


class JWSTDataHandler:
    """
    Handles JWST NIRCam-specific data loading, validation, and preprocessing.
    
    This class provides methods for:
    - Loading and validating JWST FITS files
    - Extracting and parsing NIRCam-specific metadata
    - Unit conversions and flux calibration
    - WCS validation and coordinate transformations
    - Bad pixel and data quality handling
    """
    
    def __init__(self):
        """Initialize the JWST data handler."""
        self.logger = logging.getLogger(__name__)
        
        # NIRCam detector specifications
        self.detector_info = {
            'NRCA1': {'module': 'A', 'channel': 'SHORT', 'xsize': 2048, 'ysize': 2048},
            'NRCA2': {'module': 'A', 'channel': 'SHORT', 'xsize': 2048, 'ysize': 2048},
            'NRCA3': {'module': 'A', 'channel': 'SHORT', 'xsize': 2048, 'ysize': 2048},
            'NRCA4': {'module': 'A', 'channel': 'SHORT', 'xsize': 2048, 'ysize': 2048},
            'NRCA5': {'module': 'A', 'channel': 'LONG', 'xsize': 2048, 'ysize': 2048},
            'NRCALONG': {'module': 'A', 'channel': 'LONG', 'xsize': 2048, 'ysize': 2048},
            'NRCB1': {'module': 'B', 'channel': 'SHORT', 'xsize': 2048, 'ysize': 2048},
            'NRCB2': {'module': 'B', 'channel': 'SHORT', 'xsize': 2048, 'ysize': 2048},
            'NRCB3': {'module': 'B', 'channel': 'SHORT', 'xsize': 2048, 'ysize': 2048},
            'NRCB4': {'module': 'B', 'channel': 'SHORT', 'xsize': 2048, 'ysize': 2048},
            'NRCB5': {'module': 'B', 'channel': 'LONG', 'xsize': 2048, 'ysize': 2048},
            'NRCBLONG': {'module': 'B', 'channel': 'LONG', 'xsize': 2048, 'ysize': 2048}
        }
        
        # Filter to channel mapping
        self.filter_channels = {
            'F115W': 'SHORT', 'F150W': 'SHORT', 'F200W': 'SHORT',
            'F277W': 'LONG', 'F356W': 'LONG', 'F410M': 'LONG', 'F444W': 'LONG',
            'F070W': 'SHORT', 'F090W': 'SHORT', 'F140M': 'SHORT', 'F162M': 'SHORT',
            'F182M': 'SHORT', 'F210M': 'SHORT', 'F250M': 'LONG', 'F300M': 'LONG',
            'F335M': 'LONG', 'F360M': 'LONG', 'F460M': 'LONG', 'F480M': 'LONG'
        }
        
        # Typical pixel scales (arcsec/pixel)
        self.pixel_scales = {
            'SHORT': 0.031,
            'LONG': 0.063
        }
    
    def load_jwst_image(self, filepath: str, 
                       extension: Union[int, str] = 'SCI',
                       load_dq: bool = True,
                       load_err: bool = True) -> Tuple[np.ndarray, NIRCamMetadata, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Load a JWST NIRCam image with comprehensive metadata extraction.
        
        Parameters:
        -----------
        filepath : str
            Path to the JWST FITS file
        extension : int or str, default='SCI'
            FITS extension containing the science data
        load_dq : bool, default=True
            Whether to load data quality array
        load_err : bool, default=True
            Whether to load error array
            
        Returns:
        --------
        data : numpy.ndarray
            Science image data
        metadata : NIRCamMetadata
            Extracted metadata
        dq_array : numpy.ndarray or None
            Data quality array
        err_array : numpy.ndarray or None
            Error array
            
        Raises:
        -------
        FileNotFoundError
            If the file doesn't exist
        ValueError
            If the file is not a valid JWST NIRCam image
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        self.logger.info(f"Loading JWST image: {filepath}")
        
        try:
            with fits.open(filepath) as hdul:
                # Load primary header for observation metadata
                primary_header = hdul[0].header
                
                # Find science extension
                if isinstance(extension, str):
                    sci_ext = self._find_extension_by_name(hdul, extension)
                else:
                    sci_ext = extension
                
                if sci_ext is None:
                    raise ValueError(f"Science extension '{extension}' not found in {filepath}")
                
                # Load science data and header
                data = hdul[sci_ext].data.astype(np.float64)
                sci_header = hdul[sci_ext].header
                
                # Load data quality array if requested
                dq_array = None
                if load_dq:
                    dq_ext = self._find_extension_by_name(hdul, 'DQ')
                    if dq_ext is not None:
                        dq_array = hdul[dq_ext].data
                        self.logger.debug("Loaded data quality array")
                
                # Load error array if requested
                err_array = None
                if load_err:
                    err_ext = self._find_extension_by_name(hdul, 'ERR')
                    if err_ext is not None:
                        err_array = hdul[err_ext].data.astype(np.float64)
                        self.logger.debug("Loaded error array")
                
                # Extract metadata
                metadata = self._extract_metadata(filepath, primary_header, sci_header, data.shape)
                
        except Exception as e:
            self.logger.error(f"Error loading JWST image {filepath}: {e}")
            raise
        
        self.logger.info(f"Successfully loaded {metadata.filter_name} image from {metadata.detector}")
        return data, metadata, dq_array, err_array
    
    def _find_extension_by_name(self, hdul: fits.HDUList, extname: str) -> Optional[int]:
        """Find FITS extension by name."""
        for i, hdu in enumerate(hdul):
            if hasattr(hdu, 'header') and hdu.header.get('EXTNAME', '').upper() == extname.upper():
                return i
        return None
    
    def _extract_metadata(self, filepath: Path, primary_header: fits.Header, 
                         sci_header: fits.Header, data_shape: Tuple[int, int]) -> NIRCamMetadata:
        """
        Extract comprehensive metadata from JWST FITS headers.
        
        Parameters:
        -----------
        filepath : Path
            Path to the FITS file
        primary_header : fits.Header
            Primary FITS header
        sci_header : fits.Header
            Science extension header
        data_shape : tuple
            Shape of the image data
            
        Returns:
        --------
        NIRCamMetadata
            Extracted metadata object
        """
        # Use science header preferentially, fall back to primary header
        header = sci_header if sci_header else primary_header
        
        # Basic instrument information
        filter_name = header.get('FILTER', 'UNKNOWN')
        detector = header.get('DETECTOR', primary_header.get('DETECTOR', 'UNKNOWN'))
        
        # Get detector info
        det_info = self.detector_info.get(detector, {})
        module = det_info.get('module', 'UNKNOWN')
        channel = det_info.get('channel', self.filter_channels.get(filter_name, 'UNKNOWN'))
        
        # Exposure parameters
        exposure_time = header.get('EXPTIME', primary_header.get('EXPTIME', 0.0))
        effective_exposure_time = header.get('EFFEXPTM', exposure_time)
        nframes = header.get('NFRAMES', primary_header.get('NFRAMES', 1))
        ngroups = header.get('NGROUPS', primary_header.get('NGROUPS', 1))
        nints = header.get('NINTS', primary_header.get('NINTS', 1))
        
        # Detector characteristics
        gain = header.get('GAIN', primary_header.get('GAIN', 1.0))
        readnoise = header.get('READNOIS', primary_header.get('READNOIS', 0.0))
        
        # Photometric calibration
        zero_point = header.get('PHOTZP', None)
        photometric_zero_point = header.get('PHOTMJSR', None)
        
        # Pixel scale information
        try:
            wcs = WCS(header)
            pixel_scale_x, pixel_scale_y = self._get_pixel_scales_from_wcs(wcs)
            ra_center, dec_center = self._get_image_center(wcs, data_shape)
        except Exception as e:
            self.logger.warning(f"Failed to extract WCS information: {e}")
            wcs = None
            channel_default_scale = self.pixel_scales.get(channel, 0.031)
            pixel_scale_x = pixel_scale_y = channel_default_scale
            ra_center = dec_center = None
        
        # Data units and conversion
        data_units = header.get('BUNIT', 'MJy/sr')
        flux_conversion_factor = self._calculate_flux_conversion_factor(
            data_units, pixel_scale_x, pixel_scale_y
        )
        
        # Observation metadata
        observation_date = header.get('DATE-OBS', primary_header.get('DATE-OBS', None))
        program_id = primary_header.get('PROGRAM', None)
        observation_id = primary_header.get('OBSERVTN', None)
        visit_id = primary_header.get('VISIT', None)
        
        # Create metadata object
        metadata = NIRCamMetadata(
            filename=str(filepath.name),
            filter_name=filter_name,
            detector=detector,
            module=module,
            channel=channel,
            exposure_time=exposure_time,
            effective_exposure_time=effective_exposure_time,
            nframes=nframes,
            ngroups=ngroups,
            nints=nints,
            gain=gain,
            readnoise=readnoise,
            zero_point=zero_point,
            photometric_zero_point=photometric_zero_point,
            pixel_scale_x=pixel_scale_x,
            pixel_scale_y=pixel_scale_y,
            detector_xsize=data_shape[1],
            detector_ysize=data_shape[0],
            wcs=wcs,
            ra_center=ra_center,
            dec_center=dec_center,
            data_units=data_units,
            flux_conversion_factor=flux_conversion_factor,
            observation_date=observation_date,
            program_id=program_id,
            observation_id=observation_id,
            visit_id=visit_id
        )
        
        return metadata
    
    def _get_pixel_scales_from_wcs(self, wcs: WCS) -> Tuple[float, float]:
        """
        Extract pixel scales from WCS object.
        
        Parameters:
        -----------
        wcs : WCS
            World coordinate system object
            
        Returns:
        --------
        tuple
            Pixel scales in x and y directions (arcsec/pixel)
        """
        try:
            # Get pixel scale matrix
            pixel_scale_matrix = wcs.pixel_scale_matrix
            
            # Calculate scales from diagonal elements
            pixel_scale_x = abs(pixel_scale_matrix[0, 0]) * 3600  # Convert to arcsec
            pixel_scale_y = abs(pixel_scale_matrix[1, 1]) * 3600  # Convert to arcsec
            
            return pixel_scale_x, pixel_scale_y
            
        except Exception:
            # Fall back to CD matrix if available
            try:
                cd11 = wcs.wcs.cd[0, 0]
                cd22 = wcs.wcs.cd[1, 1]
                pixel_scale_x = abs(cd11) * 3600
                pixel_scale_y = abs(cd22) * 3600
                return pixel_scale_x, pixel_scale_y
            except Exception:
                # Use CDELT if available
                try:
                    cdelt1 = abs(wcs.wcs.cdelt[0]) * 3600
                    cdelt2 = abs(wcs.wcs.cdelt[1]) * 3600
                    return cdelt1, cdelt2
                except Exception:
                    raise ValueError("Could not determine pixel scale from WCS")
    
    def _get_image_center(self, wcs: WCS, data_shape: Tuple[int, int]) -> Tuple[float, float]:
        """
        Get RA/Dec coordinates of image center.
        
        Parameters:
        -----------
        wcs : WCS
            World coordinate system object
        data_shape : tuple
            Shape of the image data (ny, nx)
            
        Returns:
        --------
        tuple
            RA and Dec of image center in degrees
        """
        center_pixel = (data_shape[1] / 2.0, data_shape[0] / 2.0)  # (x, y)
        ra_center, dec_center = wcs.pixel_to_world_values(center_pixel[0], center_pixel[1])
        return float(ra_center), float(dec_center)
    
    def _calculate_flux_conversion_factor(self, data_units: str, 
                                        pixel_scale_x: float, 
                                        pixel_scale_y: float) -> Optional[float]:
        """
        Calculate conversion factor from data units to standard flux units.
        
        Parameters:
        -----------
        data_units : str
            Units of the image data
        pixel_scale_x : float
            Pixel scale in x direction (arcsec/pixel)
        pixel_scale_y : float
            Pixel scale in y direction (arcsec/pixel)
            
        Returns:
        --------
        float or None
            Conversion factor to convert to µJy
        """
        if data_units.upper() in ['MJY/SR', 'MJYSR']:
            # Convert MJy/sr to µJy
            # 1 MJy/sr * (pixel_scale_arcsec)^2 * (1 sr / (206265 arcsec)^2) * (1e12 µJy / 1 MJy)
            pixel_area_sr = (pixel_scale_x * pixel_scale_y) / (206265.0**2)
            conversion_factor = 1e12 * pixel_area_sr  # MJy/sr to µJy
            return conversion_factor
        
        elif data_units.upper() in ['JY/PIXEL', 'JYPIXEL']:
            # Convert Jy/pixel to µJy
            return 1e6
        
        elif data_units.upper() in ['UJY/PIXEL', 'UJYPIXEL']:
            # Already in µJy
            return 1.0
        
        else:
            self.logger.warning(f"Unknown data units: {data_units}. No conversion applied.")
            return None
    
    def convert_to_microjanskys(self, data: np.ndarray, metadata: NIRCamMetadata) -> np.ndarray:
        """
        Convert image data to microjanskys.
        
        Parameters:
        -----------
        data : numpy.ndarray
            Image data in original units
        metadata : NIRCamMetadata
            Metadata containing conversion information
            
        Returns:
        --------
        numpy.ndarray
            Image data in microjanskys
        """
        if metadata.flux_conversion_factor is None:
            self.logger.warning("No flux conversion factor available")
            return data.copy()
        
        converted_data = data * metadata.flux_conversion_factor
        self.logger.debug(f"Converted data from {metadata.data_units} to µJy")
        return converted_data
    
    def validate_wcs(self, metadata: NIRCamMetadata, tolerance: float = 0.1) -> bool:
        """
        Validate WCS information for consistency and accuracy.
        
        Parameters:
        -----------
        metadata : NIRCamMetadata
            Metadata containing WCS information
        tolerance : float, default=0.1
            Tolerance for pixel scale validation (arcsec)
            
        Returns:
        --------
        bool
            True if WCS validation passes
        """
        if metadata.wcs is None:
            self.logger.error("No WCS information available")
            return False
        
        try:
            # Check if WCS can perform basic transformations
            test_pixel = (100, 100)
            world_coords = metadata.wcs.pixel_to_world_values(*test_pixel)
            back_to_pixel = metadata.wcs.world_to_pixel_values(*world_coords)
            
            # Check round-trip accuracy
            pixel_diff = np.sqrt(sum((np.array(test_pixel) - np.array(back_to_pixel))**2))
            if pixel_diff > 0.01:  # 0.01 pixel tolerance
                self.logger.warning(f"WCS round-trip error: {pixel_diff} pixels")
                return False
            
            # Validate pixel scale consistency
            expected_scale = self.pixel_scales.get(metadata.channel, 0.031)
            scale_avg = (metadata.pixel_scale_x + metadata.pixel_scale_y) / 2.0
            scale_diff = abs(scale_avg - expected_scale)
            
            if scale_diff > tolerance:
                self.logger.warning(f"Pixel scale differs from expected: {scale_avg:.3f}\" vs {expected_scale:.3f}\"")
                return False
            
            self.logger.debug("WCS validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"WCS validation failed: {e}")
            return False
    
    def create_bad_pixel_mask(self, dq_array: Optional[np.ndarray], 
                             flag_values: List[int] = None) -> Optional[np.ndarray]:
        """
        Create a bad pixel mask from data quality array.
        
        Parameters:
        -----------
        dq_array : numpy.ndarray or None
            Data quality array
        flag_values : list of int, optional
            DQ flag values to mask (default: common bad pixel flags)
            
        Returns:
        --------
        numpy.ndarray or None
            Boolean mask where True indicates bad pixels
        """
        if dq_array is None:
            return None
        
        # Default bad pixel flags for JWST
        if flag_values is None:
            flag_values = [
                1,      # DO_NOT_USE
                2,      # SATURATED
                4,      # JUMP_DET
                8,      # DROPOUT
                512,    # NONLINEAR
                1024,   # BAD_REF_PIXEL
                2048,   # NO_FLAT_FIELD
                4096,   # NO_GAIN_VALUE
                8192,   # NO_LIN_CORR
                16384,  # NO_SAT_CHECK
                32768,  # UNRELIABLE_ERROR
                65536,  # NON_SCIENCE
                131072, # DEAD
                262144, # HOT
                524288, # WARM
                1048576 # LOW_QE
            ]
        
        # Create mask for any of the specified flag values
        bad_pixel_mask = np.zeros_like(dq_array, dtype=bool)
        for flag in flag_values:
            bad_pixel_mask |= (dq_array & flag) != 0
        
        n_bad_pixels = np.sum(bad_pixel_mask)
        total_pixels = bad_pixel_mask.size
        bad_pixel_fraction = n_bad_pixels / total_pixels
        
        self.logger.info(f"Created bad pixel mask: {n_bad_pixels}/{total_pixels} "
                        f"({bad_pixel_fraction:.3%}) pixels flagged")
        
        return bad_pixel_mask
    
    def get_effective_gain(self, metadata: NIRCamMetadata) -> float:
        """
        Calculate effective gain accounting for multi-frame averaging.
        
        Parameters:
        -----------
        metadata : NIRCamMetadata
            Metadata containing exposure information
            
        Returns:
        --------
        float
            Effective gain in e-/DN
        """
        # For JWST, effective gain depends on the number of frames and groups
        # that were averaged together during readout
        effective_gain = metadata.gain * metadata.nframes
        
        self.logger.debug(f"Effective gain: {effective_gain:.2f} e-/DN "
                         f"(base gain: {metadata.gain:.2f}, nframes: {metadata.nframes})")
        
        return effective_gain
    
    def load_multiple_images(self, filepaths: Dict[str, str], 
                           **kwargs) -> Dict[str, Tuple[np.ndarray, NIRCamMetadata, 
                                                      Optional[np.ndarray], Optional[np.ndarray]]]:
        """
        Load multiple JWST images efficiently.
        
        Parameters:
        -----------
        filepaths : dict
            Dictionary mapping filter names to file paths
        **kwargs
            Additional arguments passed to load_jwst_image
            
        Returns:
        --------
        dict
            Dictionary containing loaded images and metadata for each filter
        """
        images_data = {}
        
        self.logger.info(f"Loading {len(filepaths)} JWST images")
        
        for filter_name, filepath in filepaths.items():
            try:
                data, metadata, dq_array, err_array = self.load_jwst_image(filepath, **kwargs)
                images_data[filter_name] = (data, metadata, dq_array, err_array)
                self.logger.info(f"Successfully loaded {filter_name}: {filepath}")
            except Exception as e:
                self.logger.error(f"Failed to load {filter_name} ({filepath}): {e}")
                continue
        
        self.logger.info(f"Successfully loaded {len(images_data)}/{len(filepaths)} images")
        return images_data
    
    def validate_image_consistency(self, images_data: Dict[str, Tuple]) -> bool:
        """
        Validate consistency across multiple images (WCS, pixel scales, etc.).
        
        Parameters:
        -----------
        images_data : dict
            Dictionary of loaded images and metadata
            
        Returns:
        --------
        bool
            True if all images are consistent
        """
        if len(images_data) < 2:
            return True
        
        reference_filter = list(images_data.keys())[0]
        ref_data, ref_metadata, _, _ = images_data[reference_filter]
        
        inconsistencies = []
        
        for filter_name, (data, metadata, _, _) in images_data.items():
            if filter_name == reference_filter:
                continue
            
            # Check image dimensions
            if data.shape != ref_data.shape:
                inconsistencies.append(f"{filter_name}: shape mismatch {data.shape} vs {ref_data.shape}")
            
            # Check WCS consistency
            if metadata.wcs is not None and ref_metadata.wcs is not None:
                # Compare image centers
                center_sep = np.sqrt((metadata.ra_center - ref_metadata.ra_center)**2 + 
                                   (metadata.dec_center - ref_metadata.dec_center)**2)
                if center_sep > 0.1:  # 0.1 degree tolerance
                    inconsistencies.append(f"{filter_name}: WCS center differs by {center_sep:.3f} deg")
            
            # Check pixel scale consistency within channel
            if metadata.channel == ref_metadata.channel:
                scale_diff = abs(metadata.pixel_scale_x - ref_metadata.pixel_scale_x)
                if scale_diff > 0.001:  # 1 mas tolerance
                    inconsistencies.append(f"{filter_name}: pixel scale differs by {scale_diff:.4f}\"")
        
        if inconsistencies:
            self.logger.warning("Image consistency issues found:")
            for issue in inconsistencies:
                self.logger.warning(f"  - {issue}")
            return False
        
        self.logger.info("All images are consistent")
        return True
