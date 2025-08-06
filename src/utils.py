import numpy as np
import yaml
import logging
import os
import sys
import time
import psutil
from pathlib import Path
from typing import Dict, Any, Union, Optional, List, Tuple
from contextlib import contextmanager
import warnings

from astropy.io import fits
from astropy.table import Table
import sep

# Configure logging
class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels."""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logging(level: str = 'INFO', 
                 log_file: Optional[str] = None,
                 enable_colors: bool = True) -> logging.Logger:
    """
    Set up comprehensive logging for the photometry pipeline.
    
    Parameters:
    -----------
    level : str, default='INFO'
        Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
    log_file : str, optional
        Path to log file. If None, only logs to console
    enable_colors : bool, default=True
        Whether to use colored output for console logging
        
    Returns:
    --------
    logging.Logger
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger('jwst_photometry')
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    
    if enable_colors and sys.stdout.isatty():
        console_formatter = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
    else:
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
    
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


class DataValidationError(Exception):
    """Custom exception for data validation errors."""
    pass


class ConfigurationError(Exception):
    """Custom exception for configuration errors."""
    pass


class PhotometryError(Exception):
    """Custom exception for photometry-related errors."""
    pass


@contextmanager
def memory_monitor(operation_name: str, logger: Optional[logging.Logger] = None):
    """
    Context manager to monitor memory usage during operations.
    
    Parameters:
    -----------
    operation_name : str
        Name of the operation being monitored
    logger : logging.Logger, optional
        Logger instance to use for reporting
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    logger.debug(f"Starting {operation_name} - Initial memory: {initial_memory:.1f} MB")
    
    try:
        yield
    finally:
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_delta = final_memory - initial_memory
        
        logger.debug(f"Completed {operation_name} - Final memory: {final_memory:.1f} MB "
                    f"(Δ{memory_delta:+.1f} MB)")


@contextmanager
def timing_context(operation_name: str, logger: Optional[logging.Logger] = None):
    """
    Context manager to time operations.
    
    Parameters:
    -----------
    operation_name : str
        Name of the operation being timed
    logger : logging.Logger, optional
        Logger instance to use for reporting
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    start_time = time.time()
    logger.debug(f"Starting {operation_name}")
    
    try:
        yield
    finally:
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"Completed {operation_name} in {duration:.2f} seconds")


def validate_file_exists(file_path: Union[str, Path], 
                        file_description: str = "File") -> Path:
    """
    Validate that a file exists and is readable.
    
    Parameters:
    -----------
    file_path : str or Path
        Path to the file
    file_description : str, default="File"
        Description of the file for error messages
        
    Returns:
    --------
    Path
        Validated file path
        
    Raises:
    -------
    FileNotFoundError
        If file doesn't exist or isn't readable
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"{file_description} not found: {file_path}")
    
    if not file_path.is_file():
        raise ValueError(f"{file_description} is not a regular file: {file_path}")
    
    if not os.access(file_path, os.R_OK):
        raise PermissionError(f"{file_description} is not readable: {file_path}")
    
    return file_path


def validate_output_directory(output_dir: Union[str, Path], 
                            create_if_missing: bool = True) -> Path:
    """
    Validate and optionally create output directory.
    
    Parameters:
    -----------
    output_dir : str or Path
        Path to output directory
    create_if_missing : bool, default=True
        Whether to create directory if it doesn't exist
        
    Returns:
    --------
    Path
        Validated output directory path
        
    Raises:
    -------
    ValueError
        If directory validation fails
    """
    output_dir = Path(output_dir)
    
    if output_dir.exists():
        if not output_dir.is_dir():
            raise ValueError(f"Output path exists but is not a directory: {output_dir}")
        if not os.access(output_dir, os.W_OK):
            raise PermissionError(f"Output directory is not writable: {output_dir}")
    elif create_if_missing:
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            logging.getLogger(__name__).info(f"Created output directory: {output_dir}")
        except Exception as e:
            raise ValueError(f"Failed to create output directory {output_dir}: {e}")
    else:
        raise FileNotFoundError(f"Output directory does not exist: {output_dir}")
    
    return output_dir


def read_image(file_path: Union[str, Path], 
               extension: Union[int, str] = 0,
               validate_data: bool = True) -> np.ndarray:
    """
    Read an image file with comprehensive error handling and validation.

    Parameters:
    -----------
    file_path : str or Path
        Path to the image file
    extension : int or str, default=0
        FITS extension to read
    validate_data : bool, default=True
        Whether to validate the image data
        
    Returns:
    --------
    numpy.ndarray
        Image data as float64 array
        
    Raises:
    -------
    FileNotFoundError
        If file doesn't exist
    DataValidationError
        If image data is invalid
    """
    logger = logging.getLogger(__name__)
    
    # Validate file exists
    file_path = validate_file_exists(file_path, "Image file")
    
    try:
        with timing_context(f"Loading image {file_path.name}", logger):
            with fits.open(file_path) as hdul:
                # Find the requested extension
                if isinstance(extension, str):
                    # Look for extension by name
                    ext_idx = None
                    for i, hdu in enumerate(hdul):
                        if hasattr(hdu, 'header') and hdu.header.get('EXTNAME', '').upper() == extension.upper():
                            ext_idx = i
                            break
                    if ext_idx is None:
                        raise ValueError(f"Extension '{extension}' not found in {file_path}")
                else:
                    ext_idx = extension
                
                # Load image data
                image_data = hdul[ext_idx].data
                
                if image_data is None:
                    raise DataValidationError(f"No data found in extension {extension} of {file_path}")
                
                # Convert to float64 for numerical precision
                image_data = image_data.astype(np.float64)
                
                # Validate data if requested
                if validate_data:
                    _validate_image_data(image_data, file_path)
                
                logger.debug(f"Loaded image: shape={image_data.shape}, dtype={image_data.dtype}")
                return image_data
                
    except Exception as e:
        logger.error(f"Failed to read image {file_path}: {e}")
        raise


def _validate_image_data(image_data: np.ndarray, file_path: Path) -> None:
    """
    Validate image data for common issues.
    
    Parameters:
    -----------
    image_data : numpy.ndarray
        Image data to validate
    file_path : Path
        Path to the image file (for error messages)
        
    Raises:
    -------
    DataValidationError
        If validation fails
    """
    logger = logging.getLogger(__name__)
    
    # Check for empty data
    if image_data.size == 0:
        raise DataValidationError(f"Image {file_path} contains no data")
    
    # Check dimensions
    if image_data.ndim != 2:
        if image_data.ndim == 3 and image_data.shape[0] == 1:
            # Handle 3D images with single plane
            image_data = image_data.squeeze()
            logger.warning(f"Squeezed 3D image {file_path} to 2D")
        else:
            raise DataValidationError(f"Image {file_path} has {image_data.ndim} dimensions, expected 2")
    
    # Check for all NaN or infinite values
    finite_mask = np.isfinite(image_data)
    n_finite = np.sum(finite_mask)
    
    if n_finite == 0:
        raise DataValidationError(f"Image {file_path} contains no finite values")
    
    finite_fraction = n_finite / image_data.size
    if finite_fraction < 0.5:
        logger.warning(f"Image {file_path} has only {finite_fraction:.1%} finite values")
    
    # Check data range
    finite_data = image_data[finite_mask]
    data_min, data_max = np.min(finite_data), np.max(finite_data)
    
    if data_min == data_max:
        logger.warning(f"Image {file_path} has constant value: {data_min}")
    
    logger.debug(f"Image validation passed: {file_path}")
    logger.debug(f"  Shape: {image_data.shape}")
    logger.debug(f"  Data range: [{data_min:.3e}, {data_max:.3e}]")
    logger.debug(f"  Finite pixels: {finite_fraction:.1%}")


def save_catalog(catalog: Union[Table, Dict[str, np.ndarray]], 
                format_type: str,
                output_directory: Union[str, Path],
                filename: Optional[str] = None,
                metadata: Optional[Dict[str, Any]] = None) -> Path:
    """
    Save the photometry catalog with enhanced error handling and metadata.

    Parameters:
    -----------
    catalog : Table or dict
        Photometry catalog data
    format_type : str
        Output format ('fits', 'csv', 'hdf5', 'ascii')
    output_directory : str or Path
        Directory to save the catalog
    filename : str, optional
        Output filename (auto-generated if None)
    metadata : dict, optional
        Additional metadata to include
        
    Returns:
    --------
    Path
        Path to the saved catalog file
        
    Raises:
    -------
    ValueError
        If format is not supported
    """
    logger = logging.getLogger(__name__)
    
    # Validate output directory
    output_dir = validate_output_directory(output_directory)
    
    # Generate filename if not provided
    if filename is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"photometry_catalog_{timestamp}.{format_type}"
    
    output_path = output_dir / filename
    
    # Convert dict to Table if necessary
    if isinstance(catalog, dict):
        catalog = Table(catalog)
    
    # Add metadata if provided
    if metadata:
        for key, value in metadata.items():
            catalog.meta[key] = value
    
    try:
        with timing_context(f"Saving catalog to {output_path}", logger):
            if format_type.lower() == 'fits':
                _save_catalog_fits(catalog, output_path)
            elif format_type.lower() == 'csv':
                _save_catalog_csv(catalog, output_path)
            elif format_type.lower() == 'hdf5':
                _save_catalog_hdf5(catalog, output_path)
            elif format_type.lower() == 'ascii':
                _save_catalog_ascii(catalog, output_path)
            else:
                raise ValueError(f"Unsupported catalog format: {format_type}")
        
        logger.info(f"Successfully saved catalog: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Failed to save catalog: {e}")
        raise


def _save_catalog_fits(catalog: Table, output_path: Path) -> None:
    """Save catalog in FITS format with comprehensive metadata."""
    logger = logging.getLogger(__name__)
    
    # Create primary HDU with metadata
    primary_hdu = fits.PrimaryHDU()
    primary_hdu.header['ORIGIN'] = 'JWST-Photometry Pipeline'
    primary_hdu.header['DATE'] = time.strftime('%Y-%m-%dT%H:%M:%S')
    primary_hdu.header['NSOURCES'] = len(catalog)
    
    # Add metadata from catalog
    for key, value in catalog.meta.items():
        try:
            primary_hdu.header[key] = value
        except (ValueError, TypeError):
            logger.warning(f"Could not add metadata key '{key}' to FITS header")
    
    # Create table HDU
    table_hdu = fits.BinTableHDU(catalog, name='CATALOG')
    
    # Create HDU list and save
    hdul = fits.HDUList([primary_hdu, table_hdu])
    hdul.writeto(output_path, overwrite=True)


def _save_catalog_csv(catalog: Table, output_path: Path) -> None:
    """Save catalog in CSV format."""
    catalog.write(output_path, format='csv', overwrite=True)


def _save_catalog_hdf5(catalog: Table, output_path: Path) -> None:
    """Save catalog in HDF5 format."""
    import h5py
    
    with h5py.File(output_path, 'w') as f:
        # Create main catalog group
        cat_group = f.create_group('catalog')
        
        # Save table data
        for col_name in catalog.colnames:
            cat_group.create_dataset(col_name, data=catalog[col_name])
        
        # Save metadata
        if catalog.meta:
            meta_group = f.create_group('metadata')
            for key, value in catalog.meta.items():
                try:
                    meta_group.attrs[key] = value
                except (TypeError, ValueError):
                    # Convert to string if direct storage fails
                    meta_group.attrs[key] = str(value)


def _save_catalog_ascii(catalog: Table, output_path: Path) -> None:
    """Save catalog in ASCII format."""
    catalog.write(output_path, format='ascii.ecsv', overwrite=True)


def load_config(config_file: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML file with validation.

    Parameters:
    -----------
    config_file : str or Path
        Path to the YAML configuration file
        
    Returns:
    --------
    dict
        Configuration parameters
        
    Raises:
    -------
    ConfigurationError
        If configuration loading or validation fails
    """
    logger = logging.getLogger(__name__)
    
    # Validate config file exists
    config_path = validate_file_exists(config_file, "Configuration file")
    
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        if config is None:
            raise ConfigurationError(f"Configuration file {config_path} is empty")
        
        logger.info(f"Successfully loaded configuration from {config_path}")
        return config
        
    except yaml.YAMLError as e:
        raise ConfigurationError(f"Error parsing YAML configuration {config_path}: {e}")
    except Exception as e:
        raise ConfigurationError(f"Error loading configuration {config_path}: {e}")


def process_image(image_data: np.ndarray,
                 subtract_background: bool = True,
                 background_kwargs: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Process image data with enhanced background subtraction and validation.

    Parameters:
    -----------
    image_data : numpy.ndarray
        Input image data
    subtract_background : bool, default=True
        Whether to perform background subtraction
    background_kwargs : dict, optional
        Additional arguments for SEP background estimation
        
    Returns:
    --------
    tuple
        Processed image data and background array (if computed)
        
    Raises:
    -------
    DataValidationError
        If image processing fails
    """
    logger = logging.getLogger(__name__)
    
    # Validate input
    if not isinstance(image_data, np.ndarray):
        raise DataValidationError("Input must be a numpy array")
    
    if image_data.ndim != 2:
        raise DataValidationError(f"Input must be 2D, got {image_data.ndim}D")
    
    # Copy to avoid modifying original
    processed_image = image_data.copy()
    background_array = None
    
    if subtract_background:
        try:
            with timing_context("Background estimation", logger):
                # Set default background parameters
                bg_kwargs = {
                    'mask': None,
                    'maskthresh': 0.0,
                    'filter_threshold': 0.0,
                    'filter_kernel': None
                }
                
                if background_kwargs:
                    bg_kwargs.update(background_kwargs)
                
                # Estimate background using SEP
                background = sep.Background(processed_image, **bg_kwargs)
                background_array = background.back()
                
                # Subtract background
                processed_image = processed_image - background_array
                
                logger.info(f"Background subtracted - global RMS: {background.globalrms:.3e}")
                
        except Exception as e:
            logger.warning(f"Background subtraction failed: {e}")
            if subtract_background:
                raise DataValidationError(f"Background subtraction failed: {e}")
    
    return processed_image, background_array


def create_progress_tracker(total_items: int, 
                          description: str = "Processing",
                          logger: Optional[logging.Logger] = None) -> 'tqdm':
    """
    Create a progress tracker for long-running operations.
    
    Parameters:
    -----------
    total_items : int
        Total number of items to process
    description : str, default="Processing"
        Description of the operation
    logger : logging.Logger, optional
        Logger to use for progress updates
        
    Returns:
    --------
    tqdm
        Progress bar object
    """
    try:
        from tqdm import tqdm
        return tqdm(total=total_items, desc=description, unit='items')
    except ImportError:
        # Fallback simple progress tracker
        return _SimpleProgressTracker(total_items, description, logger)


class _SimpleProgressTracker:
    """Simple fallback progress tracker when tqdm is not available."""
    
    def __init__(self, total: int, description: str, logger: Optional[logging.Logger]):
        self.total = total
        self.description = description
        self.logger = logger or logging.getLogger(__name__)
        self.current = 0
        self.last_reported = 0
    
    def update(self, n: int = 1):
        """Update progress by n items."""
        self.current += n
        progress_pct = (self.current / self.total) * 100
        
        # Report every 10%
        if progress_pct - self.last_reported >= 10:
            self.logger.info(f"{self.description}: {progress_pct:.0f}% complete ({self.current}/{self.total})")
            self.last_reported = progress_pct
    
    def close(self):
        """Finalize progress tracking."""
        if self.current >= self.total:
            self.logger.info(f"{self.description}: Complete ({self.total}/{self.total})")
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


def safe_divide(numerator: np.ndarray, 
               denominator: np.ndarray,
               fill_value: float = np.nan,
               min_denominator: float = 1e-10) -> np.ndarray:
    """
    Perform safe division with handling of zero denominators.
    
    Parameters:
    -----------
    numerator : numpy.ndarray
        Numerator array
    denominator : numpy.ndarray
        Denominator array
    fill_value : float, default=np.nan
        Value to use where denominator is zero or very small
    min_denominator : float, default=1e-10
        Minimum allowed denominator value
        
    Returns:
    --------
    numpy.ndarray
        Result of safe division
    """
    result = np.full_like(numerator, fill_value, dtype=float)
    valid_mask = np.abs(denominator) > min_denominator
    result[valid_mask] = numerator[valid_mask] / denominator[valid_mask]
    return result


def calculate_snr(flux: np.ndarray, 
                 flux_err: np.ndarray,
                 min_snr: float = 0.0) -> np.ndarray:
    """
    Calculate signal-to-noise ratio with error handling.
    
    Parameters:
    -----------
    flux : numpy.ndarray
        Flux measurements
    flux_err : numpy.ndarray
        Flux uncertainties
    min_snr : float, default=0.0
        Minimum SNR value (for clipping)
        
    Returns:
    --------
    numpy.ndarray
        Signal-to-noise ratios
    """
    snr = safe_divide(flux, flux_err, fill_value=0.0)
    snr = np.maximum(snr, min_snr)
    return snr


def apply_magnitude_zeropoint(flux: np.ndarray,
                            flux_err: np.ndarray,
                            zeropoint: float,
                            flux_units: str = 'microjansky') -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert flux measurements to magnitudes using a zeropoint.
    
    Parameters:
    -----------
    flux : numpy.ndarray
        Flux measurements
    flux_err : numpy.ndarray
        Flux uncertainties
    zeropoint : float
        Magnitude zeropoint
    flux_units : str, default='microjansky'
        Units of input flux
        
    Returns:
    --------
    tuple
        Magnitudes and magnitude uncertainties
    """
    # Handle negative or zero fluxes
    positive_mask = flux > 0
    
    magnitudes = np.full_like(flux, np.nan)
    mag_errors = np.full_like(flux_err, np.nan)
    
    # Convert positive fluxes to magnitudes
    if np.any(positive_mask):
        if flux_units.lower() == 'microjansky':
            # For microjanskys: mag = zp - 2.5 * log10(flux_uJy)
            magnitudes[positive_mask] = zeropoint - 2.5 * np.log10(flux[positive_mask])
        else:
            # Assume flux is already in the correct units for the zeropoint
            magnitudes[positive_mask] = -2.5 * np.log10(flux[positive_mask]) + zeropoint
        
        # Calculate magnitude errors: σ_mag = (2.5/ln(10)) * (σ_flux/flux)
        mag_errors[positive_mask] = (2.5 / np.log(10)) * (flux_err[positive_mask] / flux[positive_mask])
    
    return magnitudes, mag_errors


def save_results(results: Dict[str, Any], 
               output_file: Union[str, Path],
               format_type: str = 'yaml') -> None:
    """
    Save analysis results with multiple format support.

    Parameters:
    -----------
    results : dict
        Results dictionary to save
    output_file : str or Path
        Path to output file
    format_type : str, default='yaml'
        Output format ('yaml', 'json', 'pickle')
        
    Raises:
    -------
    ValueError
        If format is not supported
    """
    logger = logging.getLogger(__name__)
    output_path = Path(output_file)
    
    # Ensure output directory exists
    validate_output_directory(output_path.parent)
    
    try:
        if format_type.lower() == 'yaml':
            with open(output_path, 'w') as file:
                yaml.dump(results, file, default_flow_style=False, indent=2)
        elif format_type.lower() == 'json':
            import json
            with open(output_path, 'w') as file:
                json.dump(results, file, indent=2, default=str)
        elif format_type.lower() == 'pickle':
            import pickle
            with open(output_path, 'wb') as file:
                pickle.dump(results, file)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
        
        logger.info(f"Results saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
        raise


def check_dependencies() -> Dict[str, bool]:
    """
    Check if all required dependencies are available.
    
    Returns:
    --------
    dict
        Dictionary mapping package names to availability status
    """
    logger = logging.getLogger(__name__)
    
    required_packages = [
        'numpy', 'scipy', 'astropy', 'sep', 'photutils',
        'pypher', 'yaml', 'matplotlib', 'h5py'
    ]
    
    optional_packages = [
        'numba', 'sklearn', 'tqdm', 'PIL'
    ]
    
    results = {}
    
    for package in required_packages + optional_packages:
        try:
            __import__(package)
            results[package] = True
        except ImportError:
            results[package] = False
            if package in required_packages:
                logger.error(f"Required package '{package}' not found")
            else:
                logger.warning(f"Optional package '{package}' not found")
    
    return results


def get_memory_usage() -> float:
    """
    Get current memory usage in MB.
    
    Returns:
    --------
    float
        Memory usage in megabytes
    """
    try:
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    except:
        return 0.0


def setup_logger(name: str, 
                level: str = 'INFO', 
                log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up a logger for a specific module.
    
    Parameters:
    -----------
    name : str
        Logger name
    level : str, default='INFO'
        Logging level
    log_file : str, optional
        Path to log file
        
    Returns:
    --------
    logging.Logger
        Configured logger instance
    """
    return setup_logging(level, log_file, enable_colors=True)


def memory_monitor(func):
    """
    Decorator to monitor memory usage of functions.
    
    Parameters:
    -----------
    func : callable
        Function to monitor
        
    Returns:
    --------
    callable
        Wrapped function with memory monitoring
    """
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        with memory_monitor(f"{func.__name__}", logger):
            return func(*args, **kwargs)
    return wrapper


def validate_array(arr: np.ndarray, 
                  name: str = "Array",
                  min_shape: Optional[Tuple[int, ...]] = None,
                  allow_nan: bool = True) -> bool:
    """
    Validate numpy array properties.
    
    Parameters:
    -----------
    arr : numpy.ndarray
        Array to validate
    name : str, default="Array"
        Name for error messages
    min_shape : tuple, optional
        Minimum required shape
    allow_nan : bool, default=True
        Whether to allow NaN values
        
    Returns:
    --------
    bool
        True if validation passes
        
    Raises:
    -------
    DataValidationError
        If validation fails
    """
    if not isinstance(arr, np.ndarray):
        raise DataValidationError(f"{name} must be a numpy array, got {type(arr)}")
    
    if arr.size == 0:
        raise DataValidationError(f"{name} is empty")
    
    if min_shape and arr.shape < min_shape:
        raise DataValidationError(f"{name} shape {arr.shape} is smaller than required {min_shape}")
    
    if not allow_nan and np.any(np.isnan(arr)):
        raise DataValidationError(f"{name} contains NaN values")
    
    return True


# Legacy function compatibility
def save_catalog_fits(catalog: Table, output_directory: str) -> None:
    """Legacy function for backward compatibility."""
    warnings.warn("save_catalog_fits is deprecated. Use save_catalog instead.", 
                  DeprecationWarning, stacklevel=2)
    save_catalog(catalog, 'fits', output_directory)


def save_catalog_csv(catalog: Table, output_directory: str) -> None:
    """Legacy function for backward compatibility."""
    warnings.warn("save_catalog_csv is deprecated. Use save_catalog instead.", 
                  DeprecationWarning, stacklevel=2)
    save_catalog(catalog, 'csv', output_directory)
