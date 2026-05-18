import logging
import os

logger = logging.getLogger(__name__)
from pathlib import Path
from datetime import datetime

def load_config(config_path: Path | None) -> dict:
    """Load configuration from a .env file.

    Returns default config if config_path is None.
    Raises a ValueError if the file does not exist.
    """
    from dotenv import dotenv_values
    
    default_config = {
        "weights": {"composition": 0.6, "sharpness": 0.4},
        "burst_time_threshold": 2.0,
        "gpu_enabled": True
    }
    
    if not config_path:
        return default_config
        
    if not config_path.is_file():
        raise ValueError(f"Config file not found: {config_path}")
        
    # Read dotenv values
    env_vars = dotenv_values(config_path)
    
    if 'BURST_TIME_THRESHOLD' in env_vars and env_vars['BURST_TIME_THRESHOLD']:
        try:
            val = float(env_vars['BURST_TIME_THRESHOLD'])
            if val <= 0:
                raise ValueError('BURST_TIME_THRESHOLD must be > 0')
            default_config['burst_time_threshold'] = val
        except ValueError as e:
            logger.warning('Invalid BURST_TIME_THRESHOLD in config: %s', e)

    if 'WEIGHT_COMPOSITION' in env_vars and env_vars['WEIGHT_COMPOSITION']:
        try:
            val = float(env_vars['WEIGHT_COMPOSITION'])
            if not (0.0 <= val <= 1.0):
                raise ValueError('WEIGHT_COMPOSITION must be in [0, 1]')
            default_config['weights']['composition'] = val
        except ValueError as e:
            logger.warning('Invalid WEIGHT_COMPOSITION in config: %s', e)

    if 'WEIGHT_SHARPNESS' in env_vars and env_vars['WEIGHT_SHARPNESS']:
        try:
            val = float(env_vars['WEIGHT_SHARPNESS'])
            if not (0.0 <= val <= 1.0):
                raise ValueError('WEIGHT_SHARPNESS must be in [0, 1]')
            default_config['weights']['sharpness'] = val
        except ValueError as e:
            logger.warning('Invalid WEIGHT_SHARPNESS in config: %s', e)
            
    if 'GPU_ENABLED' in env_vars and env_vars['GPU_ENABLED']:
        val = env_vars['GPU_ENABLED'].lower()
        if val in ('true', '1', 'yes', 'on'):
            default_config['gpu_enabled'] = True
        elif val in ('false', '0', 'no', 'off'):
            default_config['gpu_enabled'] = False

    return default_config

def hash_image(image_path: Path) -> str:
    """Calculate perceptual hash (phash) for an image.
    Returns a hex string; empty string on failure.
    """
    try:
        from PIL import Image
        import imagehash
        with Image.open(image_path) as img:
            return str(imagehash.phash(img))
    except Exception:
        logger.debug('Failed to hash image: %s', image_path, exc_info=True)
        return ""

def move_to(src: Path, dest_dir: Path) -> None:
    """Move a file to the destination directory, creating it if needed.
    Raises an error if a file with the same name already exists to avoid accidental overwrites.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    if dest.exists():
        raise FileExistsError(f"Destination file already exists and would be overwritten: {dest}")
    src.replace(dest)

def get_exif_time(image_path: Path) -> datetime | None:
    """Extract original capture time from EXIF data.
    Falls back to file modification time if EXIF is missing.
    Returns None if file cannot be read.
    """
    if not image_path.exists():
        return None
        
    try:
        from PIL import Image
        from PIL.ExifTags import Base
        
        with Image.open(image_path) as img:
            exif = img.getexif()
            if exif:
                time_str = None
                
                # Check ExifIFD (where DateTimeOriginal usually lives)
                exif_ifd = exif.get_ifd(Base.ExifOffset) if hasattr(Base, 'ExifOffset') else exif.get_ifd(34665)
                if exif_ifd and 36867 in exif_ifd:
                    time_str = exif_ifd[36867]
                elif 306 in exif: # Fallback to DateTime in 0th IFD
                    time_str = exif[306]
                elif 36867 in exif:
                    time_str = exif[36867]
                    
                if time_str:
                    # Handle bytes if returned
                    if isinstance(time_str, bytes):
                        time_str = time_str.decode('utf-8').rstrip('\x00')
                    # Format is usually 'YYYY:MM:DD HH:MM:SS'
                    return datetime.strptime(time_str.strip(), "%Y:%m:%d %H:%M:%S")
    except Exception:
        logger.debug('Failed to read EXIF from: %s', image_path, exc_info=True)

    # Fallback to modification time
    try:
        mtime = os.path.getmtime(image_path)
        return datetime.fromtimestamp(mtime)
    except Exception:
        logger.debug('Failed to read mtime from: %s', image_path, exc_info=True)
        return None
