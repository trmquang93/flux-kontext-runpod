"""
Image processing utilities for FLUX.1 Kontext image editing.
Adapted from qwen server with FLUX-specific enhancements.
"""

import base64
import io
import logging
from typing import Optional, Tuple, Union

from PIL import Image, ImageOps
import numpy as np


logger = logging.getLogger(__name__)


class ImageProcessor:
    """Handles image processing operations for FLUX.1 Kontext."""
    
    def __init__(self):
        # FLUX.1 Kontext optimized settings
        self.max_size = 2048
        self.min_size = 64
        self.supported_formats = {'PNG', 'JPEG', 'JPG', 'WEBP', 'BMP'}
        
        # FLUX.1 Kontext preferred dimensions (must be divisible by 8)
        self.preferred_sizes = [
            (1024, 1024),  # Square
            (1152, 896),   # Landscape
            (896, 1152),   # Portrait
            (1344, 768),   # Wide landscape
            (768, 1344),   # Tall portrait
        ]
    
    def base64_to_pil(self, base64_string: str) -> Image.Image:
        """Convert base64 string to PIL Image."""
        try:
            # Remove data URL prefix if present
            if base64_string.startswith('data:image/'):
                base64_string = base64_string.split(',', 1)[1]
            
            # Decode base64
            image_data = base64.b64decode(base64_string)
            
            # Open as PIL image
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            logger.debug(f"Converted base64 to PIL image: {image.size}, mode: {image.mode}")
            return image
            
        except Exception as e:
            logger.error(f"Failed to convert base64 to PIL image: {e}")
            raise ValueError(f"Invalid image data: {str(e)}")
    
    def pil_to_base64(self, image: Image.Image, format: str = 'JPEG', quality: int = 95) -> str:
        """Convert PIL Image to base64 string."""
        try:
            # Ensure RGB mode
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Save to bytes buffer
            buffer = io.BytesIO()
            
            # Use appropriate save parameters based on format
            save_kwargs = {'format': format.upper()}
            if format.upper() in ['JPEG', 'JPG']:
                save_kwargs.update({'quality': quality, 'optimize': True})
            elif format.upper() == 'PNG':
                save_kwargs.update({'optimize': True})
            
            image.save(buffer, **save_kwargs)
            buffer.seek(0)
            
            # Encode to base64
            base64_string = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            logger.debug(f"Converted PIL image to base64: {image.size}, format: {format}")
            return base64_string
            
        except Exception as e:
            logger.error(f"Failed to convert PIL image to base64: {e}")
            raise ValueError(f"Image conversion failed: {str(e)}")
    
    def validate_image(self, image: Image.Image) -> Tuple[bool, Optional[str]]:
        """Validate image size and format for FLUX.1 Kontext."""
        try:
            # Check image size
            width, height = image.size
            
            if width > self.max_size or height > self.max_size:
                return False, f"Image size {width}x{height} exceeds maximum {self.max_size}x{self.max_size}"
            
            if width < self.min_size or height < self.min_size:
                return False, f"Image size {width}x{height} is too small (minimum {self.min_size}x{self.min_size})"
            
            # Check format
            if hasattr(image, 'format') and image.format and image.format not in self.supported_formats:
                return False, f"Unsupported image format: {image.format}"
            
            # Check aspect ratio (shouldn't be too extreme for FLUX)
            aspect_ratio = width / height
            if aspect_ratio > 4.0 or aspect_ratio < 0.25:
                return False, f"Extreme aspect ratio {aspect_ratio:.2f} not recommended for FLUX.1 Kontext"
            
            return True, None
            
        except Exception as e:
            return False, f"Image validation error: {str(e)}"
    
    def find_optimal_size(self, original_width: int, original_height: int) -> Tuple[int, int]:
        """Find the optimal size from FLUX.1 Kontext preferred sizes."""
        original_aspect = original_width / original_height
        
        best_size = self.preferred_sizes[0]  # Default to 1024x1024
        best_diff = float('inf')
        
        for width, height in self.preferred_sizes:
            size_aspect = width / height
            aspect_diff = abs(original_aspect - size_aspect)
            
            if aspect_diff < best_diff:
                best_diff = aspect_diff
                best_size = (width, height)
        
        return best_size
    
    def resize_image(
        self,
        image: Image.Image,
        target_width: int = None,
        target_height: int = None,
        maintain_aspect_ratio: bool = True,
        use_optimal_size: bool = False
    ) -> Image.Image:
        """Resize image for FLUX.1 Kontext processing."""
        try:
            original_width, original_height = image.size
            
            # Use optimal size if requested
            if use_optimal_size:
                target_width, target_height = self.find_optimal_size(original_width, original_height)
            
            # Use defaults if not specified
            if target_width is None:
                target_width = 1024
            if target_height is None:
                target_height = 1024
            
            # Skip if already correct size
            if original_width == target_width and original_height == target_height:
                return image
            
            if maintain_aspect_ratio:
                # Calculate new size maintaining aspect ratio
                ratio = min(target_width / original_width, target_height / original_height)
                new_width = int(original_width * ratio)
                new_height = int(original_height * ratio)
            else:
                new_width = target_width
                new_height = target_height
            
            # Ensure dimensions are divisible by 8 for diffusion models
            new_width = (new_width // 8) * 8
            new_height = (new_height // 8) * 8
            
            # Ensure minimum size
            new_width = max(new_width, 512)
            new_height = max(new_height, 512)
            
            resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            logger.info(f"Resized image from {original_width}x{original_height} to {new_width}x{new_height}")
            return resized_image
            
        except Exception as e:
            logger.error(f"Failed to resize image: {e}")
            raise ValueError(f"Image resize failed: {str(e)}")
    
    def prepare_image_for_flux(self, image: Image.Image, target_size: Tuple[int, int] = (1024, 1024)) -> Image.Image:
        """Prepare image specifically for FLUX.1 Kontext processing."""
        try:
            # Ensure RGB mode
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Validate image
            is_valid, error_msg = self.validate_image(image)
            if not is_valid:
                raise ValueError(error_msg)
            
            # Apply image normalization
            image = self.normalize_image(image)
            
            # Resize to target size
            target_width, target_height = target_size
            image = self.resize_image(
                image, 
                target_width=target_width, 
                target_height=target_height,
                maintain_aspect_ratio=True
            )
            
            logger.info(f"Prepared image for FLUX.1 Kontext: {image.size}")
            return image
            
        except Exception as e:
            logger.error(f"Failed to prepare image for FLUX: {e}")
            raise
    
    def normalize_image(self, image: Image.Image) -> Image.Image:
        """Apply image normalization optimized for FLUX.1 Kontext."""
        try:
            # Auto-orient image based on EXIF data
            image = ImageOps.exif_transpose(image)
            
            # Convert to numpy for analysis
            image_array = np.array(image)
            
            # Check brightness and contrast
            mean_brightness = np.mean(image_array)
            std_brightness = np.std(image_array)
            
            # Log image statistics for debugging
            logger.debug(f"Image stats - Mean brightness: {mean_brightness:.1f}, Std: {std_brightness:.1f}")
            
            # Gentle contrast enhancement if image is too flat
            if std_brightness < 30:  # Very low contrast
                logger.info("Applying gentle contrast enhancement")
                image = ImageOps.autocontrast(image, cutoff=1)
            
            # Brightness adjustment warnings
            if mean_brightness < 60:
                logger.info("Image appears dark - FLUX.1 Kontext may enhance brightness")
            elif mean_brightness > 180:
                logger.info("Image appears bright - FLUX.1 Kontext may reduce brightness")
            
            return image
            
        except Exception as e:
            logger.warning(f"Image normalization warning: {e}")
            return image
    
    def post_process_image(self, image: Image.Image) -> Image.Image:
        """Post-process image after FLUX.1 Kontext generation."""
        try:
            # Ensure RGB mode
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Apply gentle sharpening if needed
            image_array = np.array(image)
            
            # Check if image needs sharpening
            gray = np.dot(image_array[...,:3], [0.2989, 0.5870, 0.1140])
            laplacian_var = np.var(np.gradient(gray))
            
            if laplacian_var < 100:  # Image might be soft
                logger.debug("Applying gentle sharpening")
                from PIL import ImageFilter
                image = image.filter(ImageFilter.UnsharpMask(radius=0.5, percent=50, threshold=2))
            
            logger.debug("Post-processing completed")
            return image
            
        except Exception as e:
            logger.warning(f"Post-processing warning: {e}")
            return image
    
    def create_thumbnail(
        self,
        image: Image.Image,
        size: Tuple[int, int] = (256, 256)
    ) -> Image.Image:
        """Create a thumbnail of the image."""
        try:
            thumbnail = image.copy()
            thumbnail.thumbnail(size, Image.Resampling.LANCZOS)
            
            logger.debug(f"Created thumbnail: {thumbnail.size}")
            return thumbnail
            
        except Exception as e:
            logger.error(f"Failed to create thumbnail: {e}")
            raise ValueError(f"Thumbnail creation failed: {str(e)}")
    
    def get_image_info(self, image: Image.Image) -> dict:
        """Get detailed information about an image."""
        try:
            info = {
                'size': image.size,
                'mode': image.mode,
                'format': getattr(image, 'format', 'Unknown'),
                'has_transparency': image.mode in ('RGBA', 'LA') or 'transparency' in image.info
            }
            
            # Add color analysis
            if image.mode == 'RGB':
                image_array = np.array(image)
                info.update({
                    'mean_brightness': float(np.mean(image_array)),
                    'brightness_std': float(np.std(image_array)),
                    'color_channels': {
                        'red_mean': float(np.mean(image_array[:,:,0])),
                        'green_mean': float(np.mean(image_array[:,:,1])),
                        'blue_mean': float(np.mean(image_array[:,:,2]))
                    }
                })
            
            return info
            
        except Exception as e:
            logger.warning(f"Could not analyze image: {e}")
            return {
                'size': image.size,
                'mode': image.mode,
                'format': 'Unknown'
            }