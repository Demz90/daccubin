import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import numpy as np
from PIL import Image
import base64
import io

def generate_base_64_from_numpy_array(image_array: np.ndarray, image_format: str = "PNG") -> str:
    """
    Converts a NumPy array representing an image into a Base64 encoded string.

    Args:
        image_array: The NumPy array representing the image.
                     Assumes the array is in a format compatible with PIL
                     (e.g., HxWxC for RGB, HxW for grayscale).
        image_format: The format to save the image in before encoding (e.g., "PNG", "JPEG").
                      PNG is lossless and generally preferred unless file size is critical.

    Returns:
        A Base64 encoded string representation of the image.

    Raises:
        ValueError: If the image_format is not supported by PIL.
        TypeError: If the input is not a NumPy array.
    """
    if not isinstance(image_array, np.ndarray):
        raise TypeError(f"Expected input to be a numpy array, but got {type(image_array)}")

    try:
        # Determine the mode based on array shape, handle potential errors
        if image_array.ndim == 3:
            if image_array.shape[2] == 4:
                 mode = 'RGBA'
            elif image_array.shape[2] == 3:
                 mode = 'RGB'
            else:
                 raise ValueError(f"Unsupported number of channels: {image_array.shape[2]}")
        elif image_array.ndim == 2:
            mode = 'L' # Grayscale
        else:
            raise ValueError(f"Unsupported array dimensions: {image_array.ndim}")

        img = Image.fromarray(image_array, mode=mode)
        buffered = io.BytesIO()
        img.save(buffered, format=image_format)
        img_bytes = buffered.getvalue()
        base64_encoded = base64.b64encode(img_bytes)
        base64_string = base64_encoded.decode('utf-8')
        return base64_string
    except ValueError as e:
         # Catch PIL specific format errors or mode errors
        raise ValueError(f"Error processing image with PIL: {e}")
    except Exception as e:
        # General catch-all for unexpected errors during conversion
        raise RuntimeError(f"An unexpected error occurred during Base64 generation: {e}")

def generate_base_64_from_image_path(image_path: str) -> str:
    """
    Converts an image from a given file path into a Base64 encoded string.

    Args:
        image_path: The file path to the image.

    Returns:
        A Base64 encoded string representation of the image.

    Raises:
        FileNotFoundError: If the image file does not exist at the given path.
        ValueError: If the image format is not supported by PIL.
        Exception: For any other errors during the conversion process.
    """
    try:
        # Open the image file
        with Image.open(image_path) as img:
            buffered = io.BytesIO()
            # Infer the image format from the image object
            image_format = img.format
            img.save(buffered, format=image_format)
            img_bytes = buffered.getvalue()
            base64_encoded = base64.b64encode(img_bytes)
            base64_string = base64_encoded.decode('utf-8')
            return base64_string
    except FileNotFoundError:
        raise FileNotFoundError(f"Image file not found at path: {image_path}")
    except ValueError as e:
        raise ValueError(f"Error processing image with PIL: {e}")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred during Base64 generation: {e}")


def convert_base_64_to_numpy_array(base64_string: str) -> np.ndarray:
    """
    Converts a Base64 encoded string back into a NumPy array representing an image.

    Args:
        base64_string: The Base64 encoded string of the image.

    Returns:
        A NumPy array representing the reconstructed image.

    Raises:
        ValueError: If the base64 string is invalid or the image data cannot be parsed.
        TypeError: If the input is not a string.
    """
    if not isinstance(base64_string, str):
        raise TypeError(f"Expected input to be a string, but got {type(base64_string)}")

    try:
        base64_decoded_bytes = base64.b64decode(base64_string.encode('utf-8'))
        buffered_decoded = io.BytesIO(base64_decoded_bytes)
        reconstructed_img = Image.open(buffered_decoded)
        reconstructed_array = np.array(reconstructed_img)
        return reconstructed_array
    except (base64.binascii.Error, ValueError) as e:
        # Catch errors related to invalid base64 encoding or PIL unable to open data
        raise ValueError(f"Failed to decode Base64 string or parse image data: {e}")
    except Exception as e:
        # General catch-all for unexpected errors during reconstruction
        raise RuntimeError(f"An unexpected error occurred during NumPy array conversion: {e}")
    
# --- Image Helper Function ---
from typing import Optional

def decode_base64_to_image(base64_string: str) -> Optional[Image.Image]:
    """Decodes a Base64 string into a PIL Image object."""
    if not base64_string:
        print("Error: Received empty Base64 string.")
        return None
    try:
        image_bytes = base64.b64decode(base64_string.encode('utf-8'))
        buffer = io.BytesIO(image_bytes)
        img = Image.open(buffer)
        # It's good practice to load the image data immediately
        img.load()
        # Convert to RGB if it has an alpha channel for consistent display
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        return img
    except base64.binascii.Error as e:
        print(f"Error decoding Base64 data: {e}")
        return None
    except Exception as e:
        print(f"Error opening image from decoded Base64 string: {e}")
        return None