import os
import sys

# Add BASE_DIR to path
temp = os.path.dirname(os.path.abspath(__file__))
vals = temp.split('/')
BASE_DIR = '/'.join(vals[:-2])
BASE_DIR = '%s/' % BASE_DIR
sys.path.insert(0, BASE_DIR)

"""   
Face Detector Class
"""
import cv2
import numpy as np
import os # Still potentially useful if mp stores models locally, although not explicitly managed here
try:
    import mediapipe as mp
    _MEDIAPIPE_AVAILABLE = True
except ImportError:
    _MEDIAPIPE_AVAILABLE = False

class FaceDetector:
    """
    A face detector class that exclusively uses the MediaPipe Face Detection solution.
    """
    def __init__(self, mp_confidence: float = 0.5, model_selection: int = 1):
        """
        Initializes the MediaPipe face detector.

        Args:
            mp_confidence: Minimum detection confidence for MediaPipe (0-1). 
                           Detections below this are discarded by MediaPipe itself.
            model_selection: 0 for short-range model (best for faces within 2 meters),
                             1 for full-range model (faces up to 5 meters). Defaults to 1.

        Raises:
            ImportError: If the mediapipe library is not installed.
            RuntimeError: If the MediaPipe detector fails to initialize.
        """
        if not _MEDIAPIPE_AVAILABLE:
            raise ImportError("MediaPipe library is not installed. Please install it: pip install mediapipe")

        try:
            mp_face_detection = mp.solutions.face_detection
            self.mp_face_detector = mp_face_detection.FaceDetection(
                model_selection=model_selection,
                min_detection_confidence=mp_confidence
            )
            print(f"MediaPipe Face Detection initialized (model={model_selection}, confidence={mp_confidence}).")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize MediaPipe Face Detection: {e}")

    def detect(self, image: np.ndarray) -> list[tuple[int, int, int, int]]:
        """
        Detects faces in the input image using MediaPipe.

        Args:
            image: Input image (BGR format NumPy array).

        Returns:
            A list of tuples, where each tuple represents a detected face
            bounding box in the format (x, y, width, height). Returns an
            empty list if no faces are detected.
            
        Raises:
            ValueError: If the input image is invalid.
            RuntimeError: If an error occurs during MediaPipe processing.
        """
        if image is None or not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Input image must be a 3-channel BGR NumPy array.")
            
        h, w = image.shape[:2]
        faces = []
        
        try:
            # MediaPipe expects RGB image
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Process the image and find faces
            results = self.mp_face_detector.process(rgb_image)
            
            if results.detections:
                for detection in results.detections:
                    # Confidence is already filtered by min_detection_confidence in __init__
                    # We extract the bounding box directly
                    bboxC = detection.location_data.relative_bounding_box
                    
                    x = int(bboxC.xmin * w)
                    y = int(bboxC.ymin * h)
                    width = int(bboxC.width * w)
                    height = int(bboxC.height * h)
                    
                    # Ensure coordinates are within image bounds and dimensions are valid
                    x, y = max(0, x), max(0, y)
                    # Adjust width/height if they extend beyond image boundaries from the starting point
                    width = min(w - x, width) 
                    height = min(h - y, height)

                    if width > 0 and height > 0: # Ensure valid dimensions
                        faces.append((x, y, width, height))
        
        except Exception as e:
             # Catch potential errors during processing
             raise RuntimeError(f"Error during MediaPipe face detection processing: {e}")

        return faces
    
import cv2
import numpy as np
import os 
try:
    import mediapipe as mp
    _MEDIAPIPE_AVAILABLE = True
except ImportError:
    _MEDIAPIPE_AVAILABLE = False

# --- Assume MediaPipeFaceDetector class is defined here or imported ---
# class MediaPipeFaceDetector:
#     # ... (definition from previous response) ...
#     pass 
# -----------------------------------------------------------------------

# --- Function 1: Load Image and Detect Single Face Metadata (Updated) ---

def load_and_detect_single_face(
    image_path: str, 
    mp_confidence: float = 0.5, # Confidence for MediaPipe detector init
    model_selection: int = 1    # Model selection for MediaPipe detector init
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """
    Loads an image, detects exactly one face using MediaPipeFaceDetector, 
    and returns the image array and face bounding box.

    Args:
        image_path: Path to the image file.
        mp_confidence: Minimum confidence for MediaPipe detector initialization (0-1).
        model_selection: MediaPipe model selection (0 for short-range, 1 for full-range).

    Returns:
        A tuple containing:
            - The loaded image as a NumPy array (BGR).
            - The bounding box (x, y, w, h) of the single detected face.

    Raises:
        FileNotFoundError: If the image file does not exist.
        ValueError: If the loaded file is not a valid image, if no faces are 
                    detected, or if more than one face is detected.
        ImportError: If the mediapipe library is not installed.
        RuntimeError: If the MediaPipeFaceDetector fails to initialize or run.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found at: {image_path}")

    image = cv2.imread(image_path)
    
    if image is None:
        raise ValueError(f"Could not read or decode image file: {image_path}")
        
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input file must be read as a valid BGR image.")

    try:
        # Use the MediaPipe specific detector
        detector = MediaPipeFaceDetector(
            mp_confidence=mp_confidence, 
            model_selection=model_selection
        )
    except ImportError: # Propagate import error
        raise 
    except RuntimeError as e: # Catch init errors
        raise RuntimeError(f"Failed to initialize MediaPipeFaceDetector: {e}")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred during MediaPipeFaceDetector initialization: {e}")

    try:
        # Detect faces using the MediaPipe detector
        faces = detector.detect(image)
    except (ValueError, RuntimeError) as e: # Catch detection errors
        raise RuntimeError(f"Error during face detection: {e}")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred during face detection: {e}")


    # Check the number of detected faces
    if len(faces) == 0:
        raise ValueError(f"No faces detected in the image: {image_path}")
    elif len(faces) > 1:
        raise ValueError(f"Expected 1 face, but found {len(faces)} in image: {image_path}")
    
    # Exactly one face found
    face_bbox = faces[0]
    return image, face_bbox

# --- Function 2: Draw Rectangle on Image (Unchanged) ---

def draw_face_rectangle(
    image: np.ndarray, 
    face_bbox: tuple[int, int, int, int], 
    color: tuple[int, int, int] = (0, 255, 0), 
    thickness: int = 2
) -> np.ndarray:
    """
    Draws a rectangle on a copy of the image based on the face bounding box.
    (This function remains unchanged as it's independent of the detector)

    Args:
        image: The image as a NumPy array (BGR).
        face_bbox: The bounding box tuple (x, y, w, h).
        color: The color of the rectangle (B, G, R tuple). Defaults to green.
        thickness: The thickness of the rectangle lines. Defaults to 2.

    Returns:
        A new NumPy array (copy of the input image) with the rectangle drawn.
        
    Raises:
        ValueError: If the input image is invalid or bbox format is incorrect.
    """
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input 'image' must be a valid BGR NumPy array.")
    if not (isinstance(face_bbox, tuple) and len(face_bbox) == 4 and all(isinstance(n, int) for n in face_bbox)):
        raise ValueError("Input 'face_bbox' must be a tuple of 4 integers (x, y, w, h).")

    output_image = image.copy()
    x, y, w, h = face_bbox
    cv2.rectangle(output_image, (x, y), (x + w, y + h), color, thickness)
    return output_image

# --- Function 3: Orchestrator (Updated) ---

def process_image_for_single_face(
    image_path: str, 
    mp_confidence: float = 0.5,   # Pass confidence to detector init
    model_selection: int = 1,     # Pass model selection to detector init
    color: tuple[int, int, int] = (0, 255, 0), 
    thickness: int = 2
) -> np.ndarray:
    """
    Loads an image, detects a single face using MediaPipeFaceDetector, 
    and draws a rectangle around it.

    Args:
        image_path: Path to the image file.
        mp_confidence: Minimum confidence for MediaPipe detector initialization.
        model_selection: MediaPipe model selection (0 for short-range, 1 for full-range).
        color: Color for the bounding box (B, G, R).
        thickness: Thickness for the bounding box lines.

    Returns:
        A NumPy array (BGR) representing the image with the face rectangle.
        
    Raises:
        Propagates errors from load_and_detect_single_face and 
        draw_face_rectangle (e.g., FileNotFoundError, ValueError, ImportError, RuntimeError).
    """
    # Step 1: Load image and detect the single face's metadata using MediaPipe
    image_array, face_metadata = load_and_detect_single_face(
        image_path, 
        mp_confidence=mp_confidence, # Pass params to the updated function
        model_selection=model_selection 
    )
    
    # Step 2: Draw the rectangle using the metadata (this function call is unchanged)
    image_with_rect = draw_face_rectangle(
        image_array, 
        face_metadata, 
        color=color, 
        thickness=thickness
    )
    
    return image_with_rect

