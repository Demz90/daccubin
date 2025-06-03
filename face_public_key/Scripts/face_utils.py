import os
import sys

import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

"""   
Face Detector Class
"""
import cv2
import numpy as np
import os # Still potentially useful if mp stores models locally, although not explicitly managed here
from typing import Union
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
    print(f"Image shape: {image.shape}")
        
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input file must be read as a valid BGR image.")

    try:
        # Use the MediaPipe specific detector
        detector = FaceDetector(
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
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
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
        A tuple containing:
        - A NumPy array (BGR) representing the image with the face rectangle.
        - A tuple representing the face bounding box (x, y, w, h).
        
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
    
    return image_with_rect, face_metadata

def preprocess_face_crop_resize(image: np.ndarray, face_metadata: tuple) -> Union[np.ndarray, None]:
    """
    Crops a face using the bounding box from face metadata and resizes to 112x112.
    Normalizes the image and centers the crop around the face.

    Args:
        image: The input image (BGR NumPy array).
        face_metadata: A tuple containing the face bounding box (x, y, width, height)
                       relative to the original image.

    Returns:
        A standardized 112x112 BGR uint8 NumPy array, or None if cropping/resizing fails.
    Raises:
        ValueError: If inputs are invalid.
    """
    target_size = (112, 112)
    
    if image is None:
        raise ValueError("Input image cannot be None")
    if not isinstance(face_metadata, (tuple, list)) or len(face_metadata) != 4:
        raise ValueError("face_metadata must be a tuple or list of (x, y, width, height)")

    try:
        x, y, w, h = map(int, face_metadata)  # Ensure integer coordinates

        # Calculate center of the face
        center_x = x + w // 2
        center_y = y + h // 2

        # Calculate new box coordinates centered around the face
        half_size = min(w, h) // 2
        x1 = max(0, center_x - half_size)
        y1 = max(0, center_y - half_size)
        x2 = min(image.shape[1], center_x + half_size)
        y2 = min(image.shape[0], center_y + half_size)

        # Check if the calculated box is valid
        if x1 >= x2 or y1 >= y2:
            print(f"Warning: Invalid crop dimensions calculated ({x1},{y1})->({x2},{y2}). Skipping face.")
            return None

        # Crop the face region
        face_crop = image[y1:y2, x1:x2]

        if face_crop.size == 0:
            print(f"Warning: Face crop resulted in an empty image. Skipping face.")
            return None

        # Resize directly to target size
        resized_face = cv2.resize(
            face_crop,
            target_size,
            interpolation=cv2.INTER_LINEAR
        )

        # Ensure output is uint8
        if resized_face.dtype != np.uint8:
            resized_face = resized_face.astype(np.uint8)

        return resized_face

    except Exception as e:
        # Catch any unexpected errors during cropping or resizing
        print(f"An error occurred during crop and resize preprocessing: {e}")
        raise e  # Re-raise unexpected errors
    
import cv2
import mediapipe as mp
import numpy as np
import math
from typing import Union, Dict, Tuple # Need these for type hints

# --- MediaPipe Setup (Initialize ONCE) ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh_instance = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)
# -----------------------------------------

# def get_landmarks_from_patch(face_patch_bgr: np.ndarray) -> Union[Dict[str, Tuple[int, int]], None]:
#     """
#     Runs MediaPipe Face Mesh on a face patch to extract landmarks.
#     (Using Union for Python < 3.10 compatibility)

#     Args:
#         face_patch_bgr: A 112x112 BGR uint8 NumPy array.

#     Returns:
#         A dictionary mapping landmark names (str) to (x, y) pixel coordinates,
#         or None if no face is detected or essential landmarks are missing.
#     """
#     if face_patch_bgr is None or face_patch_bgr.shape[:2] != (112, 112):
#         # print("Warning: Input patch is invalid for landmark detection.")
#         return None

#     height, width = face_patch_bgr.shape[:2]
#     image_rgb = cv2.cvtColor(face_patch_bgr, cv2.COLOR_BGR2RGB)
#     results = face_mesh.process(image_rgb)

#     landmarks_dict = {}
#     if results.multi_face_landmarks:
#         face_landmarks = results.multi_face_landmarks[0] 

#         def get_avg_point(indices):
#             valid_indices = [i for i in indices if 0 <= i < len(face_landmarks.landmark)] 
#             if not valid_indices: return None 

#             coords = [face_landmarks.landmark[i] for i in valid_indices]
#             valid_coords = [(lm.x, lm.y) for lm in coords] 
#             if not valid_coords: return None
            
#             avg_x = sum(x for x, y in valid_coords) / len(valid_coords)
#             avg_y = sum(y for x, y in valid_coords) / len(valid_coords)
#             return (int(avg_x * width), int(avg_y * height))

#         # Define indices 
#         left_eye_indices = [33, 160, 158, 133, 153, 144] 
#         right_eye_indices = [362, 385, 387, 263, 373, 380]
#         face_width_left_indices = [234] 
#         face_width_right_indices = [454]
#         nose_tip_indices = [1]
#         mouth_left_indices = [61]
#         mouth_right_indices = [291]
#         jaw_left_indices = [172]
#         jaw_right_indices = [397]
#         jaw_bottom_indices = [152]

#         # Store all potentially useful landmarks
#         landmarks_dict['left_eye'] = get_avg_point(left_eye_indices)
#         landmarks_dict['right_eye'] = get_avg_point(right_eye_indices)
#         landmarks_dict['face_width_left'] = get_avg_point(face_width_left_indices)
#         landmarks_dict['face_width_right'] = get_avg_point(face_width_right_indices)
#         landmarks_dict['nose_tip'] = get_avg_point(nose_tip_indices)
#         landmarks_dict['mouth_left'] = get_avg_point(mouth_left_indices)
#         landmarks_dict['mouth_right'] = get_avg_point(mouth_right_indices)
#         landmarks_dict['jaw_left'] = get_avg_point(jaw_left_indices)
#         landmarks_dict['jaw_right'] = get_avg_point(jaw_right_indices)
#         landmarks_dict['jaw_bottom'] = get_avg_point(jaw_bottom_indices)
        
#         # Filter out any landmarks that failed to compute (returned None)
#         valid_landmarks = {k: v for k, v in landmarks_dict.items() if v is not None}

#         if valid_landmarks:
#              return valid_landmarks
#         else:
#             return None 

#     return None

# --- Landmark Extraction Function ---
def get_landmarks_from_patch(face_patch_bgr: np.ndarray) -> Union[Dict[str, Union[Tuple[int, int], None]], None]:
    """
    Runs MediaPipe Face Mesh on a face patch to extract landmarks relevant for 
    various geometric features.

    Args:
        face_patch_bgr: A 112x112 BGR uint8 NumPy array.

    Returns:
        A dictionary mapping landmark names (str) to (x, y) pixel coordinates 
        or None if a specific landmark couldn't be computed. Returns None if no face detected.
    """
    global face_mesh_instance # Use the globally initialized instance
    if not _MEDIAPIPE_AVAILABLE or face_mesh_instance is None:
         print("Warning: MediaPipe Face Mesh is not available or not initialized.")
         return None
         
    if face_patch_bgr is None or face_patch_bgr.shape[:2] != (112, 112):
        return None

    height, width = face_patch_bgr.shape[:2]
    image_rgb = cv2.cvtColor(face_patch_bgr, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = face_mesh_instance.process(image_rgb)
    image_rgb.flags.writeable = True

    landmarks_dict = {}
    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0] 

        # Helper to get a single scaled point
        def get_scaled_point(index):
            if 0 <= index < len(face_landmarks.landmark):
                lm = face_landmarks.landmark[index]
                # Could add visibility/presence checks here if needed:
                # if lm.visibility < 0.5 or lm.presence < 0.5: return None 
                return (int(lm.x * width), int(lm.y * height))
            return None

        # Helper for averaged point
        def get_avg_point(indices):
            points = [get_scaled_point(i) for i in indices]
            valid_points = [p for p in points if p is not None]
            if not valid_points: return None
            avg_x = sum(p[0] for p in valid_points) / len(valid_points)
            avg_y = sum(p[1] for p in valid_points) / len(valid_points)
            return (int(avg_x), int(avg_y))
            
        # --- Define indices based on MediaPipe Face Mesh diagram ---
        # Eyes (averaged centers and specific points for EAR)
        left_eye_center_indices = [33, 160, 158, 133, 153, 144] 
        right_eye_center_indices = [362, 385, 387, 263, 373, 380]
        # EAR points (corners and vertical midpoints)
        left_eye_p1_idx = 33  # Outer corner
        left_eye_p4_idx = 133 # Inner corner
        left_eye_p2_idx = 159 # Top midpoint approximation
        left_eye_p3_idx = 145 # Bottom midpoint approximation
        right_eye_p1_idx = 263 # Outer corner 
        right_eye_p4_idx = 362 # Inner corner
        right_eye_p2_idx = 386 # Top midpoint approximation
        right_eye_p3_idx = 374 # Bottom midpoint approximation
        
        # Face Width points
        face_width_left_idx = 234
        face_width_right_idx = 454
        
        # Other potentially useful points
        nose_tip_idx = 1
        mouth_left_idx = 61
        mouth_right_idx = 291
        jaw_left_idx = 172
        jaw_right_idx = 397
        jaw_bottom_idx = 152

        # --- Store Landmarks ---
        # Averaged eye centers (for interocular distance)
        landmarks_dict['left_eye'] = get_avg_point(left_eye_center_indices)
        landmarks_dict['right_eye'] = get_avg_point(right_eye_center_indices)
        
        # Face width points
        landmarks_dict['face_width_left'] = get_scaled_point(face_width_left_idx)
        landmarks_dict['face_width_right'] = get_scaled_point(face_width_right_idx)

        # EAR points - store individually
        landmarks_dict['left_eye_p1'] = get_scaled_point(left_eye_p1_idx)
        landmarks_dict['left_eye_p4'] = get_scaled_point(left_eye_p4_idx)
        landmarks_dict['left_eye_p2'] = get_scaled_point(left_eye_p2_idx) # Using P2 index for top
        landmarks_dict['left_eye_p3'] = get_scaled_point(left_eye_p3_idx) # Using P3 index for bottom
        
        landmarks_dict['right_eye_p1'] = get_scaled_point(right_eye_p1_idx)
        landmarks_dict['right_eye_p4'] = get_scaled_point(right_eye_p4_idx)
        landmarks_dict['right_eye_p2'] = get_scaled_point(right_eye_p2_idx) # Using P2 index for top
        landmarks_dict['right_eye_p3'] = get_scaled_point(right_eye_p3_idx) # Using P3 index for bottom

        # Store others
        landmarks_dict['nose_tip'] = get_scaled_point(nose_tip_idx)
        landmarks_dict['mouth_left'] = get_scaled_point(mouth_left_idx)
        landmarks_dict['mouth_right'] = get_scaled_point(mouth_right_idx)
        landmarks_dict['jaw_left'] = get_scaled_point(jaw_left_idx)
        landmarks_dict['jaw_right'] = get_scaled_point(jaw_right_idx)
        landmarks_dict['jaw_bottom'] = get_scaled_point(jaw_bottom_idx)
        
        # --- Filter out None values --- 
        # (Important: Return only successfully computed landmarks)
        valid_landmarks = {k: v for k, v in landmarks_dict.items() if v is not None}

        if valid_landmarks:
             return valid_landmarks
        else:
            # No valid landmarks could be extracted at all
            return None 

    # No face detected by MediaPipe
    return None


def visualize_landmarks_on_patch(face_patch_bgr: np.ndarray, scale: int = 4) -> np.ndarray:
    """
    Detects landmarks on a 112x112 face patch, enlarges the patch, 
    and draws the detected landmarks with numbered labels. Prints the key to console.

    Args:
        face_patch_bgr: The 112x112 input face patch (BGR uint8).
        scale: Integer factor by which to enlarge the image for visualization.

    Returns:
        An enlarged BGR uint8 NumPy array with landmarks drawn and labeled,
        or an enlarged plain patch with text if detection fails.
    """
    if face_patch_bgr is None or face_patch_bgr.shape[:2] != (112, 112):
        raise ValueError("Input must be a 112x112 NumPy array.")

    # Enlarge the image for better visualization
    enlarged_size = (112 * scale, 112 * scale)
    enlarged_patch = cv2.resize(face_patch_bgr, enlarged_size, interpolation=cv2.INTER_LINEAR) 
    
    # Detect landmarks on the original small patch
    landmarks = get_landmarks_from_patch(face_patch_bgr)
    
    landmark_key_map = {} # To store number -> name mapping

    if landmarks:
        print(f"Found {len(landmarks)} landmark points/groups. Drawing labels:") # Updated print
        
        # Define font and colors for labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = scale * 0.1 # Smaller font scale based on enlargement
        font_color = (255, 255, 0) # Cyan color for text labels
        text_thickness = 1
        point_radius = max(1, scale // 2)
        point_color = (0, 255, 0) # Green for points
        
        # Iterate with enumerate to get an index (number) for each landmark
        for i, (name, point) in enumerate(landmarks.items()):
            number = i + 1 # Start numbering from 1
            landmark_key_map[number] = name # Store the mapping

            if point: # Check if the landmark was successfully extracted
                # Scale coordinates to the enlarged image size
                enlarged_x = int(point[0] * scale)
                enlarged_y = int(point[1] * scale)
                
                # Draw the point circle
                cv2.circle(enlarged_patch, (enlarged_x, enlarged_y), radius=point_radius, color=point_color, thickness=-1) 
                
                # Draw the number label slightly offset from the point
                text_offset_x = point_radius + 2 # Offset text slightly to the right
                text_offset_y = point_radius // 2 # Offset text slightly up
                cv2.putText(
                    enlarged_patch, 
                    str(number), 
                    (enlarged_x + text_offset_x, enlarged_y - text_offset_y), 
                    font, 
                    font_scale, 
                    font_color, 
                    text_thickness
                )
        
        # --- Print the key map to the console ---
        print("-" * 20)
        print("Landmark Key:")
        for num, name in landmark_key_map.items():
            print(f"  {num}: {name}")
        print("-" * 20)
        # -----------------------------------------

    else:
        print("Landmark detection failed on patch.")
        # Draw failure text on the image
        cv2.putText(
            enlarged_patch, 
            "No Landmarks Found", 
            (10, enlarged_size[1] // 2), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            fontScale=scale * 0.2, 
            color=(0, 0, 255), 
            thickness=1 + scale // 4
        )
        
    return enlarged_patch  