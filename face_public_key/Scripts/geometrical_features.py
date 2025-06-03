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

import math
from typing import Union, Dict, Tuple # Or remove Union if using Python 3.10+

def calculate_interocular_distance_ratio(landmarks: Union[Dict[str, Tuple[int, int]], None]) -> Union[float, None]:
    """
    Calculates the ratio of the distance between the eyes to the face width,
    using specific landmarks.

    Args:
        landmarks: A dictionary of landmark points obtained from 
                   get_landmarks_from_patch, potentially None. Expected keys: 
                   'left_eye', 'right_eye', 'face_width_left', 'face_width_right'.

    Returns:
        The calculated ratio as a float, or None if calculation fails, 
        landmarks are missing, or input dict is None.
    """
    if not landmarks:
        # print("Debug: landmarks dictionary is None.") # Optional debug
        return None

    # Use .get() for safer access in case keys are unexpectedly missing
    le = landmarks.get('left_eye')
    re = landmarks.get('right_eye')
    fwl = landmarks.get('face_width_left')
    fwr = landmarks.get('face_width_right')

    # Check if all required points were found by .get()
    if not all([le, re, fwl, fwr]):
        # print("Debug: Missing one or more required landmarks for ratio calculation.") # Optional debug
        return None

    try:
        # Calculate Euclidean distance using math.sqrt for compatibility
        eye_dist_sq = (re[0] - le[0])**2 + (re[1] - le[1])**2
        face_width_sq = (fwr[0] - fwl[0])**2 + (fwr[1] - fwl[1])**2
        
        # Avoid sqrt(0) and division by zero
        if face_width_sq <= 0: 
            # print("Warning: Calculated face width squared is zero or negative.")
            return None
            
        eye_dist = math.sqrt(eye_dist_sq)
        face_width = math.sqrt(face_width_sq)

        ratio = eye_dist / face_width
        return ratio
        
    except TypeError as te:
        # Catch potential errors if coordinates are not numbers
        print(f"Error calculating interocular distance ratio: Invalid coordinate types - {te}")
        return None
    except Exception as e:
        # Catch any other unexpected calculation errors
        print(f"Error calculating interocular distance ratio: {e}")
        return None

def calculate_facial_width_height_ratio(landmarks: Union[Dict[str, Tuple[int, int]], None]) -> Union[float, None]:
    """
    Calculates the ratio of face width (cheek landmarks) to face height 
    (eye midpoint to jaw bottom).

    Args:
        landmarks: Dictionary potentially containing 'face_width_left', 'face_width_right',
                   'left_eye', 'right_eye', 'jaw_bottom'.

    Returns:
        The width/height ratio as a float, or None on failure.
    """
    if not landmarks: return None

    fwl = landmarks.get('face_width_left')
    fwr = landmarks.get('face_width_right')
    le = landmarks.get('left_eye')
    re = landmarks.get('right_eye')
    jb = landmarks.get('jaw_bottom')

    if not all([fwl, fwr, le, re, jb]):
        return None

    try:
        # Calculate width
        face_width_sq = (fwr[0] - fwl[0])**2 + (fwr[1] - fwl[1])**2
        if face_width_sq <= 1e-6: return None
        face_width = math.sqrt(face_width_sq)

        # Calculate height
        eye_mid_y = (le[1] + re[1]) / 2.0
        face_height = abs(jb[1] - eye_mid_y)
        if face_height <= 1e-6: return None # Height is essentially zero

        ratio = face_width / face_height
        return ratio

    except TypeError as te:
        print(f"Error calculating width/height ratio: Invalid coordinate types - {te}")
        return None
    except Exception as e:
        print(f"Error calculating width/height ratio: {e}")
        return None

def calculate_jaw_angle(landmarks: Union[Dict[str, Tuple[int, int]], None]) -> Union[float, None]:
    """
    Calculates the angle of the jaw formed at the chin landmark.

    Args:
        landmarks: Dictionary potentially containing 'jaw_left', 'jaw_bottom', 'jaw_right'.

    Returns:
        The jaw angle in degrees, or None on failure.
    """
    if not landmarks: return None

    jl = landmarks.get('jaw_left')
    jb = landmarks.get('jaw_bottom')
    jr = landmarks.get('jaw_right')

    if not all([jl, jb, jr]):
        return None

    try:
        # Vectors from jaw_bottom to jaw_left/right
        v1 = (jl[0] - jb[0], jl[1] - jb[1])
        v2 = (jr[0] - jb[0], jr[1] - jb[1])

        # Magnitudes squared
        v1_mag_sq = v1[0]**2 + v1[1]**2
        v2_mag_sq = v2[0]**2 + v2[1]**2

        # Check for zero length vectors
        if v1_mag_sq <= 1e-6 or v2_mag_sq <= 1e-6:
            return None

        v1_mag = math.sqrt(v1_mag_sq)
        v2_mag = math.sqrt(v2_mag_sq)

        # Dot product
        dot_product = v1[0] * v2[0] + v1[1] * v2[1]

        # Cosine of the angle, clamped for numerical stability
        cos_angle = max(-1.0, min(1.0, dot_product / (v1_mag * v2_mag)))

        # Angle in radians
        angle_rad = math.acos(cos_angle)

        # Convert to degrees
        angle_deg = math.degrees(angle_rad)

        return angle_deg

    except Exception as e:
        print(f"Error calculating jaw angle: {e}")
        return None

def calculate_nose_mouth_distance_ratio(landmarks: Union[Dict[str, Tuple[int, int]], None]) -> Union[float, None]:
    """
    Calculates the ratio of nose-to-mouth-center distance to inter-ocular distance.

    Args:
        landmarks: Dictionary potentially containing 'nose_tip', 'mouth_left', 
                   'mouth_right', 'left_eye', 'right_eye'.

    Returns:
        The calculated ratio as a float, or None on failure.
    """
    if not landmarks: return None

    nt = landmarks.get('nose_tip')
    ml = landmarks.get('mouth_left')
    mr = landmarks.get('mouth_right')
    le = landmarks.get('left_eye')
    re = landmarks.get('right_eye')

    if not all([nt, ml, mr, le, re]):
        return None

    try:
        # Calculate mouth center
        mouth_center_x = (ml[0] + mr[0]) / 2.0
        mouth_center_y = (ml[1] + mr[1]) / 2.0

        # Calculate nose-to-mouth distance
        nose_mouth_dist_sq = (nt[0] - mouth_center_x)**2 + (nt[1] - mouth_center_y)**2
        
        # Calculate inter-ocular distance
        eye_dist_sq = (re[0] - le[0])**2 + (re[1] - le[1])**2

        if eye_dist_sq <= 1e-6: # Avoid division by zero
            return None

        nose_mouth_dist = math.sqrt(nose_mouth_dist_sq)
        eye_dist = math.sqrt(eye_dist_sq)
        
        ratio = nose_mouth_dist / eye_dist
        return ratio

    except Exception as e:
        print(f"Error calculating nose-mouth distance ratio: {e}")
        return None

def calculate_face_symmetry_score(landmarks: Union[Dict[str, Tuple[int, int]], None]) -> Union[float, None]:
    """
    Calculates a face symmetry score based on horizontal distances of paired 
    landmarks from a vertical centerline. Score near 0 indicates higher symmetry.

    Args:
        landmarks: Dictionary potentially containing 'left_eye', 'right_eye', 
                   'face_width_left', 'face_width_right', 'mouth_left', 'mouth_right',
                   'jaw_left', 'jaw_right', 'nose_tip', 'jaw_bottom'.

    Returns:
        The symmetry score (average normalized horizontal distance difference) 
        as a float, or None on failure or if no pairs are available.
    """
    if not landmarks: return None

    # Points needed for centerline and width
    nt = landmarks.get('nose_tip')
    jb = landmarks.get('jaw_bottom')
    fwl = landmarks.get('face_width_left')
    fwr = landmarks.get('face_width_right')

    if not all([nt, jb, fwl, fwr]):
        return None # Cannot calculate centerline or normalize

    try:
        # Calculate centerline X
        centerline_x = (nt[0] + jb[0]) / 2.0
        
        # Calculate face width
        face_width_sq = (fwr[0] - fwl[0])**2 + (fwr[1] - fwl[1])**2
        if face_width_sq <= 1e-6: return None
        face_width = math.sqrt(face_width_sq)

        # Landmark pairs to compare [ (left_key, right_key), ... ]
        pairs = [
            ('left_eye', 'right_eye'),
            ('face_width_left', 'face_width_right'),
            ('mouth_left', 'mouth_right'),
            ('jaw_left', 'jaw_right')
        ]

        symmetry_diffs = []
        for left_key, right_key in pairs:
            p_left = landmarks.get(left_key)
            p_right = landmarks.get(right_key)

            # Calculate score only if both points in the pair exist
            if p_left and p_right:
                h_dist_left = abs(p_left[0] - centerline_x)
                h_dist_right = abs(p_right[0] - centerline_x)
                
                # Normalized difference for this pair
                normalized_diff = abs(h_dist_left - h_dist_right) / face_width
                symmetry_diffs.append(normalized_diff)

        # Calculate average score if any pairs were processed
        if not symmetry_diffs:
            # print("Warning: No complete landmark pairs found for symmetry calculation.")
            return None 

        average_symmetry_score = sum(symmetry_diffs) / len(symmetry_diffs)
        return average_symmetry_score

    except Exception as e:
        print(f"Error calculating face symmetry score: {e}")
        return None
    

def calculate_all_geometric_features(landmarks: Union[Dict[str, Tuple[int, int]], None]) -> Dict[str, Union[float, None]]:
    """
    Calculates all implemented geometric features based on the provided landmarks.

    Args:
        landmarks: A dictionary of landmark points obtained from get_landmarks_from_patch,
                   potentially None.

    Returns:
        A dictionary where keys are feature names (str) and values are the 
        calculated feature values (float) or None if the calculation failed 
        or landmarks were missing. Returns an empty dict if input landmarks is None.
    """
    if not landmarks:
        return {} # Return empty dict if no landmarks provided

    features = {}

    # Call each individual calculation function
    features['interocular_distance_ratio'] = calculate_interocular_distance_ratio(landmarks)
    # Note: EAR requires specific p1-p4 points, ensure get_landmarks_from_patch provides them if using calculate_average_ear
    # features['eye_aspect_ratio'] = calculate_average_ear(landmarks) # Assuming EAR function is available and landmarks dict has EAR points
    features['facial_width_height_ratio'] = calculate_facial_width_height_ratio(landmarks)
    features['jaw_angle'] = calculate_jaw_angle(landmarks)
    features['nose_mouth_distance_ratio'] = calculate_nose_mouth_distance_ratio(landmarks)
    features['face_symmetry_score'] = calculate_face_symmetry_score(landmarks)
    
    # Add calls for other implemented geometric features here...
    # e.g., features['cheekbone_prominence'] = calculate_cheekbone_prominence(landmarks)

    return features