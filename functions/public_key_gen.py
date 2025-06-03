import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import numpy as np
"""
Generating The Hash
"""
# --- Global Constants ---
EMBEDDING_DIM = 512
N_FEATURES = 5
TOTAL_DIM = EMBEDDING_DIM + N_FEATURES  # 517
HASH_BITS = 256
# Fixed order for the 5 features
FEATURE_ORDER = sorted([
    'interocular_distance_ratio',
    'facial_width_height_ratio',
    'jaw_angle',
    'nose_mouth_distance_ratio',
    'face_symmetry_score'
])
# Fixed seed for reproducibility IN THIS SESSION.
# For production, generate ONCE and SAVE/LOAD this matrix.
PROJECTION_SEED = 42

import matplotlib.pyplot as plt
import face_utils
import geometrical_features
import face_embeddings
import public_key_gen
import milvus_utils 
import image_utils

def get_projection_matrix(rows, cols, seed):
    """
    Generates or retrieves the fixed random projection matrix.

    Args:
        rows (int): Number of rows (hash bits).
        cols (int): Number of columns (total vector dimension).
        seed (int): Random seed for reproducibility.

    Returns:
        np.ndarray: The random projection matrix (shape: rows x cols).
    """
    print(f"Generating projection matrix ({rows}x{cols}) with seed {seed}...")
    # Ensure the random state is reset and fixed for this generation
    rng = np.random.RandomState(seed)
    projection_matrix = rng.randn(rows, cols)
    print("Projection matrix generated.")
    return projection_matrix

def extract_and_order_features(metadata_dict):
    """
    Extracts the 5 facial features in a fixed order from a dictionary.

    Args:
        metadata_dict (dict): Dictionary containing raw feature values.
                                Keys must include those in FEATURE_ORDER.

    Returns:
        np.ndarray: A 1D NumPy array containing the 5 raw feature values
                    in the order defined by FEATURE_ORDER.
    """
    # Check if all required keys are present
    missing_keys = [key for key in FEATURE_ORDER if key not in metadata_dict]
    if missing_keys:
        raise ValueError(f"Metadata dictionary is missing keys: {missing_keys}")

    # Extract features in the fixed order
    raw_features = np.array([metadata_dict[key] for key in FEATURE_ORDER], dtype=np.float32)
    # print(f"Extracted raw features (shape {raw_features.shape}): {raw_features}")
    return raw_features

def create_combined_vector(embedding, raw_features, feature_weight=1.0):
    """
    Normalizes the embedding, applies weight to raw features, and concatenates.

    Args:
        embedding (np.ndarray): The 1D face embedding vector (shape EMBEDDING_DIM,).
        raw_features (np.ndarray): The 1D raw facial features vector (shape N_FEATURES,).
        feature_weight (float): The weight applied to the raw_features before concatenation.
                                Crucial for balancing scales. Default is 1.0.

    Returns:
        np.ndarray: The combined 1D vector (shape TOTAL_DIM,).
    """
    if embedding.shape != (EMBEDDING_DIM,):
        raise ValueError(f"Embedding must have shape ({EMBEDDING_DIM},), got {embedding.shape}")
    if raw_features.shape != (N_FEATURES,):
        raise ValueError(f"Raw features must have shape ({N_FEATURES},), got {raw_features.shape}")

    # 1. Normalize embedding (L2 norm)
    norm_embedding = embedding / np.linalg.norm(embedding)
    # Handle potential zero vector embedding case
    if np.all(norm_embedding == 0):
         print("Warning: Embedding vector is zero, normalization results in zero vector.")
         # Or handle as an error depending on requirements
         # raise ValueError("Cannot normalize zero vector embedding")

    # 2. Apply weight to features
    weighted_features = feature_weight * raw_features
    # print(f"Weighted features (weight={feature_weight}): {weighted_features}")

    # 3. Concatenate
    combined_vector = np.concatenate((norm_embedding, weighted_features))
    # print(f"Combined vector shape: {combined_vector.shape}") # Should be (517,)
    return combined_vector

def generate_public_key(embedding, metadata_dict, projection_matrix, feature_weight=1.0):
    """
    Generates the 256-bit similarity-preserving public key hex string.

    Args:
        embedding (np.ndarray): The 1D face embedding vector.
        metadata_dict (dict): Dictionary with raw facial features.
        projection_matrix (np.ndarray): The fixed random projection matrix (P).
        feature_weight (float): Weight for raw features. Default 1.0.

    Returns:
        str: The 64-character hexadecimal representation of the 256-bit key.
    """
    # 1. Extract features
    raw_features = extract_and_order_features(metadata_dict)

    # 2. Create combined vector
    combined_vector = create_combined_vector(embedding, raw_features, feature_weight)

    # 3. Compute dot products (LSH projection)
    dots = np.dot(projection_matrix, combined_vector) # Shape: (HASH_BITS,)

    # 4. Generate hash bits (SimHash quantization)
    hash_bits = (dots >= 0).astype(np.uint8) # Array of 256 zeros/ones

    # 5. Pack bits into bytes
    # np.packbits expects bits in MSB order if interpreted as bytes.
    # The order might not matter for Hamming distance, but consistency does.
    packed_bytes = np.packbits(hash_bits) # Results in 32 bytes

    # 6. Convert bytes to hex string
    hex_key = packed_bytes.tobytes().hex()

    # print(f"Generated key: {hex_key}")
    return hex_key

def compare_keys(key1_hex, key2_hex):
    """
    Compares two public keys by calculating the Hamming distance.

    Args:
        key1_hex (str): The hex string of the first key (64 chars).
        key2_hex (str): The hex string of the second key (64 chars).

    Returns:
        int: The Hamming distance (number of differing bits, 0-256).
             Returns -1 if keys have invalid format.
    """
    if len(key1_hex) != 64 or len(key2_hex) != 64:
        print("Error: Keys must be 64-character hex strings.")
        return -1
    try:
        # Convert hex strings back to bytes
        bytes1 = bytes.fromhex(key1_hex)
        bytes2 = bytes.fromhex(key2_hex)

        if len(bytes1) != HASH_BITS // 8 or len(bytes2) != HASH_BITS // 8:
             print(f"Error: Decoded keys must be {HASH_BITS // 8} bytes long.")
             return -1

        # Calculate Hamming distance bit by bit
        distance = 0
        for b1, b2 in zip(bytes1, bytes2):
            # XOR finds differing bits (result has 1 where bits differ)
            diff = b1 ^ b2
            # Count the number of set bits (1s) in the difference byte
            distance += bin(diff).count('1')

        return distance
    except ValueError:
        print("Error: Invalid hex character in keys.")
        return -1

def generate_public_key_from_features_and_embeddings(image_face_embedding, facial_features):
    import numpy as np
    import hashlib  # Needed for Hamming distance calculation later, though not SHA256

    P = get_projection_matrix(HASH_BITS, TOTAL_DIM, PROJECTION_SEED)
    # print(f"Shape of P: {P.shape}")  # Should be (256, 517)
    # print(f"Feature order: {FEATURE_ORDER}")
    # print(f"Total vector dimension: {TOTAL_DIM}")
    # print(f"Target hash bits: {HASH_BITS}")

    feature_w = 0.01
    # feature_w = 0.1
    # print(f"\n--- Generating Key 1 (Weight: {feature_w}) ---")
    face_public_key = generate_public_key(image_face_embedding, facial_features, P, feature_weight=feature_w)
    # print(f"Generated Key 1: {key1}")

    return face_public_key

def generate_face_public_key_from_image(image_file_path):
    """
    Processes an input image to detect a single face, extract features, and generate a face public key.

    Args:
        image_file_path (str): The file path to the input image.

    Returns:
        str: The generated face public key.
    """
    # 1 - Detect face
    final_image, face_metadata = face_utils.process_image_for_single_face(
        image_file_path, 
        mp_confidence=0.1,  # Example: override default confidence
        model_selection=0
    )

    # 2 - Preprocess face
    resized_face = face_utils.preprocess_face_crop_resize(final_image, face_metadata)

    # 3 - Extract facial landmarks
    valid_landmarks = face_utils.get_landmarks_from_patch(resized_face)
    facial_features = geometrical_features.calculate_all_geometric_features(valid_landmarks)
    # print("Visualizing landmarks on face")
    # visualized_image = face_utils.visualize_landmarks_on_patch(resized_face, scale=4)  # Enlarge 4x
    # plt.imshow(visualized_image)

    # 4 - Generate face embedding
    image_face_embedding = face_embeddings.get_facenet_embedding_from_image(resized_face)

    # 5 - Generate face public key
    face_public_key = public_key_gen.generate_public_key_from_features_and_embeddings(image_face_embedding, facial_features)

    return face_public_key

