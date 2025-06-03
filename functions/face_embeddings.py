import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

# face_embeddings.py
# Uses the keras-facenet library for simpler embedding generation.
# Remember to install: pip install keras-facenet tensorflow opencv-python

import numpy as np
import cv2
import os
from keras_facenet import FaceNet

# Initialize the FaceNet embedder globally or within the function as needed.
# Global initialization avoids reloading the model on every call.
try:
    print("Initializing FaceNet embedder...")
    embedder = FaceNet()
    print("FaceNet embedder initialized successfully.")
except Exception as e:
    print(f"Error initializing FaceNet embedder: {e}")
    print("Please ensure keras-facenet and its dependencies (like tensorflow) are installed correctly.")
    embedder = None # Set to None if initialization fails

def get_facenet_embedding_from_image(image):
    """
    Generates a FaceNet embedding for a single pre-cropped face image.

    Args:
        image (numpy.ndarray): A pre-cropped face image (BGR format from cv2.imread is usually fine).

    Returns:
        list: The list of embedding vectors, or None if initialization failed, input is invalid, or embedding generation fails.
    """
    if embedder is None:
        print("Error: FaceNet embedder was not initialized successfully.")
        return None

    if not isinstance(image, np.ndarray):
        print("Error: Input must be a NumPy array image.")
        return None

    try:
        print(f"Generating embedding for the input image...")
        # The library expects a list, so wrap the single image in a list
        # It handles preprocessing (resizing, normalization) internally.
        embeddings_list = embedder.embeddings([image]) # Expecting a list like [array]
        print("Embedding generated successfully.")
        return embeddings_list[0]

    except Exception as e:
        # This catches errors *during* the embedder.embeddings call or subsequent processing
        print(f"Error generating embedding: {e}")
        return None
