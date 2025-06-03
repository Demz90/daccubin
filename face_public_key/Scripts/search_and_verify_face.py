import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import argparse
import base64
import io
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from pymongo import MongoClient
from bson.objectid import ObjectId
from pymilvus import connections, utility, Collection # Assuming these are needed directly

# Assuming these modules are accessible from the script's location
# Adjust path if necessary, e.g., using sys.path.append
import milvus_utils
import public_key_gen
# import image_utils # Not strictly needed if only displaying input image directly

# --- Constants ---
MONGO_DB_NAME = "theaccubin_images"
MONGO_COLLECTION_NAME = "image_storage"
MILVUS_COLLECTION_NAME = "face_public_keys"

# --- MongoDB Helper Functions (Copied for self-containment) ---
def initialise_mongo_cloud_db_client():
    """Initializes a MongoDB client."""
    # Replace with secure credential handling (e.g., environment variables) if possible
    mongo_server_public_ip_address = '95.217.233.18'
    mongodb_server_username = 'utom_admin'
    mongodb_server_password = 'utom_secure_2024'
    CONNECTION_STRING = f"mongodb://{mongodb_server_username}:{mongodb_server_password}@{mongo_server_public_ip_address}:27017/"
    try:
        client = MongoClient(CONNECTION_STRING)
        # Test connection
        client.admin.command('ping')
        # print("MongoDB connection successful.")
        return client
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        raise

def get_document_dict_for_mongo_db_collection_by_object_id(client, db_name, collection_name, object_id):
    """Retrieve a document from MongoDB by its ObjectID."""
    if not client:
        print("MongoDB client is not initialized.")
        return None
    try:
        db = client[db_name]
        collection = db[collection_name]
        # Ensure object_id is an ObjectId instance
        if isinstance(object_id, str):
            object_id = ObjectId(object_id)
        doc_dict = collection.find_one({"_id": object_id})
        return doc_dict
    except Exception as e:
        print(f"Error retrieving document from MongoDB (ID: {object_id}): {e}")
        return None

# --- Image Helper Function ---
def decode_base64_to_image(base64_string: str) -> Image.Image | None:
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

# --- Main Search and Verification Logic ---
def search_and_verify(image_path: str):
    """
    Generates a public key, searches Milvus, retrieves the corresponding image
    from MongoDB via Base64 ID, and displays the input and retrieved images.
    """
    if not os.path.exists(image_path):
        print(f"Error: Input image path does not exist: {image_path}")
        return

    print(f"Processing image: {image_path}")

    # 1. Connect to Services
    print("Connecting to Milvus...")
    try:
        milvus_utils.connect_to_milvus() # Assumes connection details are in env vars
        milvus_collection = milvus_utils.get_collection(MILVUS_COLLECTION_NAME)
        # Ensure collection is loaded for searching
        milvus_collection.load()
        print(f"Connected to Milvus and loaded collection '{MILVUS_COLLECTION_NAME}'.")
    except Exception as e:
        print(f"Error connecting to Milvus or getting collection: {e}")
        return

    # 2. Generate Query Vector
    print("Generating face public key...")
    try:
        hex_key = public_key_gen.generate_face_public_key_from_image(image_path)
        if not hex_key:
             print("Error: Failed to generate face public key.")
             return
        query_vector_binary = bytes.fromhex(hex_key)
        print(f"Generated query vector (first 16 bytes): {query_vector_binary[:16].hex()}...")
    except ValueError:
        print(f"Error: Invalid hex key generated: {hex_key}")
        return
    except Exception as e:
        print(f"Error during public key generation: {e}")
        return

    # 3. Perform Milvus Search
    print("Searching Milvus for the closest match...")
    # search_params = {
    #     "metric_type": "HAMMING",
    #     "params": {"nprobe": 16} # Adjust nprobe based on index and performance needs
    # }
    # try:
    #     results = milvus_collection.search(
    #         data=[query_vector_binary],
    #         anns_field="face_public_key_vector",
    #         param=search_params,
    #         limit=1,
    #         output_fields=["image_base_64_id", "NIN"] # Request NIN as well for context
    #     )
    # except Exception as e:
    #     print(f"Error during Milvus search: {e}")
    #     return

    # Use the new utility function
    try:
        closest_matches = milvus_utils.find_closest_face_key(milvus_collection, hex_key, top_k=1)
    except Exception as e:
        print(f"Error during Milvus search via utility function: {e}")
        return

    # 4. Process Search Results
    # if not results or not results[0]:
    #     print("No match found in Milvus collection.")
    #     return
    if not closest_matches:
        print("No match found in Milvus collection via utility function.")
        return

    # Unpack the first result from the utility function
    # mongo_id_str, retrieved_nin, distance = closest_matches[0]

    # Get the dictionary for the closest match
    closest_match_data = closest_matches[0]

    # # Extract necessary info using .get() for safety
    # mongo_id_str = closest_match_data.get("image_base_64_id")
    # retrieved_nin = closest_match_data.get("NIN", "N/A")
    # distance = closest_match_data.get("distance", float('inf')) # Get distance, default to infinity if missing
    # # You can access other fields similarly if needed:
    # # first_name = closest_match_data.get("first_name", "N/A")

    # # Check if essential mongo_id_str was found (already checked in util function, but good practice)
    # if not mongo_id_str:
    #      print("Error: Closest match data dictionary is missing 'image_base_64_id'.")
    #      return

    # print(f"Closest match found: Distance={distance}, NIN={retrieved_nin}, MongoID={mongo_id_str}")

    # # if not mongo_id_str: # This check is now redundant due to the extraction above
    # #     print("Error: Retrieved Milvus entity does not contain 'image_base_64_id'.")
    # #     return

    # # 5. Retrieve Image from MongoDB
    # print(f"Retrieving image data from MongoDB using ID: {mongo_id_str}...")
    # mongo_client = None
    # mongo_doc = None
    # try:
    #     mongo_client = initialise_mongo_cloud_db_client()
    #     mongo_doc = get_document_dict_for_mongo_db_collection_by_object_id(
    #         mongo_client, MONGO_DB_NAME, MONGO_COLLECTION_NAME, mongo_id_str
    #     )
    # except Exception as e:
    #     print(f"An error occurred during MongoDB interaction: {e}")
    #     # Fall through to closing client
    # finally:
    #     if mongo_client:
    #         mongo_client.close()
    #         # print("MongoDB connection closed.")

    # if not mongo_doc:
    #     print(f"Error: Could not find document in MongoDB with ID: {mongo_id_str}")
    #     return

    retrieved_base64_string = mongo_doc.get("base64_string")
    if not retrieved_base64_string:
        print(f"Error: MongoDB document (ID: {mongo_id_str}) does not contain 'base64_string' field.")
        return
    print("Successfully retrieved Base64 data from MongoDB.")

    # 6. Prepare Images for Display
    print("Loading images for display...")
    try:
        input_img = Image.open(image_path)
        input_img.load() # Load image data
        # Convert input to RGB for consistency
        if input_img.mode == 'RGBA':
             input_img = input_img.convert('RGB')
    except Exception as e:
        print(f"Error loading input image file {image_path}: {e}")
        return

    retrieved_img = decode_base64_to_image(retrieved_base64_string)
    if not retrieved_img:
        print("Failed to decode retrieved Base64 string into an image.")
        return

    # 7. Display Images
    print("Displaying images...")
    fig, axes = plt.subplots(1, 2, figsize=(10, 5)) # Adjust figure size as needed

    axes[0].imshow(input_img)
    axes[0].set_title("Input Image")
    axes[0].axis('off')

    axes[1].imshow(retrieved_img)
    axes[1].set_title(f"DB Match (NIN: {retrieved_nin})\nDistance: {distance:.4f}")
    axes[1].axis('off')

    fig.tight_layout()
    plt.show()

    print("Verification process complete.")


# --- Script Entry Point ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search Milvus for a face key and display the input vs matched image.")
    parser.add_argument("image_path", help="Path to the input image file.")
    args = parser.parse_args()

    search_and_verify(args.image_path) 