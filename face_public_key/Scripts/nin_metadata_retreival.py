import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import public_key_gen
import milvus_utils 
import image_utils
import nin_metadata_insertion

def find_closest_face_key_and_retrieve_data(image_path):
    # --- Constants ---
    mongo_db_name = "theaccubin_images"
    mongo_collection_name = "image_storage"
    milvus_collection_name = "face_public_keys"
    face_public_keys_collection = milvus_utils.get_collection(milvus_collection_name)
    
    # Generate base64 string and face public key from the image
    input_image_base_64_string = image_utils.generate_base_64_from_image_path(image_path)
    input_image_face_public_key = public_key_gen.generate_face_public_key_from_image(image_path)
    
    # Find the closest match in Milvus
    closest_match = milvus_utils.find_closest_face_key(face_public_keys_collection, input_image_face_public_key)[0]
    
    # Extract necessary info using .get() for safety
    mongo_id_str = closest_match.get("image_base_64_id")
    retrieved_nin = closest_match.get("NIN", "N/A")
    distance = closest_match.get("distance", float('inf'))  # Get distance, default to infinity if missing
    
    print(f"Closest match found: Distance={distance}, NIN={retrieved_nin}, MongoID={mongo_id_str}")
    
    # Initialize the return dictionary
    retrieved_nin_dict = {
        "NIN": retrieved_nin,
        "distance": distance,
        "input_base_64_string": input_image_base_64_string,
        "retrieved_base64_string": None
    }
    
    # If the distance is greater than 50, return a message indicating no user was found
    if distance > 50:
        return {'message': 'No user found'}
    
    # Retrieve Image from MongoDB
    print(f"Retrieving image data from MongoDB using NIN: {retrieved_nin}...")
    mongo_client = None
    mongo_doc = None
    try:
        mongo_client = nin_metadata_insertion.initialise_mongo_cloud_db_client()
        filter_criteria = {"associated_nin": retrieved_nin}
        mongo_doc = nin_metadata_insertion.get_documents_by_filter_criteria(
            mongo_client, mongo_db_name, mongo_collection_name, filter_criteria
        )[0]  # Assuming the NIN is unique and returns a single document
    except Exception as e:
        print(f"An error occurred during MongoDB interaction: {e}")
    finally:
        if mongo_client:
            mongo_client.close()
    
    # Update the retrieved base64 string in the return dictionary
    if mongo_doc:
        retrieved_nin_dict["retrieved_base64_string"] = mongo_doc['base64_string']
    
    return retrieved_nin_dict