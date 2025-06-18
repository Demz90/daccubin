import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import matplotlib.pyplot as plt
import face_utils
import geometrical_features
import face_embeddings
import public_key_gen
import milvus_utils
import image_utils
from pymongo import MongoClient, ASCENDING, IndexModel
from bson.objectid import ObjectId
# Define MongoDB constants
MONGO_DB_NAME = "theaccubin_images"
MONGO_COLLECTION_NAME = "image_storage"

def drop_image_storage_collection():
    """
    Drops the image_storage collection from the database.
    
    Args:
        client (MongoClient): A MongoDB client instance.
    """
    client = initialise_mongo_cloud_db_client()
    db = client[MONGO_DB_NAME]
    db[MONGO_COLLECTION_NAME].drop()

def get_all_entries():
    """
    Returns all entries in the image_storage collection.
    
    Returns:
        list: A list of all documents in the collection.
    """
    client = initialise_mongo_cloud_db_client()
    db = client[MONGO_DB_NAME]
    collection = db[MONGO_COLLECTION_NAME]
    
    try:
        # Find all documents in the collection
        all_entries = list(collection.find({}))
        for ent in all_entries:
            print(ent.keys())
        return all_entries
    finally:
        client.close()

"""
Prelim
"""
# Connect to Milvus once at the start
# Consider if Mongo client should also be managed globally or per function call
milvus_utils.connect_to_milvus()

"""
Inserting NIN input data into Milvus
"""
def initialise_mongo_cloud_db_client():
    """
    Initializes a MongoDB client to connect to a MongoDB server in the cloud.

    This function reads environment variables to obtain the public IP address of the MongoDB server,
    the MongoDB server's username, and the MongoDB server's password. It then creates a connection
    string and establishes a connection to the MongoDB server using the PyMongo library.

    Returns:
        MongoClient: A connected MongoDB client.

    Raises:
        ValueError: If any of the required environment variables are not set.
    """

    # Read environment variables *** later change to your own
    mongo_server_public_ip_address = '37.27.215.123'
    mongodb_server_username = 'utom'
    mongodb_server_password = 'utom2025'

    # # Print the environment variables (optional)
    # print(f"MongoDB Server IP: {mongo_server_public_ip_address}")
    # print(f"MongoDB Username: {mongodb_server_username}")
    # print("MongoDB Password: [Hidden]")

    # Create the MongoDB connection string
    CONNECTION_STRING = "mongodb://%s:%s@%s:27017/" % (mongodb_server_username, mongodb_server_password, mongo_server_public_ip_address)
    print (CONNECTION_STRING)
    # Establish a connection to the MongoDB server
    client = MongoClient(CONNECTION_STRING)

    return client

def input_data_into_mongo_db_collection(client, db_name, collection_name, input_data_dict):
    """
    Insert a single document into a MongoDB collection.

    Args:
        client (MongoClient): A MongoDB client instance.
        db_name (str): The name of the database.
        collection_name (str): The name of the collection.
        input_data_dict (dict): The document data to insert.
    """
    # Connect to the specific collection
    db = client[db_name]
    collection = db[collection_name]

    # Add data to the collection
    insert_result = collection.insert_one(input_data_dict)
    return insert_result.inserted_id

def get_documents_by_filter_criteria(client, db_name, collection_name, filter_criteria):
    """
    Get documents from a MongoDB collection based on specified key-value pairs.

    Parameters:
    - collection: MongoDB collection object
    - key_value_pairs: Dictionary containing key-value pairs for filtering
    
    Example Filter Criteria:
    filter_criteria = {
        # "task_name": "process_dripshot",
        "task_name": "create_dripshot_video",
        "task_status": "completed"
    }
    Returns:
    - List of documents matching the specified key-value pairs
    """
    db = client[db_name]
    collection = db[collection_name]

    query = {key: value for key, value in filter_criteria.items()}
    result = collection.find(query)

    return list(result)

def generate_nin_input_dict(nin_input_params):
    
    milvus_collection_name = "face_public_keys"
    face_public_keys_collection = milvus_utils.get_collection(milvus_collection_name)
    
    image_path = nin_input_params['image_path']
    # current_nin = str(random.randint(1000000000, 9999999999)) # Generate a random NIN for Mongo doc
    current_nin = nin_input_params['nin']

    # 1. Generate Base64 string from image
    if "base64_string" not in nin_input_params:
        image_base_64_string = image_utils.generate_base_64_from_image_path(image_path)
    else:
        image_base_64_string = nin_input_params["base64_string"]

    # print(f"Generated Base64 string length: {len(image_base_64_string)}")

    # 2. Connect to MongoDB
    mongo_client = initialise_mongo_cloud_db_client()

    # 3. Prepare and Insert document into MongoDB
    mongo_doc = {
        "base64_string": image_base_64_string,
        "associated_nin": current_nin # Store NIN for easy reference
        # Consider adding a timestamp: "created_at": datetime.datetime.utcnow()
    }

    mongo_id = None
    try:
        mongo_id = input_data_into_mongo_db_collection(
            mongo_client, MONGO_DB_NAME, MONGO_COLLECTION_NAME, mongo_doc
        )
        print(f"Stored image in MongoDB ({MONGO_DB_NAME}/{MONGO_COLLECTION_NAME}) with ID: {mongo_id}")
    except Exception as e:
        print(f"Failed to insert image data into MongoDB: {e}")
        # Decide on error handling: re-raise, return None, etc.
        raise # Re-raise the exception to halt the process
    finally:
        # Ensure the client connection is closed
        if mongo_client:
            mongo_client.close()
            # print("MongoDB connection closed.")

    if not mongo_id:
         # This case should ideally be unreachable if the exception is raised
         raise ValueError("Failed to obtain MongoDB ID for the image.")

    # 4. Generate face public key for Milvus
    face_public_key = public_key_gen.generate_face_public_key_from_image(image_path)
    
    # Find the closest match in Milvus
    closest_match = milvus_utils.find_closest_face_key(face_public_keys_collection, face_public_key, top_k=3)
    
    # reduce the threshould `1` to find exact ffaces, `40` for very similar face
    duplicate_face = any(match["distance"] < 40 for match in closest_match) 

    # 5. Define the data dictionary for Milvus insertion
    nin_input_dict = {
        "face_public_key": face_public_key, # Include if needed elsewhere, not directly in Milvus schema
        "face_public_key_vector": bytes.fromhex(face_public_key),
        # "image_base_64_string": image_base_64_string, # Removed
        "image_base_64_id": str(mongo_id), # Use the MongoDB ObjectId string
        "first_name": nin_input_params.get("first_name", "John"),
        "middle_name": nin_input_params.get("middle_name", "Doe"),
        "surname": nin_input_params.get("surname", "Smith"),
        "date_of_birth": nin_input_params.get("date_of_birth", "01 Jan 90"),
        "issue_date": nin_input_params.get("issue_date", "01 Jan 20"),
        "nationality": nin_input_params.get("nationality", "USA"),
        "sex": nin_input_params.get("sex", "M"),
        "height": nin_input_params.get("height", "175"),
        "NIN": current_nin, # Use the same NIN as stored in Mongo doc reference
        "duplicate_face": duplicate_face
    }

    return nin_input_dict

def generate_and_insert_nin_input(nin_input_params):

    nin_input_dict = generate_nin_input_dict(nin_input_params)

    collection_name = "face_public_keys"
    face_public_keys_collection = milvus_utils.get_collection(collection_name)

    # Insert the data into the collection
    milvus_utils.insert_data_into_face_public_keys(face_public_keys_collection, nin_input_dict)

    return nin_input_dict