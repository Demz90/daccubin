"""    
Setting Up Milvus
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import random
import time
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
    AnnSearchRequest, RRFRanker, WeightedRanker # For potential future hybrid search/ranking
)

def connect_to_milvus():
    """Establishes connection to Zilliz Cloud."""
    # Get credentials from environment variables
    zilliz_cloud_uri = os.getenv("zilliz_cloud_uri")
    zilliz_cloud_token = os.getenv("zilliz_cloud_token") # API Key / Token

    if not zilliz_cloud_uri or not zilliz_cloud_token:
        raise ValueError("Please set zilliz_cloud_uri and zilliz_cloud_token environment variables.")
    
    print(f"Connecting to Milvus at {zilliz_cloud_uri}...")
    try:
        # Ensure the URI starts with a valid scheme
        if not zilliz_cloud_uri.startswith(("http://", "https://", "tcp://", "unix://")):
            raise ValueError(f"Invalid URI scheme: {zilliz_cloud_uri}. Must start with http, https, tcp, or unix.")
        
        connections.connect("default",
                            uri=zilliz_cloud_uri,
                            token=zilliz_cloud_token,
                            # Use secure=True if your URI indicates TLS/SSL (usually does for cloud)
                            secure=True)
        print("Successfully connected to Milvus.")
    except Exception as e:
        print(f"Failed to connect to Milvus: {e}")
        raise

def create_face_public_keys_collection():
    """Creates the face_public_keys collection with predefined parameters if it doesn't exist."""
    collection_name = "face_public_keys"
    dimension = 256
    index_params = {
        "metric_type": "HAMMING",
        "index_type": "BIN_IVF_FLAT",
        "params": {"nlist": 16}
    }

    if utility.has_collection(collection_name):
        print(f"Collection '{collection_name}' already exists.")
        return Collection(collection_name)

    print(f"Creating collection '{collection_name}'...")
    # Define fields: Primary key (auto-generated), binary vector, and detailed metadata
    field_id = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True)
    field_vector = FieldSchema(name="face_public_key_vector", dtype=DataType.BINARY_VECTOR, dim=dimension)
    field_face_public_key = FieldSchema(name="face_public_key", dtype=DataType.VARCHAR, max_length=512)
    field_image_base64_id = FieldSchema(name="image_base_64_id", dtype=DataType.VARCHAR, max_length=100)
    field_first_name = FieldSchema(name="first_name", dtype=DataType.VARCHAR, max_length=100)
    field_middle_name = FieldSchema(name="middle_name", dtype=DataType.VARCHAR, max_length=100)
    field_surname = FieldSchema(name="surname", dtype=DataType.VARCHAR, max_length=100)
    field_date_of_birth = FieldSchema(name="date_of_birth", dtype=DataType.VARCHAR, max_length=9)
    field_issue_date = FieldSchema(name="issue_date", dtype=DataType.VARCHAR, max_length=9)
    field_nationality = FieldSchema(name="nationality", dtype=DataType.VARCHAR, max_length=3)
    field_sex = FieldSchema(name="sex", dtype=DataType.VARCHAR, max_length=1)
    field_height = FieldSchema(name="height", dtype=DataType.VARCHAR, max_length=5)
    field_nin = FieldSchema(name="NIN", dtype=DataType.VARCHAR, max_length=19)

    # Define collection schema
    schema = CollectionSchema(
        fields=[
            field_id, field_vector, field_face_public_key, field_image_base64_id, field_first_name, field_middle_name,
            field_surname, field_date_of_birth, field_issue_date, field_nationality,
            field_sex, field_height, field_nin
        ],
        description="Collection for storing face public keys and detailed metadata",
        enable_dynamic_field=False # Explicit schema
    )

    # Create collection
    collection = Collection(name=collection_name, schema=schema)
    print(f"Collection '{collection_name}' created.")

    # Create index on the vector field for efficient search
    print(f"Creating index {index_params['index_type']} for field 'face_public_key_vector'...")
    collection.create_index(field_name="face_public_key_vector", index_params=index_params)
    utility.wait_for_index_building_complete(collection_name)
    print("Index created successfully.")

    return collection

def insert_data_into_face_public_keys(collection: Collection, nin_input_dict: dict):
    """Inserts data into the face_public_keys collection and returns the ID of the inserted data. Raises an error if compulsory fields are missing."""
    # Check for compulsory fields
    if 'face_public_key_vector' not in nin_input_dict or 'image_base_64_id' not in nin_input_dict:
        raise ValueError("Both 'face_public_key_vector' and 'image_base_64_id' are required fields.")

    # Prepare data with placeholders for missing fields
    data = {
        "face_public_key": nin_input_dict.get("face_public_key", ""),  # Add face_public_key
        "face_public_key_vector": nin_input_dict.get("face_public_key_vector"),
        "image_base_64_id": str(nin_input_dict.get("image_base_64_id")),
        "first_name": nin_input_dict.get("first_name", "Ahmed"),
        "middle_name": nin_input_dict.get("middle_name", "Amadi"),
        "surname": nin_input_dict.get("surname", "Ayoola"),
        "date_of_birth": nin_input_dict.get("date_of_birth", "03 May 67"),
        "issue_date": nin_input_dict.get("issue_date", "21 May 14"),
        "nationality": nin_input_dict.get("nationality", "NGA"),
        "sex": nin_input_dict.get("sex", "M"),
        "height": nin_input_dict.get("height", "180"),
        "NIN": nin_input_dict.get("NIN", "5412753456")
    }

    # Insert data into the collection and retrieve the result
    result = collection.insert([data])
    print("Data inserted successfully.")
    print(result)

    # Return the ID of the inserted data
    return result.primary_keys[0] if result.primary_keys else None

def clear_all_collections():
    """Deletes all collections in the Milvus database."""
    try:
        collections = utility.list_collections()
        for collection_name in collections:
            print(f"Deleting collection '{collection_name}'...")
            utility.drop_collection(collection_name)
        print("All collections have been deleted successfully.")
    except Exception as e:
        print(f"Failed to clear collections: {e}")
        raise


def get_collection(collection_name):
    """Retrieves a collection object based on the collection name after connecting to the Milvus DB."""
    if not utility.has_collection(collection_name):
        raise ValueError(f"Collection '{collection_name}' does not exist.")
    return Collection(collection_name)




"""
--- Data Generation and Insertion ---
"""
import time
def generate_dummy_key_hex():
    """Generates a random 64-character hex string (256 bits)."""
    dimension=256
    num_bytes = dimension // 8
    random_bytes = os.urandom(num_bytes)
    return random_bytes.hex()

def insert_dummy_data(collection: Collection, num_records: int):
    """Inserts dummy data into the collection."""
    if collection.num_entities > 0:
        print(f"Collection already contains {collection.num_entities} entities. Skipping dummy data insertion.")
        return

    print(f"Generating and inserting {num_records} dummy records...")
    keys_binary = []
    metadata_ids = []
    for i in range(num_records):
        hex_key = generate_dummy_key_hex()
        # Convert hex key to bytes for BINARY_VECTOR storage
        binary_key = bytes.fromhex(hex_key)
        keys_binary.append(binary_key)
        metadata_ids.append(f"user_meta_{i:03d}") # e.g., user_meta_000

    # Prepare data in the format expected by Milvus SDK (list of lists/tuples)
    entities = [
        keys_binary,
        metadata_ids
    ]

    # Insert data
    insert_result = collection.insert(entities)
    print(f"Inserted {len(insert_result.primary_keys)} records.")

    # Important: Flush data to make it searchable
    print("Flushing collection...")
    start_flush = time.time()
    collection.flush()
    end_flush = time.time()
    print(f"Flush completed in {end_flush - start_flush:.4f} seconds.")

def find_closest_key(collection: Collection, query_key_hex: str):
    """
    Searches for the closest matching key (smallest Hamming distance).

    Args:
        collection: The loaded Milvus collection object.
        query_key_hex: The 64-character hex public key to search for.

    Returns:
        A tuple (closest_metadata_id, distance) or (None, None) if no match found.
    """
    if len(query_key_hex) != 64:
        print("Error: Query key must be a 64-character hex string.")
        return None, None

    try:
        query_vector_binary = bytes.fromhex(query_key_hex)
    except ValueError:
        print("Error: Invalid hex character in query key.")
        return None, None

    print(f"\nSearching for key closest to: {query_key_hex[:10]}...")

    search_vectors = [query_vector_binary]  # Search expects a list of vectors

    start_search = time.time()
    # Perform the search
    results = collection.search(
        data=search_vectors,
        anns_field="face_public_key_vector",  # The field to search on
        param={"metric_type": "HAMMING", "params": {"nprobe": 16}},  # Example search parameters
        limit=1,  # We only want the single closest match
        output_fields=["user_metadata_id"]  # Specify which fields to return from the match
    )
    end_search = time.time()
    print(f"Search completed in {end_search - start_search:.4f} seconds.")

    # Process results
    # Results is a list (one element per query vector). Each element contains hits.
    if results and results[0]:
        closest_hit = results[0][0]  # Get the first hit for the first query vector
        distance = closest_hit.distance  # Hamming distance
        metadata_id = closest_hit.entity.get("user_metadata_id")
        print(f"Closest match found: Metadata ID='{metadata_id}', Hamming Distance={distance}")
        return metadata_id, distance
    else:
        print("No similar key found in the collection.")
        return None, None

def find_closest_face_key(collection: Collection, query_key_hex: str, top_k: int = 1):
    """
    Searches the 'face_public_keys' collection for the closest matching key(s).

    Args:
        collection: The loaded 'face_public_keys' Milvus collection object.
        query_key_hex: The hex public key (face vector) to search for.
        top_k: The number of closest matches to return.

    Returns:
        A list of tuples, where each tuple contains (mongo_id_str, nin, distance)
        for a hit, sorted by distance. Returns an empty list if no match found or error.
    """
    # Define the list of all metadata fields we want to retrieve
    output_fields_list = [
        "image_base_64_id", "first_name", "middle_name", "surname",
        "date_of_birth", "issue_date", "nationality", "sex", "height", "NIN"
    ]

    # Validate input hex key length (assuming 256-bit binary vectors -> 64 hex chars)
    if len(query_key_hex) != (collection.schema.fields[1].params.get('dim', 0) // 4):
        print(f"Error: Query key hex length {len(query_key_hex)} does not match expected dimension {collection.schema.fields[1].params.get('dim', 0)}.")
        return []

    try:
        query_vector_binary = bytes.fromhex(query_key_hex)
    except ValueError:
        print("Error: Invalid hex character in query key.")
        return []

    print(f"\nSearching '{collection.name}' for key closest to: {query_key_hex[:10]}... (top_k={top_k})")

    search_vectors = [query_vector_binary]
    search_params = {
        "metric_type": "HAMMING",
        "params": {"nprobe": 16}
    }

    start_search = time.time()
    try:
        results = collection.search(
            data=search_vectors,
            anns_field="face_public_key_vector",
            param=search_params,
            limit=top_k,
            output_fields=output_fields_list
        )
    except Exception as e:
        print(f"Error during Milvus search: {e}")
        if "collection not loaded" in str(e):
            print("Attempting to load the collection and retrying...")
            try:
                collection.load()
                results = collection.search(
                    data=search_vectors,
                    anns_field="face_public_key_vector",
                    param=search_params,
                    limit=top_k,
                    output_fields=output_fields_list
                )
            except Exception as retry_exception:
                print(f"Retry failed: {retry_exception}")
                return []
        else:
            return []
    end_search = time.time()
    print(f"Search completed in {end_search - start_search:.4f} seconds.")

    # Process results
    hit_results = []
    if results and results[0]:
        print(f"Found {len(results[0])} potential match(es). Processing...")
        for hit in results[0]:
            distance = hit.distance
            entity_data = {field: getattr(hit.entity, field, None) for field in output_fields_list}
            entity_data['distance'] = distance
            if entity_data["image_base_64_id"]:
                hit_results.append(entity_data)
            else:
                print(f"Warning: Found hit with distance {distance} but missing 'image_base_64_id'. Skipping.")
        hit_results.sort(key=lambda x: x['distance'])
    else:
        print("No similar key found in the collection.")

    return hit_results

