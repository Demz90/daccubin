## Add the root path so modules can be easily imported
import os
import sys
temp = os.path.dirname(os.path.abspath(__file__))
print(temp)
BASE_DIR = os.path.abspath(os.path.join(temp, '..'))
BASE_DIR = BASE_DIR + os.sep
print(BASE_DIR)
sys.path.insert(0, BASE_DIR)

from functions import milvus_utils

##  - Sort out the setting up of the vector DB
milvus_utils.connect_to_milvus()
milvus_utils.clear_all_collections()
milvus_utils.create_face_public_keys_collection()