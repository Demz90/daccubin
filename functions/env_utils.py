## Add the root path so modules can be easily imported
import os
import sys
temp = os.path.dirname(os.path.abspath(__file__))
print(temp)
BASE_DIR = os.path.abspath(os.path.join(temp, '..'))
BASE_DIR = BASE_DIR + os.sep
print(BASE_DIR)
sys.path.insert(0, BASE_DIR)

from dotenv import load_dotenv, dotenv_values, set_key

def load_in_env_vars():
    # Load environment variables from a specific .env file
    env_var_path = os.path.join(BASE_DIR, '.env')

    # - load the .env file so we can get relevant aws keys to download the secret
    load_dotenv(env_var_path)
