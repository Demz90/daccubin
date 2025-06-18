
import warnings
warnings.filterwarnings("ignore")

import json
import dramatiq
from functions.nin_metadata_insertion import generate_and_insert_nin_input


def process_nin_user_data(task_json_string):
    """
    Generate documentations for modules and repos

    Args:
        task_json_string (str): JSON string containing the task details
    """

    task_dict = json.loads(task_json_string)
    task_message_dict = task_dict['task_message_dict']

    nin_input_params = task_message_dict.get("nin_input_params")
    input_dict = generate_and_insert_nin_input(nin_input_params)
    return input_dict
