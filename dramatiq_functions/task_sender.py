import json
import time
from typing import *
from bson import ObjectId
from functions import mongo_utils as mongo
from functions import rabbitmq_utils as rabbit_mq
from dramatiq_functions import dramatiq_utils as dram_utils


def generate_task_dict(task_service_name, task_name, task_queue_name, task_data):
    """
    Generic function to generate a task dictionary
    
    Parameters:
    - task_service_name: Name of the service this task belongs to
    - task_name: Name of the specific task
    - task_queue_name: Queue name for dramatiq
    - task_data: Dictionary containing the task-specific data
    
    Returns:
    dict: A dictionary representing the task
    """
    task_id = str(ObjectId())
    task_send_time = int(time.time())
    
    task_dict = {
        'task_id': task_id,
        'task_service_name': task_service_name,
        'task_name': task_name,
        'task_queue_name': task_queue_name,
        'task_status': 'sent',
        'task_message_dict': task_data,
        'task_send_time': task_send_time,
        'task_pickup_time': 0,
        'task_end_time': 0,
        'task_time_to_pickup': 0,
        'task_time_taken': 0,
        'task_message': 'NA',
        'local_machine_public_ip': 'NA',
        'worker_id': 'NA'
    }
    
    return task_dict

def send_task(task_dict, task_function, output=False):
    """
    Generic function to send a task to dramatiq
    
    Parameters:
    - task_dict: The task dictionary
    - task_function: The dramatiq actor function to call
    - output: if the task is to return an output
    
    Returns:
    str: The task_id of the sent task
    """
    task_log_service_mongo_db_name = 'utom_task_log_service'
    task_logs_collection_name = 'task_logs'
    
    # Initialize connections
    channel = rabbit_mq.initialize_rabbitmq_client_and_create_channel()
    mongo_client = mongo.initialise_mongo_cloud_db_client()
    
    # Convert task dict to JSON
    task_json = json.dumps(task_dict, cls=dram_utils.CustomEncoder)
    
    # Save to MongoDB
    dram_utils.save_task_log_to_mongo(
        task_dict, 
        mongo_client, 
        task_log_service_mongo_db_name, 
        task_logs_collection_name
    )
    
    # Send to dramatiq
    try:
        message = task_function.send(task_json)
    except:
        print('Error sending task, retrying with new connection')
        channel = rabbit_mq.initialize_rabbitmq_client_and_create_channel()
        message = task_function.send(task_json)
    
    print(f'Task has been sent to dramatiq with task_id: {task_dict["task_id"]}')
    if output:
        return message
    
    return task_dict["task_id"]

def send_process_nin_user_task(nin_input_dict: Dict):

    from dramatiq_functions.dramatiq_app import insert_nin_user_data
    
    task_data = {"nin_input_params": nin_input_dict}

    task_dict = generate_task_dict(
        "Nin Service",
        "process_nin_user_data",
        "insert_nin_user_data_task_queue",
        task_data
    )

    return send_task(task_dict, insert_nin_user_data)
