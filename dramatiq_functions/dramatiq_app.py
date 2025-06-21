import dramatiq
import warnings
import time
warnings.filterwarnings("ignore")

import os
import sys
import time
import json
import atexit
from dramatiq.middleware import CurrentMessage
from dramatiq_functions import dramatiq_utils as dram_utils
from functions import mongo_utils as mongo
from dotenv import load_dotenv
load_dotenv(r"C:\Users\Samuel.Taiwo\Documents\samuCodes\daccubin\.env")

# Setup broker
broker = dram_utils.setup_dramatiq_broker()
dramatiq.set_broker(broker)
dramatiq.get_broker().add_middleware(CurrentMessage())

# Initialize connections
mongo_client = mongo.initialise_mongo_cloud_db_client()
print('Initialized mongo connection')

# Function to close connections at shutdown
def close_connections():
    if mongo_client:
        mongo_client.close()
        print("Mongo connection closed")

atexit.register(close_connections)

# Common task processing function
def process_task(task_function, data, task_message_dict):
    worker_id = '111'  # In production, this should be unique per worker
    local_machine_public_ip = '127.0.0.1'  # In production, get actual IP
    
    task_id_str = task_message_dict['task_id']
    task_log_service_mongo_db_name = 'utom_task_log_service'
    task_logs_collection_name = 'task_logs'
    task_started_count_collection_name = 'task_started_count'
    
    # Use the helper function from dram_utils
    can_process_task = dram_utils.check_task_can_be_processed(
        task_id_str, 
        mongo_client,
        task_log_service_mongo_db_name, 
        task_logs_collection_name
    )
    
    if can_process_task:
        task_send_time = int(task_message_dict['task_send_time'])
        task_pickup_time = int(time.time())
        task_time_to_pickup = int(task_pickup_time - task_send_time)
        
        # Use the helper function from dram_utils
        dram_utils.update_task_to_started(
            task_id_str, 
            task_pickup_time, 
            task_time_to_pickup,
            local_machine_public_ip,
            worker_id,
            mongo_client,
            task_log_service_mongo_db_name,
            task_logs_collection_name,
            task_started_count_collection_name
        )
        
        try:
            print(f"Starting task execution for task ID: {task_id_str}")
            result = task_function(data)
            task_message = 'Task ran end to end successfully'
            task_status = 'completed'
            print(f"Task completed successfully for task ID: {task_id_str}")
        except Exception as e:
            import traceback
            task_message = f'There was an error: {str(e)}\nTraceback: {traceback.format_exc()}'
            print(task_message)
            task_status = 'failed'
            result = None
        
        task_end_time = int(time.time())
        task_time_taken = int(task_end_time - task_send_time)
        task_process_time = int(task_time_taken - task_time_to_pickup) if task_time_taken >= task_time_to_pickup else 0
        
        # Use the helper function from dram_utils
        dram_utils.update_task_completion(
            task_id_str,
            task_status,
            task_pickup_time,
            task_time_to_pickup,
            task_end_time,
            task_time_taken,
            task_process_time,
            task_message,
            local_machine_public_ip,
            worker_id,
            mongo_client,
            task_log_service_mongo_db_name,
            task_logs_collection_name
        )
        
        print(f"Task ID: {task_id_str}, Worker ID: {worker_id} processed. Status: {task_status}")
        return result
    else:
        print(f"Task ID: {task_id_str}, Worker ID: {worker_id} skipped (duplicate).")
        return None

@dramatiq.actor(queue_name="insert_nin_user_data_task_queue", max_retries=1, time_limit=900000)
def insert_nin_user_data(data):
    from dramatiq_functions import dramatiq_app_tasks
    
    task_data = json.loads(data)
    task_message_dict = task_data
    
    return process_task(
        dramatiq_app_tasks.process_nin_user_data,
        data,
        task_message_dict
    )
