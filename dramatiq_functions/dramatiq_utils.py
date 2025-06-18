import warnings
warnings.filterwarnings("ignore")
import os
import dramatiq
from dramatiq.brokers.rabbitmq import RabbitmqBroker
from bson import ObjectId
from datetime import datetime
import json

# MongoDB task logging functions
def save_task_log_to_mongo(task_json, mongo_client, db_name, collection_name):
    try:
        db = mongo_client[db_name]
        collection = db[collection_name]
        insert_result = collection.insert_one(task_json)
    except Exception as e:
        print('The task ID seems to be duplicate so we are trying it again after creating a new unique ID')
        try:
            new_id = ObjectId()
            task_json['_id'] = new_id
            task_json['task_id'] = str(ObjectId())
            db = mongo_client[db_name]
            collection = db[collection_name]
            insert_result = collection.insert_one(task_json)
            print('successfully did the insert with new task ID')
        except Exception as e:
            print(f"An error occurred: {str(e)}")

def update_task_started_count(mongo_client, db_name, collection_name, task_id):
    database = mongo_client[db_name]
    collection = database[collection_name]
    existing_record = collection.find_one({'task_id': task_id})

    if existing_record:
        new_count = existing_record['count'] + 1
        collection.update_one({'task_id': task_id}, {'$set': {'count': new_count}})
    else:
        collection.insert_one({'task_id': task_id, 'count': 1})

def get_tasks_by_filter_criteria(filter_criteria):
    from functions import mongo_utils as mongo
    mongo_client = mongo.initialise_mongo_cloud_db_client()

    task_log_service_mongo_db_name = 'utom_task_log_service'
    task_logs_collection_name = 'task_logs'

    results = mongo.get_documents_by_filter_criteria(
        mongo_client, 
        task_log_service_mongo_db_name, 
        task_logs_collection_name, 
        filter_criteria
    )
    
    return results

def update_task_status(mongo_client, db_name, collection_name, task_id, updated_data):
    try:
        db = mongo_client[db_name]
        collection = db[collection_name]
        collection.update_one({'task_id': task_id}, {'$set': updated_data})
        return True
    except Exception as e:
        print(f"Error updating task status: {str(e)}")
        return False

def setup_dramatiq_broker():

    rabbitmq_server_ip_address  =  os.environ.get("utom_rabbitmq_server_ip_address")
    rabbitmq_server_username    =  os.environ.get("utom_rabbitmq_server_username")
    rabbitmq_server_password    =  os.environ.get("utom_rabbitmq_server_password")

    broker = RabbitmqBroker(
        url=f"amqp://{rabbitmq_server_username}:{rabbitmq_server_password}@{rabbitmq_server_ip_address}:5672"
    )
    return broker

class CustomEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (ObjectId, datetime)):
            return str(o)
        return super().default(o)

# Helper functions for task processing moved from dramatiq_app.py
def check_task_can_be_processed(task_id, mongo_client, db_name, collection_name):
    from functions import mongo_utils as mongo
    try:
        filter_criteria = {'task_id': task_id}
        # Ensure we use the get_documents_by_filter_criteria from mongo_utils
        tasks = mongo.get_documents_by_filter_criteria(mongo_client, db_name, collection_name, filter_criteria)
        
        if tasks and len(tasks) > 0:
            db_task_status = tasks[0]['task_status']
            # Task can be processed only if its status is 'sent'
            return db_task_status == 'sent'
        # If no task found, assume it can be processed (might be the first time)
        return True 
    except Exception as e:
        print(f"Warning: Could not check task status in check_task_can_be_processed: {str(e)}")
        # Default to allowing processing if there's an error checking status
        return True

def update_task_to_started(task_id, pickup_time, time_to_pickup, ip, worker_id, 
                          mongo_client, db_name, collection_name, count_collection_name):
    try:
        updated_data = {
            'task_status': 'started',
            'task_pickup_time': pickup_time,
            'task_time_to_pickup': time_to_pickup,
            'task_pickup_local_machine_public_ip': ip,
            'task_pickup_worker_id': worker_id,
        }
        
        # Use the existing update_task_status from this utils file
        update_task_status(mongo_client, db_name, collection_name, task_id, updated_data)
        # Use the existing update_task_started_count from this utils file
        update_task_started_count(mongo_client, db_name, count_collection_name, task_id)
    except Exception as e:
        print(f"Warning: Could not update task to started status: {str(e)}")

def update_task_completion(task_id, status, pickup_time, time_to_pickup, end_time, 
                          time_taken, process_time, message, ip, worker_id, 
                          mongo_client, db_name, collection_name):
    from functions import mongo_utils as mongo
    try:
        updated_data = {
            'task_status': status,
            'task_pickup_time': pickup_time,
            'task_time_to_pickup': time_to_pickup,
            'task_end_time': end_time,
            'task_time_taken': time_taken,
            'task_process_time': process_time,
            'task_message': message,
            'local_machine_public_ip': ip, # Ensure this key matches the one used in update_task_to_started
            'worker_id': worker_id,        # Ensure this key matches the one used in update_task_to_started
        }
        
        # Reinitialize mongo client to potentially avoid connection timeout issues after long tasks
        fresh_mongo_client = mongo.initialise_mongo_cloud_db_client()
        # Use the existing update_task_status from this utils file
        update_task_status(fresh_mongo_client, db_name, collection_name, task_id, updated_data)
    except Exception as e:
        print(f"Warning: Could not update task completion status: {str(e)}")
        # Attempt retry with another fresh connection
        try:
            fresh_mongo_client = mongo.initialise_mongo_cloud_db_client()
            update_task_status(fresh_mongo_client, db_name, collection_name, task_id, updated_data)
        except Exception as e_retry:
            print(f"Warning: Could not update task completion status after retry: {str(e_retry)}")
