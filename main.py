import pandas as pd
from faker import Faker
import random
from datetime import timedelta
from functions.nin_metadata_insertion import generate_and_insert_nin_input
from functions.nin_metadata_retreival import find_closest_face_key_and_retrieve_data

sample_imgs_df = pd.read_csv("Sample images.csv")
images = sample_imgs_df["Photo"].to_list()[233:] # start here ::FQ504935294566KL
ids =  sample_imgs_df["VNIN"].to_list()[233:]

def generate_fake_identity():
    fake = Faker()

    first_name = fake.first_name()
    middle_name = fake.first_name()
    surname = fake.last_name()
    
    # Generate date of birth: between 18 and 80 years old
    date_of_birth = fake.date_of_birth(minimum_age=18, maximum_age=80)

    # Issue date: between date of birth + 18 years and today
    min_issue_date = date_of_birth + timedelta(days=18*365)
    issue_date = fake.date_between(start_date=min_issue_date, end_date='today')

    nationality = fake.country()
    sex = random.choice(['M', 'F'])  # Could be extended with 'X' or 'Other'

    # Height in cm: between 150 and 200 for realism
    height = f"{random.randint(150, 200)}"

    return {
        "first_name": first_name,
        "middle_name": middle_name,
        "surname": surname,
        "date_of_birth": date_of_birth.isoformat(),
        "issue_date": issue_date.isoformat(),
        "nationality": nationality[:3],
        "sex": sex,
        "height": height
    }

count = 0
for id, img in zip(ids, images):
    fake_iden = generate_fake_identity()
    fake_iden.update({
        "image_path": img,
        "nin": id
    })

    # generate_and_insert_nin_input(fake_iden)

    retrieved_metadata = find_closest_face_key_and_retrieve_data(img)
    {retrieved_metadata["NIN"], retrieved_metadata["distance"]}
    if retrieved_metadata["NIN"] != id:
        with open("false.txt", "w") as f:
            f.write(f"Wrong Match - {id}\n")

    if count > 500:
        break
    count +=1
