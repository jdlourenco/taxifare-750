from google.cloud import storage
from taxifare.data import BUCKET_NAME
import joblib
import pandas as pd

MODEL_PATH = "models/myfirstmodel/linear_model.joblib"

def get_model():
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(MODEL_PATH)
    blob.download_to_filename("local_model.joblib")
    
    model = joblib.load("local_model.joblib")
    return model
    
def get_test_set(test_path):
    df = pd.read_csv(test_path)
    return df

def generate_csv(test_set, predictions):
    test_set["fare_amount"] = predictions
    test_set = test_set[["key", "fare_amount"]]
    test_set.to_csv("kaggle_results.csv", index=False)
    

model    = get_model()
test_set = get_test_set("raw_data/test.csv")
predictions = model.predict(test_set)
generate_csv(test_set, predictions)
