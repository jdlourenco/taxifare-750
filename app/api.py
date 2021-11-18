from fastapi import FastAPI
import pandas as pd
import joblib

app = FastAPI()

model = joblib.load("linear_model.joblib")

@app.get("/")
def index():
    return {"ok": True}

@app.get("/predict")
def predict(pickup_datetime, lon1, lat1, lon2, lat2, passcount):

    X = pd.DataFrame({
        "pickup_datetime": [pickup_datetime],
        "pickup_longitude": [float(lon1)],
        "pickup_latitude": [float(lat1)],
        "dropoff_longitude": [float(lon2)],
        "dropoff_latitude": [float(lat2)],
        "passenger_count": [int(passcount)],
    })
    
    print(f"X={X}")
    
    prediction = model.predict(X)
    
    print(f"prediction={prediction}")
    
    return {"pred": prediction[0]}
