from fastapi import FastAPI
import pickle

app = FastAPI()

# load model
with open("../models/crop_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("../models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("../models/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)


@app.post("/recommend")
def recommend_crop(N: float, P: float, K: float, temp: float, humidity: float, ph: float, rainfall: float):

    data = [[N, P, K, temp, humidity, ph, rainfall]]

    scaled = scaler.transform(data)

    prediction = model.predict(scaled)

    crop = label_encoder.inverse_transform(prediction)

    return {"recommended_crop": crop[0]}