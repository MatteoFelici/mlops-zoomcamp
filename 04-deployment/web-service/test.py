# import predict
import requests

ride = {
    'PULocationID': '142',
    'DOLocationID': '209',
    'tpep_pickup_datetime': '2026-01-05 10:54:04',
    'trip_distance': 5.42
}

# features = predict.prepare_features(ride)
# preds = predict.predict(features)
# print(preds)

url = 'http://localhost:9696/predict'
response = requests.post(url, json=ride)
print(response.json())