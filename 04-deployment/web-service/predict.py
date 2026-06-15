import pickle
from datetime import datetime


with open('lin_reg.bin', 'rb') as f:
    (dv, model) = pickle.load(f)
    
def prepare_features(ride):
    features = {}
    features['locationID'] = f'{ride['PULocationID']}_{ride['DOLocationID']}' 
    tmp_dt = datetime.strptime(ride['tpep_pickup_datetime'], '%Y-%m-%d %H:%M:%S')
    features['pickup_hour'] = tmp_dt.hour
    features['dow'] = tmp_dt.strftime('%A')
    features['trip_distance'] = ride['trip_distance']
    return features

def predict(features):
    X = dv.transform(features)
    preds = model.predict(X)
    return preds