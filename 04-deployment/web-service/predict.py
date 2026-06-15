import pickle
from datetime import datetime
from flask import Flask, request, jsonify


app = Flask('duration-prediction')
    

def prepare_features(ride):
    features = {}
    features['locationID'] = f'{ride['PULocationID']}_{ride['DOLocationID']}' 
    tmp_dt = datetime.strptime(ride['tpep_pickup_datetime'], '%Y-%m-%d %H:%M:%S')
    features['pickup_hour'] = tmp_dt.hour
    features['dow'] = tmp_dt.strftime('%A')
    features['trip_distance'] = ride['trip_distance']
    return features


def predict(features):
    with open('lin_reg.bin', 'rb') as f:
        (dv, model) = pickle.load(f)
    X = dv.transform(features)
    preds = model.predict(X)
    return preds[0]


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    ride = request.get_json()
    features = prepare_features(ride)
    pred = predict(features)
    
    result = {'duration': pred}
    
    return jsonify(result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=9696)