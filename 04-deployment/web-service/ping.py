from flask import Flask


app = Flask('ping')


@app.route('/ping', methods=['GET'])
def predict_endpoint():
    return 'PONG'


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=9696)