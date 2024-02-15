from flask import Flask, request, jsonify
import torch
import joblib
from recommander import ContNN, NCF
import predictor

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    prediction = predictor.predict_api(data)
    return jsonify({'prediction': prediction})


@app.route('/recommend', methods=['POST'])
def recomm_ncf():
    data = request.get_json()
    user_id = data['user_id']

    return jsonify({'recommendation': NCF.nfc_recommander(user_id)})

@app.route('/recommend_NN', methods=['POST'])
def recomm_nn():
    data = request.get_json()
    user_features = data
    top_n = 10

    return jsonify({'recommendation': ContNN.recommend_top_n_items(user_features, top_n)})

if __name__ == '__main__':
    app.run(port=5000, debug=True)