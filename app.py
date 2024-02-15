from flask import Flask, request, jsonify
import torch
import joblib
from recommander import ContNN, NCF
import predictor

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    data = replace_underscores_in_json(data)
    prediction = predictor.predict_api(data).tolist()
    return jsonify({'prediction': prediction})


@app.route('/recommend', methods=['POST'])
def recomm_ncf():
    data = request.get_json()
    user_id = data['user_id']

    return jsonify({'recommendation': NCF.nfc_recommander(user_id).tolist()})

@app.route('/recommend_NN', methods=['POST'])
def recomm_nn():
    user_features = request.get_json()
    top_n = 10
    
    user_features = replace_underscores_in_json(user_features)
    return jsonify({'recommendation': ContNN.recommend_top_n_items(user_features, top_n).tolist()})

def replace_underscores_in_json(obj):
    if isinstance(obj, dict):
        return {k: replace_underscores_in_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [replace_underscores_in_json(element) for element in obj]
    elif isinstance(obj, str):
        return obj.replace('_', ' ')
    else:
        return obj
    
if __name__ == '__main__':
    app.run(port=5000, debug=True)