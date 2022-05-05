import numpy as np
import hydra
from flask import Flask, request, jsonify
from torch import dropout

from models.classifier_model import ClassifierModel
from common import PROJECT_ROOT
from models.siamese_model import SiameseModel


app = Flask(__name__)
pre_trained = SiameseModel.load_from_checkpoint('/Users/konradkaranowski/PythonProjects/Speech-And-Speaker-Detection/artifacts/model-30249sth:v0/model.ckpt')
model = ClassifierModel.load_from_checkpoint('model', model=pre_trained)


@app.route('/')
def home() -> str:
    return 'Works'


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    query = data['query']
    query = [(np.array(a, dtype=np.float), sr) for a, sr in query]
    predictions = model.predict(query)
    print(type(jsonify(predictions)))
    return jsonify(predictions)


if __name__ == '__main__':
    app.run(debug=False)
