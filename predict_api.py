import numpy as np
import hydra
from flask import Flask, request, jsonify

from models.siamese_model import SiameseModel
from common import PROJECT_ROOT


app = Flask(__name__)
hydra.initialize(config_path='config', job_name='test_api')
model = SiameseModel.load_from_checkpoint('logs/Speech_And_Speaker_Detection/893660w5/checkpoints/epoch=0-step=1040.ckpt')


@app.route('/')
def home():
    return 'Works'


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    query, support = data['query'], data['support']
    query = [(np.array(a, dtype=np.float), sr) for a, sr in query]
    support = (np.array(support[0], dtype=np.float), support[1])
    predictions = model.predict(query, support)
    return jsonify(predictions)


if __name__ == '__main__':
    app.run(debug=False)
