# Speech-And-Speaker-Detection
This repository is a part of the final project of Artifical Intelligence course led by bsc. Julita Bielaniewicz. Project's goal was to create the application for voice unlocking. System should detect not only the spoken word (password), but also the speaker (system's owner). The assumption of the project was to create a network from scratch, which is why I didn't use conformer and other more complicated and more computationally expensive methods.
This repository consists of code for training machine learning model, and also has a simply flask's API for predicting using it. Another part, application itself is located [here](https://github.com/konrad-karanowski/Speaker-Lock). 

# About the model
Model's training is divided into two parts: pre-training and fine-tuning.

# Audio-features
During the experiments, two types of  signal preprocessing methods were used: Mel-Spectrograms and Mel-Frequency Cepstra Coefficients. First method was performing better during the real-world scenario and is recommended for this task.

## Pre-training stage
During pre-training stage, model was trained in Siamese-Network setup with two separated heads in order to obtain acoustic-word-embeddings and acoustic-speaker-embeddings. TripletMarginLoss was used as a cost function. During this stage, model was trained on [this](https://www.kaggle.com/datasets/bharatsahu/speech-commands-classification-dataset) data.

## Fine-tuning stage
During fine-tuning stage, model was trained as a simple classifier using acoustic-word-embeddings and acoustic-speaker-embeddings from previous stage. The whole model was fine-tuned on new dataset, collected by myself with similar structure to the previous one. 

# How to use
For training the model just run:
```
python train.py
```
This project uses the Hydra framework, so you can put optional parameters. Whole configuration setup is stored in a config directory and its subdirectories.

For hosting proper model run:
```
python predict_api.py
```

# Acknowledgements
* Backbone is a modified version of a backbone from paper [Unsupervised Training of Siamese Networks for Speaker Verification](https://upcommons.upc.edu/bitstream/handle/2117/332092/1882.pdf;jsessionid=999E22DB13AA0EE105470549C219468C?sequence=1)
* Hydra setup derived from: [https://github.com/grok-ai/nn-template](https://github.com/grok-ai/nn-template)
