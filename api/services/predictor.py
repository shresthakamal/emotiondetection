import flask
from flask import request, render_template
from emotiondetection.main import test_pipeline
from api.config import config

from bson.json_util import dumps
from bson.objectid import ObjectId
from flask import jsonify, request


def home():
    return render_template("index.html")


def sentence_prediction(sentence):
    output = test_pipeline(config.MODEL[1], input_text=sentence)
    return output


def predict():
    payload = request.form["review"]
    print("[INFO: (api/service/predictor)]: User feedback received")
    prediction = sentence_prediction(payload)
    return render_template("predict.html", input_text=payload, output=prediction)
