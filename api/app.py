from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from api.urls import app_routes
from flask_pymongo import PyMongo


def init_app():
    app = Flask(__name__)

    # app.config("mongodb://db:27017/emotion_detection")
    # mongo = PyMongo(app)
    # db = mongo.db
    # db = client["emotiondetection"]
    # CORS(app)

    app_routes(app)

    return app


if __name__ == "__main__":
    app = init_app()
    app.run(host="0.0.0.0", port=5000, debug=True)
