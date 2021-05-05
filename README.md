ISEAR - Emotion-Detection Project
==============================

Emotion Detection Training Project at Fusemachines - Training for AI Engineers

Project Organization
------------
```
├── api
│   ├── app.py
│   ├── config
│   │   ├── config.py
│   ├── data
│   │   └── db
│   ├── services
│   │   ├── predictor.py
│   ├── static
│   │   ├── banner.jpg
│   │   └── feedback.png
│   ├── templates
│   │   ├── feedback.png
│   │   ├── index.html
│   │   └── predict.html
│   └── urls.py
├── checkpoints
│   ├── processor
│   │   └── processor.pkl
│   └── SKLEARN
│       └── multinominal_naive_bayes.pkl
├── data
│   ├── processed
│   │   └── processed_data.pkl
│   └── raw
│       └── ISEAR.csv
├── Dockerfile
├── emotiondetection
│   ├── config
│   │   ├── config.py
│   │   ├── model_params.py
│   │   └── tfidf_parameters.py
│   ├── data
│   │   └── make_dataset.py
│   ├── dispatcher
│   │   ├── dispatcher.py
│   ├── features
│   │   ├── build_features.py
│   ├── main.py
│   ├── models
│   │   ├── test_model.py
│   │   └── train_model.py
│   ├── README.md
│   ├── utils
│   │   ├── lemmatiser.py
│   │   ├── save_object.py
│   │   ├── stemmer.py
│   │   ├── text_processor.py
│   │   └── tfidfprocessor.py
│   └── visualisation
│       └── visualize.py
├── LICENSE
├── Makefile
├── notebooks
│   ├── emotiondetection.ipynb
├── README.md
├── requirements.txt
├── test_environment.py
└── tox.ini
    
--------
```
<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

## Installing cookiecutter
```
conda config --add channels conda-forge
conda install cookiecutter
```

## Installing pre-commit 
```
conda install pre-commit
```
[Pre-commit hooks](https://pre-commit.com/) are also added. `.pre-commit-config.yaml` is the pre-commit configuration file.
