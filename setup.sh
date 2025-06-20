#!/bin/bash
pip install -r requirements.txt
python -m nltk.downloader punkt stopwords averaged_perceptron_tagger wordnet omw-1.4
