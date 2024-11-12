# Tweet Sentiment Extraction
This project is organized in the following directories
- config: stores configuration files for the roBERTa model
- traning: directory with a notebook to train the model offline and a notebook to test the inference workflow (CHANGE THAT)
- tweet_sentiment_service: this is the inference service

## tweet_sentiment_service
The inference service is built on top of two modules: the model module (SentimentModel) and the inference module (SentimentExtractor). 
- SentimentModel handles building the model, loading trained weights and prediction (in the strict sense). 
- SentimentExtractor handles tokenization, datapreprocessing and prediction in a more broad context (considering the preprocessing, encoding of inputs and decoding of outputs).

Goal of separation is to keep responsibilities isolated (Single Responsability Principle), making future changes simpler.

The service itself is an API (app.py)

## Setup for local development and testing
In a dedicated virtual environment, run **poetry install**

