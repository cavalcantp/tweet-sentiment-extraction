# Tweet Sentiment Extraction
This project is organized in the following directories

```
- traning: directory with a notebook to train the model offline
- tweet_sentiment_service: this is the inference service
- tests: unit tests for the service
- Dockerfile: to build service img
- test_workflow: in real project this would be a e2e test workflow, including testing the service with PROD settings. Here we limit to testing the API with PROD settings (trained weights).
- analyze_latency: file in which latency of service is analyzed and some intial improvements are proposed.
```

If you want to run this service, you need to add the following folder and files:
```
- config: stores configuration files for the roBERTa model (pretrained-roberta-base.h5, config-roberta-base.json, merges-roberta-base.txt, vocab-roberta-base.json)

- weights_final.h5
```

## Service
The inference service is built on top of two modules: the model module (SentimentModel) and the inference module (SentimentExtractor). 
- SentimentModel handles building the model, loading trained weights and prediction (in the strict sense). 
- SentimentExtractor handles tokenization, datapreprocessing and prediction in a more broad context (considering the preprocessing, encoding of inputs and decoding of outputs).

Goal of separation is to keep responsibilities isolated (Single Responsability Principle), making future changes simpler.

The service itself is an API (app.py)

## Local development and testing
In a dedicated virtual environment, run ```poetry install```

### Unit testing
The unit tests were developed mocking functionalities, trying to isolate the targeted code at each test. The idea behind using mocking was to accelerate test execution (ex: in a scenario of automatic test execution within a CICD pipeline)

Ideally, mocking should be avoided and a light (simple) version of the model should be used for testing, which was not done here, due to time constrains. In a large scale project, what could be done is: have on a registry (MLFlow), the weights tagged as PROD, which would be the version deployed, and the weights tagged as TEST, which would be a version used in unit testing. This is discussed better in section *IMPORTANT OBSERVATION*

```
PYTHONPATH=. pytest  -vv**
```

#### End to End Testing (including testing the API)

Build the image and spin up a container, then run the e2e workflow
to test for standard request.
```
docker build -t tweet_sentiment_service .

docker run -p 8000:8000 tweet_sentiment_service
```

## IMPORTANT OBSERVATION
I added the weights and config files directly to the Docker image, but in a real production, I would use a model registry like MLFlow. That would
- keep the image build more efficient in the CICD pipeline.
- enable retraining the model and deploying it to PROD, without the need of redeploying the service.
- possibilitate getting rid of some of the mocking in the unit test, by using a light-version model during test execution in the CICD pipeline, like mentioned previously



