# Python image as the base, compatible with dependencies
FROM python:3.9.5-slim

# Working directory in the container
WORKDIR /app

# Copy requirements and install them
COPY pyproject.toml ./
RUN pip install poetry && poetry config virtualenvs.create false && poetry install --no-dev
RUN uvicorn --version 

# Copy the application code to the container
COPY tweet_sentiment_service /app/tweet_sentiment_service
COPY config /app/config
COPY weights_final.h5 /app/weights_final.h5

# Expose the port on which the API will run
EXPOSE 8000

# Run the app using Uvicorn
CMD ["poetry", "run", "uvicorn", "tweet_sentiment_service.app:app", "--host", "0.0.0.0", "--port", "8000"]
