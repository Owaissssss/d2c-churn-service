# D2C Customer Churn Intelligence — Part 4: Live FastAPI Production Scoring Service

## Student & Course Information
* **Student Name:** Mohammad Owais
* **Registration Number:** iitp_aiml_2506198
* **Email:** zkzonfamily@gmail.com
* **Course:** IIT Patna AI/ML Course Capstone

This repository houses **Part 4** of the Capstone project, establishing a production-grade FastAPI microservice that wraps our optimized Random Forest Classifier to serve real-time e-commerce inference requests.

## Microservice Architecture Layout
* `requirements.txt`: Frozen production environmental package boundaries ensuring unified tracking.
* `app.py`: High-performance FastAPI application script constructing Pydantic validation boundaries, operational health checks, and data dummy formatting.
* `test_pipeline.py`: Automated verification framework spawning background local servers, sending programmatic requests, and evaluating inference signatures.
* `model.pkl`: Serialized, pre-compiled advanced machine learning classifier weight parameters.
* `rfm_modeling_snapshot.csv`: Historical data tracking schema properties utilized for live dummy dimension formatting.
* `monitoring_plan.md`: Comprehensive strategy mapping post-deployment tracking metrics (data drift, prediction stability, and system errors) along with the operational responsible-use framework.
## API Endpoint Specifications

### 1. Health Gateway Check
* **Method**: `GET`
* **Path**: `/`
* **Response**: Returns a confirmation payload declaring service state and threshold limits.

### 2. Live Churn Prediction Scoring
* **Method**: `POST`
* **Path**: `/predict`
* **Request Payload**: Explicit JSON format mapping raw customer behavioral matrices (Recency, Frequency, Monetary, and Categorical strings).
* **Inference Logic**: Dynamically maps categorical tags using a fallback data framework, routes arrays to `model.pkl`, applies the optimized `0.40` threshold boundary, and returns targeted retention actions.

## Operational Verification Guide
1. Launch your virtual terminal framework.
2. Initialize trapped module constraints:
   ```bash
   pip install -r requirements.txt