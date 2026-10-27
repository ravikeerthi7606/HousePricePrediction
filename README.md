# House Price Prediction

An end-to-end Machine Learning project for predicting house prices using Scikit-learn, MLflow, FastAPI, and Docker.

## Overview

This project develops a machine learning pipeline to predict residential house sale prices from property features.

The project covers model development, preprocessing, evaluation, experiment tracking, model artifact management, API development, and containerized deployment.

## Tech Stack

- **Language:** Python 3.10
- **Machine Learning:** Scikit-learn
- **Data Processing:** Pandas, NumPy
- **Experiment Tracking:** MLflow
- **API:** FastAPI
- **Model Serialization:** Joblib
- **Testing:** Pytest
- **Deployment:** Docker

## Dataset

- **Dataset:** Ames Housing dataset
- **Rows:** 2,930
- **Features:** 79

The dataset contains residential property attributes used for house price prediction.


## Machine Learning Workflow

1. Load and inspect the dataset.
2. Perform exploratory data analysis and data validation.
3. Split the data into training, validation, and test sets.
4. Build preprocessing and modeling pipelines.
5. Train and evaluate candidate models.
6. Track experiments and metrics using MLflow.
7. Select a candidate model.
8. Save model artifacts and metadata.
9. Serve predictions using FastAPI.
10. Package the application using Docker.

## Model Evaluation

The selected provisional model is **Gradient Boosting**.

### Previously observed test metrics

| Metric | Value |
|---|---:|
| RMSE | 25,443.64 |
| MAE | 14,353.67 |
| R² | 0.9102 |

Additional diagnostic result:
- Negative predictions observed: 0

**Evaluation note:** The test partition was inspected during candidate model comparison. These results are previously observed diagnostic results, not an unbiased final performance estimate from an untouched test set.

RMSE and MAE are reported in the dataset's target currency units.

## Model Tracking

MLflow is used to track model experiments and evaluation metrics.

Tracked information may include:
- Model parameters
- Evaluation metrics
- Run information
- Model artifacts

The prediction application loads the selected model artifact from the project's model directory.

## API

The model is served using FastAPI.

### Run locally

Create and activate a Python 3.10 virtual environment, then install dependencies.

```bash
python -m venv .venv
```

Windows:

```cmd
.venv\Scripts\activate
pip install -r requirements.txt
```

Start the API:

```bash
uvicorn HousePrice.api:app --app-dir src --reload
```

### API documentation

Open:

- Swagger UI: http://localhost:8000/docs
- OpenAPI schema: http://localhost:8000/openapi.json

### Prediction

Use the `/predict` endpoint with the expected input schema.

Refer to Swagger UI for the exact request fields and response format.

*Add a verified request/response example before publishing.*

## Docker Deployment

The Docker image has been built and the prediction API has been successfully tested in a running container.

### Build the image

```bash
docker build -t house-price-api .
```

### Run the container

```bash
docker run --rm -p 8000:8000 house-price-api
```

Access the API:

http://localhost:8000/docs

### Stop the container

Press `Ctrl + C` in the running terminal.

## Testing

Run the test suite from the project root:

```bash
python -m pytest
```

## Project Structure

```text
House Price Prediction/
├── src/
│   └── HousePrice/
│       ├── api.py
│       ├── predict.py
│       └── ...
├── models/
│   └── ...
├── tests/
│   └── ...
├── Dockerfile
├── requirements.txt
├── .dockerignore
├── .gitignore
└── README.md
```

Adjust this structure to match the actual repository.

## Limitations

- Predictions are estimates and not professional property valuations.
- Performance may vary across locations and market conditions.
- Previously observed test metrics should not be interpreted as a fresh unbiased final evaluation.
- Model performance depends on data quality and the distribution of input features.

## Future Improvements

- Establish a fresh, untouched final evaluation set.
- Add input validation and data drift monitoring.
- Add CI with automated testing.
- Improve model serving and deployment automation.
- Add monitoring for prediction quality and API latency.

## Author

**Ravi Keerthi R.**

B.Tech, Mechatronics Engineering

- GitHub: https://github.com/ravikeerthi7606
- LinkedIn: https://www.linkedin.com/in/ravikeerthi7606

---

*Built as an end-to-end Machine Learning engineering project.*
