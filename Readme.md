# Student Performance Prediction App

This project predicts a student's math score using academic and demographic attributes such as gender, race, parental education, lunch type, test preparation, reading score, and writing score.

It combines:
- a machine learning pipeline
- a Flask web app
- a simple browser-based frontend
- MLflow experiment tracking
- Docker deployment support

## Project Overview

The app performs the following workflow:

- loads the student dataset
- splits data into train and test sets
- preprocesses numerical and categorical features
- trains multiple regression models
- evaluates them using R² and error metrics
- selects the best model
- saves the model and preprocessing object
- exposes a prediction API and a simple frontend form

## Tech Stack

- Python 3.11
- Flask
- scikit-learn
- XGBoost
- CatBoost
- MLflow
- Pandas / NumPy
- Jupyter Notebook
- Docker
- Gunicorn

## Project Structure

```text
MLOPS/
├── app.py
├── dockerfile
├── requirements.txt
├── setup.py
├── Readme.md
├── template.py
├── artifacts/
│   ├── model.pkl
│   ├── preprocessor.pkl
│   ├── raw.csv
│   ├── train.csv
│   └── test.csv
├── logs/
├── mlruns/
├── notebook/
│   ├── StudentsPerformance.csv
│   ├── 1 . EDA STUDENT PERFORMANCE  (1).ipynb
│   └── 2. MODEL TRAINING.ipynb
├── src/
│   └── mlproject/
│       ├── __init__.py
│       ├── exceptions.py
│       ├── logger.py
│       ├── utils.py
│       ├── components/
│       │   ├── data_ingestion.py
│       │   ├── data_transformation.py
│       │   └── model_trainer.py
│       └── pipelines/
│           ├── prediction_pipeline.py
│           └── training_pipeline.py
├── templates/
│   └── index.html
└── catboost_info/
```

## How to Run Locally (Without Docker)

### 1. Clone the project

```bash
git clone <repository-url>
cd MLOPS
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

Windows:
```bash
venv\Scripts\activate
```

Linux/macOS:
```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the app

```bash
python app.py
```

### 5. Open the app in browser

```text
http://localhost:5000
```

The app will show a simple HTML form where the user can enter student data and click the prediction button.

## Prediction API

The app exposes a REST API for predictions.

### Endpoint

```http
POST /predict
```

### Example JSON request

```json
{
  "gender": "female",
  "race/ethnicity": "group C",
  "parental level of education": "some college",
  "lunch": "standard",
  "test preparation course": "completed",
  "writing score": 80,
  "reading score": 85
}
```

### Example response

```json
{
  "prediction": 78.42,
  "unit": "math_score"
}
```

## Frontend

The app includes a simple HTML frontend in the `templates` folder:

- user enters input values
- form is submitted to `/predict`
- predicted score is displayed on the page

## MLflow Tracking

The training pipeline logs metrics and model information using MLflow.

Run MLflow UI locally:

```bash
mlflow ui
```

Then open:

```text
http://localhost:5000
```

## Docker Deployment

A Dockerfile is included for deployment.

### Build the image

```bash
docker build -t student-performance-app .
```

### Run the container

```bash
docker run -p 5000:5000 student-performance-app
```

Then open:

```text
http://localhost:5000
```

## Model Artifacts

After training, the project saves important model files such as:

- `artifacts/model.pkl`
- `artifacts/preprocessor.pkl`
- `artifacts/train.csv`
- `artifacts/test.csv`

## Notes

- If the model files are not available, the app will try to train them automatically on startup.
- This project is a simple MLOps learning project and is suitable for local deployment and experimentation.

## Future Improvements

You can extend this project by adding:

- a more advanced frontend design
- real validation and error handling
- CI/CD pipeline
- cloud deployment on AWS/Azure/GCP
- model registry and versioning
- monitoring and retraining workflow

## Conclusion

This project is a practical example of building a small machine learning application with MLOps best practices, including training, tracking, deployment, and a simple user interface.
