# EME2020_project
# Machine learning project designed to assess the likelihood of someone dropping out given a myriad of factors.
# Data will be preprocessed such that its standardized and well understood by our Logistic Regression model.
# The model will output a probability score from 0.0 to 1


```markdown
# Student Dropout Prediction

A machine learning project designed to assess the likelihood of someone dropping out, given a myriad of factors [7]. Data is preprocessed so it's standardized and well understood by the Logistic Regression model [7], and the model outputs a probability score from 0.0 to 1 [7].

## How It Works

1. **Data Loading & Cleaning** — `training_model.py` loads `students_dropout_academic_success.csv` with a standard comma separator [6] and cleans column headers by stripping stray tabs, spaces, and quotes [6].
2. **Preprocessing & Feature Filtering** — The target column is mapped to binary, and features are filtered down to the ones present in the dataset [6].
3. **Train/Test Split & Scaling** — Data is split into training and test sets and scaled using `StandardScaler` (fit on training data, applied to both) [6].
4. **Model Training** — A Logistic Regression model is trained on the scaled data.
5. **Saving Artifacts** — The trained model, scaler, and feature list are saved via `joblib` to `model.pkl`, `scaler.pkl`, and `features.pkl` [6].
6. **Serving Predictions** — `app.py` imports Flask, joblib, pandas, os, and sys [1], and defines routes: `/` renders the input form (`HTML_PAGE`) using the saved features, binary feature labels, and field config [1], while `/predict` accepts a POST request, parses JSON input, and builds the feature row before predicting [1].

## Features Used

Application mode, Application order, Course, Daytime/evening attendance, Previous qualification, Previous qualification (grade), Nacionality, Admission grade, Displaced, Educational special needs, Debtor, Tuition fees up to date, Gender, Scholarship holder, Age at enrollment, International, Curricular units 1st sem (credited/enrolled/evaluations/approved/grade/without evaluations), Curricular units 2nd sem (credited/enrolled/evaluations/approved/grade/without evaluations), Unemployment rate, Inflation rate, GDP [2][4].

## Model Details

- Algorithm: Logistic Regression (scikit-learn, solver `lbfgs`, `max_iter=1000`) [3]
- Preprocessing: `StandardScaler` with mean and standard deviation computed from training data [4]
- scikit-learn version used: 1.9.1 [3][4]

## Project Structure

| File | Purpose |
|---|---|
| `setup.py` | Installs/checks required dependencies [5] |
| `training_model.py` | Loads data, trains the model, saves artifacts [6] |
| `app.py` | Flask app serving the prediction form and `/predict` endpoint [1] |
| `model.pkl` | Trained Logistic Regression model [3] |
| `scaler.pkl` | Fitted StandardScaler [4] |
| `features.pkl` | List of features used in training [2] |

## Requirements

- flask
- pandas
- numpy
- scikit-learn
- joblib
- tqdm

(Defined in `setup.py` as the required packages, with pip name mapped to import name) [5]

## Setup & Usage

1. **Install dependencies**
   ```bash
   python setup.py
   ```
   This checks each required package and installs any missing ones automatically [5].

2. **Add the dataset**
   Place `students_dropout_academic_success.csv` in the project folder.

3. **Train the model**
   ```bash
   python training_model.py
   ```
   This loads the dataset [6], preprocesses and filters features [6], splits and scales the data [6], trains the Logistic Regression model, and saves `model.pkl`, `scaler.pkl`, and `features.pkl` [6]. Once finished, it prints: "Model, Scaler, and Features saved successfully!" [6]

4. **Run the web app**
   ```bash
   python app.py
   ```
   This starts the Flask server, which loads the saved artifacts and serves the prediction form [1].

5. **Make a prediction**
   Open the app in your browser, fill in the form, and submit — this sends a POST request to `/predict`, which builds the input row and returns a dropout probability [1].

## Output

The model outputs a probability score from 0.0 to 1 representing likelihood of dropout [7].

## Notes

After the final installation step, you can run `training_model.py` followed by `app.py` [5].
```
