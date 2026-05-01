# Student Performance Predictor

### Live Demo — https://student-performance-predictor-01n6.onrender.com

An AI-powered web application that predicts student math scores using Machine Learning — built with Flask, trained on 1,000+ student records, and deployed live on Render.

---

## About The Project

This project is an end-to-end Machine Learning web application developed as part of an internship project. It takes student details as input and predicts their Math score using a Random Forest Regression model trained on real student data.

---

## Features

- Math Score Prediction using Random Forest ML algorithm
- 7 Input Features — Gender, Race/Ethnicity, Parental Education, Lunch Type, Test Preparation, Reading Score, Writing Score
- Performance Grade — Automatic A+, A, B+, B, C, D grading based on predicted score
- Score Gauge — Visual progress bar showing score out of 100
- Radar Chart — Interactive chart comparing Reading, Writing, and Math scores
- Bar Chart — Side by side comparison of all three subject scores
- Personalized Suggestions — Smart study tips based on predicted score
- Dark AI Theme — Professional modern UI design

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | HTML, CSS, JavaScript, Chart.js |
| Backend | Python, Flask |
| Machine Learning | scikit-learn, pandas, pickle |
| Deployment | Render (Gunicorn) |
| Version Control | Git, GitHub |

---

## Machine Learning Details

| Property | Details |
|---|---|
| Algorithm | Random Forest Regressor |
| Dataset | Students Performance in Exams (Kaggle) |
| Training Records | 1,000 students |
| Input Features | 7 |
| Target Variable | Math Score (0 to 100) |
| Model Format | Pickle (.pkl) |

---

## Project Structure

student-performance-predictor/
├── app.py                   Flask web application and routes
├── train.py                 ML model training script
├── model.pkl                Trained Random Forest model
├── data.csv                 Student performance dataset
├── requirements.txt         Python dependencies
├── Procfile                 Deployment configuration
└── templates/
└── index.html           Frontend UI with charts

---

## How It Works

1. User fills the form with student details and previous scores
2. Flask receives the POST request and extracts form data
3. Data is preprocessed using get_dummies for categorical variables
4. Random Forest model predicts the math score
5. Results are displayed with grade, charts, and personalized suggestions

---

## Performance Grades

| Score Range | Grade | Status |
|---|---|---|
| 90 to 100 | A+ | Excellent |
| 80 to 89 | A | Excellent |
| 70 to 79 | B+ | Good |
| 60 to 69 | B | Good |
| 50 to 59 | C | Average |
| Below 50 | D | Needs Improvement |

---

## Quick Start

git clone https://github.com/kmishra2026/student-performance-predictor.git
cd student-performance-predictor
pip install -r requirements.txt
python train.py
python app.py

Open browser at http://127.0.0.1:5000

---

## Deployment

Deployed on Render using Gunicorn WSGI server.

Build Command — pip install -r requirements.txt
Start Command — gunicorn app:app

Live App — https://student-performance-predictor-01n6.onrender.com

---

## License

This project is licensed under the MIT License.

---

## Author

Khushi Mishra
Computer Engineering Student
G.H. Raisoni College of Engineering and Management, Pune
GitHub — https://github.com/kmishra2026
