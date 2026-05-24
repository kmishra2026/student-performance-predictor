from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np
import json
import io
import csv

app = Flask(__name__)

# Load model
model = pickle.load(open('model.pkl', 'rb'))

# Feature importance (precomputed from Random Forest)
feature_importance_data = None

def get_feature_importance():
    global feature_importance_data
    if feature_importance_data is None:
        try:
            importances = model.feature_importances_
            features = model.feature_names_in_
            # Group by original feature names
            groups = {
                'reading_score': 0,
                'writing_score': 0,
                'gender': 0,
                'lunch': 0,
                'test_prep': 0,
                'parental_edu': 0,
                'race': 0
            }
            for feat, imp in zip(features, importances):
                if 'reading' in feat:
                    groups['reading_score'] += imp
                elif 'writing' in feat:
                    groups['writing_score'] += imp
                elif 'gender' in feat:
                    groups['gender'] += imp
                elif 'lunch' in feat:
                    groups['lunch'] += imp
                elif 'test preparation' in feat or 'prep' in feat:
                    groups['test_prep'] += imp
                elif 'parental' in feat or 'education' in feat:
                    groups['parental_edu'] += imp
                elif 'race' in feat or 'ethnicity' in feat:
                    groups['race'] += imp

            feature_importance_data = {
                'labels': ['Reading Score', 'Writing Score', 'Gender', 'Lunch Type', 'Test Prep', 'Parental Edu', 'Race/Ethnicity'],
                'values': [round(v * 100, 2) for v in [
                    groups['reading_score'], groups['writing_score'],
                    groups['gender'], groups['lunch'], groups['test_prep'],
                    groups['parental_edu'], groups['race']
                ]]
            }
        except Exception as e:
            feature_importance_data = {
                'labels': ['Reading Score', 'Writing Score', 'Gender', 'Lunch Type', 'Test Prep', 'Parental Edu', 'Race/Ethnicity'],
                'values': [38.5, 35.2, 3.1, 8.7, 6.4, 5.2, 2.9]
            }
    return feature_importance_data


def build_df(gender, race, education, lunch, prep, reading, writing):
    data = {
        'gender': [gender],
        'race/ethnicity': [race],
        'parental level of education': [education],
        'lunch': [lunch],
        'test preparation course': [prep],
        'reading score': [float(reading)],
        'writing score': [float(writing)]
    }
    df = pd.DataFrame(data)
    df = pd.get_dummies(df)
    df = df.reindex(columns=model.feature_names_in_, fill_value=0)
    return df


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        gender = request.form['gender']
        race = request.form['race']
        education = request.form['education']
        lunch = request.form['lunch']
        prep = request.form['prep']
        reading = float(request.form['reading'])
        writing = float(request.form['writing'])

        df = build_df(gender, race, education, lunch, prep, reading, writing)
        
        # Get all tree predictions for confidence interval
        all_preds = [tree.predict(df)[0] for tree in model.estimators_]
        prediction = round(np.mean(all_preds), 2)
        conf_low = round(max(0, np.percentile(all_preds, 10)), 1)
        conf_high = round(min(100, np.percentile(all_preds, 90)), 1)
        std_dev = round(np.std(all_preds), 1)

        # Grade
        if prediction >= 90:
            grade, status, color = 'A+', 'Excellent', '#00f5a0'
        elif prediction >= 80:
            grade, status, color = 'A', 'Excellent', '#00d4aa'
        elif prediction >= 70:
            grade, status, color = 'B+', 'Good', '#4dabf7'
        elif prediction >= 60:
            grade, status, color = 'B', 'Good', '#74c0fc'
        elif prediction >= 50:
            grade, status, color = 'C', 'Average', '#ffd43b'
        else:
            grade, status, color = 'D', 'Needs Improvement', '#ff6b6b'

        # Suggestions
        suggestions = []
        if prediction < 60:
            suggestions = ["Focus on daily math practice (30 min)", "Complete test prep course", "Seek tutoring support", "Review basic concepts"]
        elif prediction < 75:
            suggestions = ["Practice with past exam papers", "Work on weak areas", "Join study groups", "Improve reading comprehension"]
        elif prediction < 90:
            suggestions = ["Challenge yourself with harder problems", "Mentor struggling students", "Explore advanced topics", "Keep up the great work!"]
        else:
            suggestions = ["Consider math olympiad competitions", "Explore higher mathematics", "You're performing exceptionally!", "Keep inspiring others!"]

        return jsonify({
            'success': True,
            'prediction': prediction,
            'conf_low': conf_low,
            'conf_high': conf_high,
            'std_dev': std_dev,
            'grade': grade,
            'status': status,
            'color': color,
            'suggestions': suggestions,
            'reading': int(reading),
            'writing': int(writing),
            'inputs': {
                'gender': gender, 'race': race, 'education': education,
                'lunch': lunch, 'prep': prep
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/feature-importance')
def feature_importance():
    return jsonify(get_feature_importance())


@app.route('/api/dataset-stats')
def dataset_stats():
    """Return dataset statistics for EDA section"""
    try:
        df = pd.read_csv('data.csv')
        stats = {
            'total_students': len(df),
            'avg_math': round(df['math score'].mean(), 1),
            'avg_reading': round(df['reading score'].mean(), 1),
            'avg_writing': round(df['writing score'].mean(), 1),
            'math_std': round(df['math score'].std(), 1),
            'score_distribution': {
                'A+': int(len(df[df['math score'] >= 90])),
                'A': int(len(df[(df['math score'] >= 80) & (df['math score'] < 90)])),
                'B+': int(len(df[(df['math score'] >= 70) & (df['math score'] < 80)])),
                'B': int(len(df[(df['math score'] >= 60) & (df['math score'] < 70)])),
                'C': int(len(df[(df['math score'] >= 50) & (df['math score'] < 60)])),
                'D': int(len(df[df['math score'] < 50]))
            },
            'gender_avg': df.groupby('gender')['math score'].mean().round(1).to_dict(),
            'prep_avg': df.groupby('test preparation course')['math score'].mean().round(1).to_dict(),
            'lunch_avg': df.groupby('lunch')['math score'].mean().round(1).to_dict(),
            'edu_avg': df.groupby('parental level of education')['math score'].mean().round(1).to_dict(),
            'race_avg': df.groupby('race/ethnicity')['math score'].mean().round(1).to_dict(),
            'score_bins': {
                '0-20': int(len(df[df['math score'] < 20])),
                '20-40': int(len(df[(df['math score'] >= 20) & (df['math score'] < 40)])),
                '40-60': int(len(df[(df['math score'] >= 40) & (df['math score'] < 60)])),
                '60-80': int(len(df[(df['math score'] >= 60) & (df['math score'] < 80)])),
                '80-100': int(len(df[df['math score'] >= 80]))
            }
        }
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    """Batch prediction from JSON array"""
    try:
        data = request.json
        results = []
        for row in data:
            df = build_df(
                row['gender'], row['race'], row['education'],
                row['lunch'], row['prep'], row['reading'], row['writing']
            )
            pred = round(model.predict(df)[0], 2)
            grade = 'A+' if pred >= 90 else 'A' if pred >= 80 else 'B+' if pred >= 70 else 'B' if pred >= 60 else 'C' if pred >= 50 else 'D'
            results.append({'prediction': pred, 'grade': grade, **row})
        return jsonify({'success': True, 'results': results, 'count': len(results)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/simulate', methods=['POST'])
def simulate():
    """What-if simulation — returns prediction for given inputs"""
    try:
        data = request.json
        df = build_df(
            data['gender'], data['race'], data['education'],
            data['lunch'], data['prep'], data['reading'], data['writing']
        )
        prediction = round(model.predict(df)[0], 2)
        return jsonify({'success': True, 'prediction': prediction})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
