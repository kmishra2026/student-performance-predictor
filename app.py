from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html', result=None, reading=0, writing=0)

@app.route('/predict', methods=['POST'])
def predict():
    gender = request.form['gender']
    race = request.form['race']
    education = request.form['education']
    lunch = request.form['lunch']
    prep = request.form['prep']
    reading = float(request.form['reading'])
    writing = float(request.form['writing'])
    data = {'gender': [gender], 'race/ethnicity': [race], 'parental level of education': [education], 'lunch': [lunch], 'test preparation course': [prep], 'reading score': [reading], 'writing score': [writing]}
    df = pd.DataFrame(data)
    df = pd.get_dummies(df)
    df = df.reindex(columns=model.feature_names_in_, fill_value=0)
    prediction = model.predict(df)
    return render_template('index.html', result=round(prediction[0], 2), reading=int(reading), writing=int(writing))

if __name__ == '__main__':
    app.run(debug=True)
