from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

#loading trained model
with open('wmodel.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        #to get form values from the features model is trained on
        features = [
            float(request.form['MinTemp']),
            float(request.form['MaxTemp']),
            float(request.form['Rainfall']),
            float(request.form['WindGustSpeed']),
            float(request.form['WindSpeed9am']),
            float(request.form['WindSpeed3pm']),
            float(request.form['Humidity9am']),
            float(request.form['Humidity3pm']),
            float(request.form['Pressure9am']),
            float(request.form['Pressure3pm']),
            float(request.form['Temp9am']),
            float(request.form['Temp3pm']),
            float(request.form['RainToday']),
            float(request.form['Location']),
            float(request.form['Month']),
            float(request.form['Day']),
        ]

        prediction_value = model.predict([features])[0]
        probability      = model.predict_proba([features])[0][1]

        if prediction_value == 1:
            result = f"🌧️ Rain Tomorrow — {probability*100:.1f}% chances of rain"
        else:
            result = f"☀️ No Rain Tomorrow — {(1-probability)*100:.1f}% chances of staying dry"

    except Exception as e:
        result = f"Error: {str(e)}"

    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)