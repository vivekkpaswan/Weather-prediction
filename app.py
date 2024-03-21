from flask import Flask, render_template, request
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import numpy as np

app = Flask(__name__)

def predict_weather(input_data):
    
    clf = joblib.load('your_model.pkl')

    #input_data = input_data.reshape(1, -1)

    scaler = StandardScaler()
    input_data = scaler.fit_transform([input_data])
    
    predicted_weather = clf.predict(input_data)

    # Return the predicted weather
    if predicted_weather == 0:
        return "Drizzle"
    elif predicted_weather == 1:
        return "Fog"
    elif predicted_weather == 2:
        return "Rain"
    elif predicted_weather == 3:
        return "Snow"
    else:
        return "Sun"

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Assuming input data is provided as a form input
        input_data = np.array([
            float(request.form['feature1']),
            float(request.form['feature2']),
            float(request.form['feature3']),
            float(request.form['feature4'])
        ])

        predicted_weather = predict_weather(input_data)
        return render_template('result.html', weather=predicted_weather)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
