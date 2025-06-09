from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('heart_disease_rf_model.pkl')

features = ['age', 'sex', 'chest pain type', 'resting bp s', 'cholesterol',
            'fasting blood sugar', 'resting ecg', 'max heart rate',
            'exercise angina', 'oldpeak', 'ST slope']

# Dropdown mappings
dropdown_options = {
    'sex': [(0, 'Female'), (1, 'Male')],
    'chest pain type': [
        (0, 'Typical Angina'),
        (1, 'Atypical Angina'),
        (2, 'Non-anginal Pain'),
        (3, 'Asymptomatic')
    ],
    'fasting blood sugar': [(0, '< 120 mg/dl'), (1, '> 120 mg/dl')],
    'resting ecg': [
        (0, 'Normal'),
        (1, 'ST-T wave abnormality'),
        (2, 'Left ventricular hypertrophy')
    ],
    'exercise angina': [(0, 'No'), (1, 'Yes')],
    'ST slope': [
        (0, 'Upsloping'),
        (1, 'Flat'),
        (2, 'Downsloping')
    ]
}

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            input_data = [float(request.form[feature]) for feature in features]
            prediction = model.predict([input_data])[0]
            prediction = 'Heart Disease Detected' if prediction == 1 else 'No Heart Disease'
        except Exception as e:
            prediction = f"Error: {str(e)}"
    return render_template('index.html', features=features, prediction=prediction, dropdown_options=dropdown_options)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
