from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('model/best_model.pkl')

# Render the home page
@app.route('/')
def home():
    return render_template('index.html')

# Handle the form submission and predict
@app.route('/predict', methods=['POST'])
def predict():
    # Get the form data
    form_data = request.form.to_dict()
    
    # Prepare the data for prediction
    new_data = pd.DataFrame({
        'age': [int(form_data['age'])],
        'sex': [int(form_data['sex'])],
        'cp': [int(form_data['cp'])],
        'trtbps': [int(form_data['trtbps'])],
        'chol': [int(form_data['chol'])],
        'fbs': [int(form_data['fbs'])],
        'restecg': [int(form_data['restecg'])],
        'thalachh': [int(form_data['thalachh'])],
        'exng': [int(form_data['exng'])],
        'oldpeak': [float(form_data['oldpeak'])],
        'slp': [int(form_data['slp'])],
        'caa': [int(form_data['caa'])],
        'thall': [int(form_data['thall'])]
    })

    # Make prediction
    prediction = model.predict(new_data)
    
    # Decode the prediction
    prediction_label = 'Heart Attack' if prediction[0] == 1 else 'No Heart Attack'

    # Render the result page with prediction
    return render_template('result.html', prediction=prediction_label)

if __name__ == '__main__':
    app.run(debug=True)
