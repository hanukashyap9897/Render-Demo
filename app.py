from flask import Flask, request, render_template
import pickle
import numpy as np

# Load model
with open("model.pkl", 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form data
        features = [float(request.form.get(key)) for key in ['StudentID', 'CGPA', 'Internships', 'Projects', 'AptitudeTestScore']]
        final_features = np.array([features])

        # Predict
        prediction = model.predict(final_features)
        output = 'Placed' if prediction[0] == 1 else 'Not Placed'

        return render_template('index.html', prediction_text=f'Prediction: {output}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)
