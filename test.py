from flask import Flask, request, render_template
import pickle

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('demo.html')  # Serve demo.html

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the form
    pH = float(request.form['pH'])
    pH_trend = float(request.form['pH_trend'])
    pH_variability = float(request.form['pH_variability'])

    # Make a prediction
    input_data = [[pH, pH_trend, pH_variability]]
    prediction = model.predict(input_data)

    # Map prediction to wound type
    wound_types = {0: 'chronic', 1: 'healing', 2: 'infected'}
    predicted_wound_type = wound_types[prediction[0]]

    # Pass the input values and prediction back to the template
    return render_template('demo.html', 
                           prediction_text=f'Predicted Wound Type: {predicted_wound_type}',
                           pH=pH,
                           pH_trend=pH_trend,
                           pH_variability=pH_variability)

if __name__ == '__main__':
    app.run(debug=True)