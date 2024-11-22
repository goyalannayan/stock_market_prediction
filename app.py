from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model
with open("stock_price_model.pkl", "rb") as file:
    model = pickle.load(file)

@app.route('/')
def index():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input data
        input_data = request.form
        date = pd.to_datetime(input_data['Date']).toordinal()  # Convert date to ordinal
        open_price = float(input_data['Open'])
        high = float(input_data['High'])
        low = float(input_data['Low'])
        adj_close = float(input_data['Adj Close'])
        volume = float(input_data['Volume'])

        # Prepare data for prediction
        input_features = np.array([[date, open_price, high, low, adj_close, volume]])
        prediction = model.predict(input_features)[0]

        # Return the result to the template
        return render_template('index.html', prediction=round(prediction, 2))

    except Exception as e:
        return render_template('index.html', error=str(e), prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
