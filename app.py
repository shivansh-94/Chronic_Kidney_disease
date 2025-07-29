from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the pretrained model and scaler once when the app starts
model = pickle.load(open('ckd_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# List of numeric columns that need scaling (must match training data order)
num_cols = [
    'age', 'blood_pressure', 'blood_glucose_random', 'blood_urea', 'serum_creatinine',
    'sodium', 'potassium', 'hemoglobin', 'packed_cell_volume',
    'white_blood_cell_count', 'red_blood_cell_count'
]

# Features order list in the exact order your model expects
feature_order = [
    'age', 'blood_pressure', 'specify_gravity', 'albumin', 'sugar', 'red_blood_cells', 
    'pus_cell', 'pus_cell_clumps', 'bacteria', 'blood_glucose_random', 'blood_urea', 
    'serum_creatinine', 'sodium', 'potassium', 'hemoglobin', 'packed_cell_volume', 
    'white_blood_cell_count', 'red_blood_cell_count', 'hypertension', 'diabetes_mellitus', 
    'coronary_artery_disease', 'appetite', 'pedal_edema', 'anemia'
]

# Map form field name (HTML) to model feature name if they differ
# For example, in indexnew.html, you have 'specify_gravity', so match it exactly

@app.route('/')
def home():
    return render_template('indexnew.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form data in correct order as floats/ints
        input_data = []
        for feature in feature_order:
            val = request.form.get(feature)  
            if val is None:
                # Return error or handle missing input gracefully
                return render_template('indexnew.html', error=f"Missing input for {feature}")
            input_data.append(val)

        # Convert inputs to numeric types as needed
        # Assuming all numeric and categorical features are represented as numbers (0/1 etc.) in the form
        input_array = np.array(input_data, dtype=float).reshape(1, -1)

        # Scale only numeric columns
        # Find indexes of numeric columns in feature_order
        num_indexes = [feature_order.index(col) for col in num_cols]

        # Copy to avoid changing original
        input_scaled = input_array.copy()
        input_scaled[:, num_indexes] = scaler.transform(input_scaled[:, num_indexes])

        # Model prediction
        pred = model.predict(input_scaled)[0]
        
        # Optional: probability can also be used e.g. model.predict_proba(input_scaled)
        
        if pred == 1:
            prediction_text = "Sorry, you MAY have Chronic Kidney Disease. Please consult a doctor."
        else:
            prediction_text = "Great! You DON'T have Chronic Kidney Disease."

        return render_template('result.html', prediction_text=prediction_text)
    except Exception as e:
        # Log error or print for debugging in development
        print("Prediction error:", e)
        return render_template('indexnew.html', error="There was an error processing your input. Please check values and try again.")

if __name__ == "__main__":
    app.run(debug=True)
