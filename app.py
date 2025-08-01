from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

model = pickle.load(open('ckd_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

num_cols = [
    'age', 'blood_pressure', 'blood_glucose_random', 'blood_urea', 'serum_creatinine',
    'sodium', 'potassium', 'hemoglobin', 'packed_cell_volume',
    'white_blood_cell_count', 'red_blood_cell_count'
]

feature_order = [
    'age', 'blood_pressure', 'specify_gravity', 'albumin', 'sugar', 'red_blood_cells', 
    'pus_cell', 'pus_cell_clumps', 'bacteria', 'blood_glucose_random', 'blood_urea', 
    'serum_creatinine', 'sodium', 'potassium', 'hemoglobin', 'packed_cell_volume', 
    'white_blood_cell_count', 'red_blood_cell_count', 'hypertension', 'diabetes_mellitus', 
    'coronary_artery_disease', 'appetite', 'pedal_edema', 'anemia'
]


@app.route('/')
def home():
    return render_template('indexnew.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = []
        for feature in feature_order:
            val = request.form.get(feature)  
            if val is None:
                return render_template('indexnew.html', error=f"Missing input for {feature}")
            input_data.append(val)

        input_array = np.array(input_data, dtype=float).reshape(1, -1)

        
        num_indexes = [feature_order.index(col) for col in num_cols]

        
        input_scaled = input_array.copy()
        input_scaled[:, num_indexes] = scaler.transform(input_scaled[:, num_indexes])

        
        pred = model.predict(input_scaled)[0]
        
        
        
        if pred == 1:
            prediction_text = "Sorry, you MAY have Chronic Kidney Disease. Please consult a doctor."
        else:
            prediction_text = "Great! You DON'T have Chronic Kidney Disease."

        return render_template('result.html', prediction_text=prediction_text)
    except Exception as e:
        
        print("Prediction error:", e)
        return render_template('indexnew.html', error="There was an error processing your input. Please check values and try again.")

if __name__ == "__main__":
    app.run(debug=True)
