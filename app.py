import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]
    
    features_name = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
       'Credit_History', 'Property_Area_Rural', 'Dependents_0',
       'Property_Area_Rural', 'Property_Area_Semiurban', 'Property_Area_Urban']
    
    df = pd.DataFrame(features_value, columns=features_name)
    output = model.predict(df)
        
    if output == 0:
        res_val = "Loan Rejected"
    else:
        res_val = "Loan Accepted"
        

    return render_template('index.html', prediction_text='Application has {}'.format(res_val))

if __name__ == "__main__":
    app.run()
