from flask import Flask,render_template,request,url_for
import pandas as pd
from pycaret.classification import *
import numpy as np
import pickle

app=Flask(__name__)
model = load_model('final_decision')
cols=['Gender','Married','Dependents','Education','Self_Employed','ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History','Property_Area']


@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [x for x in request.form.values()]
    final_features = np.array(int_features)
    dat_useen=pd.DataFrame([final_features],columns=cols)
    prediction = predict_model(model, data=dat_useen)
    prediction = int(prediction.Label[0])
    return render_template("home.html", prediction_text='There will be a default if it is one and no default if it is zero {}'.format(prediction))



if __name__ == '__main__':
    app.run(debug=True)
