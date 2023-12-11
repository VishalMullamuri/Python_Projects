from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np
import os

app = Flask(__name__) 

app.config['UPLOAD_FOLDER'] = 'static/'
model = pickle.load(open("predictor.pkl", "rb"))

@app.route('/')
def home():
    pic1 = os.path.join(app.config['UPLOAD_FOLDER'], 'image.jpg')
    return render_template("index.html", image=pic1)

@app.route('/predict', methods=['GET', 'POST']) 
def predict():    
    Year   = int(request.form["Year"]) 
    Present_Price   = float(request.form["Present_Price"])
    Kms_Driven   = float(request.form["Kms_Driven"])
    Fuel_Type   = float(request.form["Fuel_Type"])
    Seller_Type = float(request.form["Seller_Type"])
    Transmission   = float(request.form["Transmission"])
    Owner   = float(request.form["Owner"])
    
    input_data = (Year, Present_Price, Kms_Driven, Fuel_Type, Seller_Type, Transmission, Owner)
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    prediction = model.predict(input_data_reshaped)[0]
    prediction = round(prediction, 2)
    return str(prediction)

    
if __name__ == '__main__':
    app.run(debug=True) 