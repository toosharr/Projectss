from flask import Flask,request,render_template
import numpy as np
import pandas as pd
import pickle

model = pickle.load(open("C:/Users/Lenovo/OneDrive/Desktop/DRDO PROJECT/model_saved.pkl", 'rb'))

import sklearn

app=Flask(__name__)
@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict",methods=['POST'])
def predict():
    N=int(request.form['Nitrogen'])
    P=int(request.form['Phosporous'])
    K=int(request.form['Potassium'])
    Temp=float(request.form['Temperature'])
    humidity=float(request.form['Humidity'])
    ph=float(request.form['PH'])
    rainfall=float(request.form['Rainfall'])

    feature_list=[N,P,K,Temp,humidity,ph,rainfall]
    single_pred=np.array(feature_list).reshape(1,-1)

    prediction=model.predict(single_pred)

    crop_dict={13:'banana',10:'watermelon',18:'mothbeans',7:'orange',
        3:'jute',16:'blackgram',4:'cotton',8:'apple',19:'pigeonpeas',22:'coffee',
        21:'chickpea',15:'lentil',12:'mango',9:'muskmelon',5:'coconut',6:'papaya',
        2:'maize',20:'kidneybeans',14:'pomegranate',11:'grapes',
        17:'mungbean',1:'rice' }

    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        result="{} is a best crop to be cultivated ".format(crop)
    else:
        result="Sorry are not able to recommend a proper crop for this environment"
    return render_template('index.html',result=result)

#python main
if __name__=="__main__":
    app.run(debug=True)