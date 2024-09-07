import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
data=pd.read_csv("Crop_recommendation.csv")
#X=data.drop('crop_num',axis=1)
crop_dict={'banana':13,
'watermelon':10,
'mothbeans':18,
'orange':7,
'jute':3,
'blackgram':16,
'cotton':4,
'apple':8,
'pigeonpeas':19,
'coffee':22,
'chickpea':21,
'lentil':15,
'mango':12,
'muskmelon':9,
'coconut':5,
'papaya':6,
'maize':2,
'kidneybeans':20,
'pomegranate':14,
'grapes':11,
'mungbean':17,
'rice':1 }

# mapping the crop name to the crop number
data['crop_num']=data['label'].map(crop_dict)
data.drop('label',axis=1,inplace=True)
x = data.drop('crop_num',axis=1)
print(x)

y=data['crop_num']
print(y)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
rfc = RandomForestClassifier()
model=rfc.fit(X_train,y_train)
import pickle
with open("model_saved.pkl", 'wb') as file:
    pickle.dump(model, file)
