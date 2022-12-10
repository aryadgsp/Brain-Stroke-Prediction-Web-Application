from flask import Flask, request, render_template, abort, redirect, url_for, send_file
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import sklearn
import pickle

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/tips')
def tips():
    return render_template('tips.html')

@app.route('/services')
def services():
    return render_template('service.html')

@app.route('/location')
def location():
    return render_template('loc.html')

@app.route('/')
def help():
    return render_template('index.html')

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'GET':
        return render_template('pred.html')
    elif request.method == 'POST':
        
        with open('model.pkl', 'rb') as r:
            model = pickle.load(r)
        
        age = np.log2(float(request.form['age']))
        glukose = np.log2(float(request.form['glukose']))
        weight = int(request.form['weight'])
        height = int(request.form['height'])
        bmi = np.log2(float(weight / (height/100)**2))
        gender = int(request.form['gender'])
        hypertension = int(request.form['hypertension'])
        heart = int(request.form['heart'])
        maried = int(request.form['maried'])
        work = int(request.form['work'])
        residence = int(request.form['residence'])
        smoke = int(request.form['smoke'])
        
        features = np.array((gender, age, hypertension, heart, maried, work, residence, glukose, bmi, smoke))
        datas = np.reshape(features, (1,-1))
        
        isStroke = model.predict(datas)
        
        return render_template('pred.html', gender=gender, age=age, hypertension=hypertension, heart=heart, maried=maried, work=work, residence=residence, glukose=glukose, bmi=bmi, smoke=smoke, pred=isStroke)
        # return render_template('result.html', finalData=isStroke)


if __name__ == "__main__":
    app.run(host='0.0.0.0')