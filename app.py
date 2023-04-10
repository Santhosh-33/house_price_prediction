
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)
# load the data
data = pd.read_csv('real estate.csv')
data.drop('X1 transaction date',axis=1)

# Load the pickled linear regression model
with open('lr_model.pkl', 'rb') as f:
    model = pickle.load(f)


# define the routes and views
@app.route('/')
def home():
	return render_template('index.html')

@app.route('/result', methods=['POST'])
def index():
	age = float(request.form['age'])
	distance = float(request.form['distance'])
	stores = float(request.form['stores'])
	latitude = float(request.form['latitude'])
	longitude = float(request.form['longitude'])
	
	x = np.array([[age,distance,stores,latitude,longitude]])
	
	prediction = model.predict(x)
	
	return render_template('result.html', prediction=prediction)

# start the Flask app
if __name__ == '__main__':
	app.run(debug=True)






