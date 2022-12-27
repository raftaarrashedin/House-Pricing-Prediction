import pickle
#from djangoshortcuts import render
from flask import Flask,request,app,jsonify,render_template
#from matplotlib import Scalar
#from matplotlib import Scalar
import numpy as np

app = Flask(__name__, template_folder='template')

# Load the model
regmodel = pickle.load(open('regmodel.pkl','rb'))
scalar = pickle.load(open('scaling.pkl','rb'))

@app.route('/')
def home() :
    #return "<h1>Hi, this is raftaar</h1>"
    return render_template('home.html')

@app.route('/predict_api',methods = ['POST'])

def predict_api():
    data = request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1)) 
    new_data = scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output = regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])
                      
@app.route('/predict', methods=['POST'])
def predict() :
    data = [float(x) for x in request.form.values()]
    final_input = scalar.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output = regmodel.predict(final_input)[0]
    return render_template("home.html",prediction_text = "The house price prediction is {}".format(output))


if __name__ == "__main__" :
    app.run(debug=True)