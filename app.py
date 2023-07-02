from flask import Flask, request, jsonify, render_template , Response ,render_template_string
import time
import glob 
import os
from flask_cors import CORS, cross_origin
from alzheimer_disease.utils.main_utils import decodeImage
from alzheimer_disease.pipeline.predict_pipeline import PredictPipeline
from alzheimer_disease.exception import AlzException
import subprocess
import webbrowser

import json


os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)


class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = PredictPipeline(self.filename)


@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')


@app.route("/train", methods=['GET','POST'])
@cross_origin()
def trainRoute():
    if request.method == 'GET':
        return render_template('train.html',training_result=False)
    elif request.method == 'POST':

        try : 
            activation = str(request.form.get('activation'))
            optimizer = str(request.form.get('optimizer'))
            batch_size = int(request.form.get('batch_size'))
            dropout_rate = float(request.form.get('dropout_rate'))
            epochs = int(request.form.get('epochs'))
            use_early_stopping = request.form.get('use_early_stopping')
            load_weights = request.form.get('load_weights')
            use_lr_scheduler = request.form.get('use_lr_scheduler')

            command = f"python main.py {activation} {optimizer} {batch_size} {dropout_rate} {epochs} {use_early_stopping} {load_weights} {use_lr_scheduler}"

            
          
            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            exit_code = process.returncode

            if exit_code != 0:
                error_message = error_message = "An error occurred while training the model: " #+ stderr.decode('utf-8').strip().split('\n')[-1]
                return render_template('error_hyperparameter.html', error_message=error_message)

            # Load the training result from the JSON file
            with open(r'D:\Projects\Alzeimer\Alzheimer-s-Disease-Detection\artifacts\model_trainer\evaluation_results.json', 'r') as f:
                training_result = json.load(f)

                print("training result : ",training_result)

            return render_template('training_result.html', training_result=training_result)
        except Exception as e:
            error_message = str(e)
            return render_template('error_hyperparameter.html', error_message=error_message)




@app.route("/predict", methods=['GET','POST'])
@cross_origin()
def predictRoute():
      if request.method == 'GET':
        return render_template('predict.html',training_result=False)
      else:
        image = request.json['image']
        decodeImage(image, clApp.filename)
        result = clApp.classifier.predict()
        return jsonify(result)

@app.route('/mlflow-ui')
@cross_origin()
def open_mlflow_ui():
    subprocess.Popen(['mlflow', 'ui'])

    # Open MLflow UI in a new tab of the web browser
    webbrowser.open_new_tab('http://localhost:5000')  # Change the URL if needed

    return render_template('mlflow_ui.html')

if __name__ == "__main__":
    clApp = ClientApp()
    app.run(host='0.0.0.0', port=8080,debug=True) #local host
   # app.run(host='0.0.0.0', port=80,debug=True) #for AZURE


   