import os
import tensorflow as tf
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from keras.models import load_model
from flask_socketio import SocketIO, emit

async_mode = None
app = Flask(__name__)
 
UPLOAD_FOLDER = 'static/uploads/'
 
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
 
#def auc(y_true, y_pred):
#    auc = tf.metrics.auc(y_true, y_pred)[1]
#    keras.backend.get_session().run(tf.local_variables_initializer())
#    return auc

#model = load_model('model/detection_model-ex-063--loss-0029.575.h5', custom_objects={'auc': auc})

socketio = SocketIO(app, async_mode=async_mode)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html', async_mode=socketio.async_mode)
 
@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        flash('Image successfully uploaded')
        return render_template('index.html', filename=filename)
    else:
        flash('Allowed image types are - png, jpg, jpeg')
        return redirect(request.url)

@app.route('/acne/',methods=['POST'])
def acnedetect():
    execution_path = 'model'
    input_path = 'static/uploads/'
    output_path = 'static/output/'

    detector = CustomObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(os.path.join(execution_path , 'detection_model-ex-063--loss-0029.575.h5'))
    detector.setJsonPath(os.path.join(execution_path , 'detection_config.json'))
    detector.loadModel()
    detections = detector.detectObjectsFromImage(input_image = os.path.join(input_path, 'acne.jpg'), output_image_path = os.path.join(output_path, 'acne.jpg'))

    return render_template('index.html', detections=detections)

@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == "__main__":
    app.run()