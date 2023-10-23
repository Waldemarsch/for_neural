from flask import Flask, render_template, request, session
import os
from werkzeug.utils import secure_filename
from utils import load_model, predict_image
from model import UNet
from PIL import Image

# WSGI Application
# Defining upload folder path
UPLOAD_FOLDER = os.path.join('static', 'uploads')
# # Define allowed files
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tif'}

app = Flask(__name__)

# Configure upload folder for Flask application
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define secret key to enable session
app.secret_key = 'This is your secret key to utilize session in Flask'

@app.route('/')
def index():
    return render_template('index_upload_and_display_image.html')

@app.route('/',  methods=("POST", "GET"))
def upload_file():
    if request.method == 'POST':
        # Upload file flask
        uploaded_img = request.files['uploaded-file']
        # Extracting uploaded data file name
        img_filename = secure_filename(uploaded_img.filename)
        path_to_save = os.path.join(app.config['UPLOAD_FOLDER'], img_filename)
        # Creating directory
        if not os.path.exists(app.config['UPLOAD_FOLDER']):       
            os.makedirs(app.config['UPLOAD_FOLDER'])
        # Upload file to database (defined uploaded folder in static path)
        if not os.path.exists(path_to_save):
            uploaded_img.save(path_to_save)
        # Storing uploaded file path in flask session
        session['uploaded_img_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], img_filename)

        return render_template('index_upload_and_display_image_page2.html', user_image=session['uploaded_img_file_path'])

@app.route('/make_mask', methods=['POST'])
def make_mask():
    # Retrieving uploaded file path from session
    model = load_model(UNet, './models/unet_new')
    img_file_path = session.get('uploaded_img_file_path', None)
    mask_path = img_file_path.split('.')[0] + '_mask' + '.' + img_file_path.split('.')[1] 
    mask = predict_image(model=model, image_path=img_file_path)
    mask_img = Image.fromarray(mask)
    path_to_save = mask_path
    print(mask_path)
    session['mask_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], path_to_save)
    mask_img.save(path_to_save)
    
    # Display image in Flask application web page
    return render_template('index_show_image_and_mask.html', user_image=img_file_path, mask_path=mask_path)

if __name__ == '__main__':
    app.run(debug=True)