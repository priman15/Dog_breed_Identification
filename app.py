from flask import Flask,render_template,request,url_for,redirect,flash
#from flask_ngrok import run_with_ngrok
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
import pickle
from PIL import Image
import io
import numpy as np
import tensorflow_hub as hub
app = Flask(__name__)
#run_with_ngrok(app)   

def create_data_batches(x, y=None, batch_size=32, valid_data=False, test_data=False):
  """
  x : array of images filepath
  y : array of images label
  batch_size : size of the batch we want to create
  valid_data, test_data : to specify the type of dataset we want to create

  Creates batches from pairs of image (x) and label (y).
  Shuffles the data if it's training data . Doesn't shuffle it if it's validation data.
  In test data we use only images (no labels)
  """
  # If the data is a test dataset, we don't have labels
  if test_data:
    # Get the slices of an array in the form of tensors, we only pass filepaths
    data = tf.data.Dataset.from_tensor_slices((tf.constant(x)))
    # Preprocess each image object with our 'process_image' function
    data = data.map(process_image)
    # Turn our data into batches
    data_batch = data.batch(32)
    return data_batch

def process_image(image_path):
  """
  Takes an image file path and turns it into a Tensor.
  """
  # Read in image file
  image = tf.io.read_file(image_path)
  # Turn the jpeg image into numerical Tensor with 3 colour channels (Red, Green, Blue)
  image = tf.image.decode_jpeg(image, channels=3)
  # Convert the colour channel values from 0-225 values to 0-1 values
  image = tf.image.convert_image_dtype(image, tf.float32)
  # Resize the image to our desired size (224, 244)
  image = tf.image.resize(image, size=[224, 224])
  return image

UPLOAD_FOLDER = os.path.join(app.root_path ,'static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
@app.route("/")
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_image():
    unique_labels=['affenpinscher', 'afghan_hound', 'african_hunting_dog', 'airedale',
 'american_staffordshire_terrier', 'appenzeller', 'australian_terrier',
 'basenji', 'basset', 'beagle', 'bedlington_terrier', 'bernese_mountain_dog',
 'black-and-tan_coonhound', 'blenheim_spaniel', 'bloodhound', 'bluetick',
 'border_collie', 'border_terrier', 'borzoi', 'boston_bull',
 'bouvier_des_flandres', 'boxer', 'brabancon_griffon', 'briard',
 'brittany_spaniel', 'bull_mastiff', 'cairn', 'cardigan',
 'chesapeake_bay_retriever', 'chihuahua', 'chow', 'clumber', 'cocker_spaniel',
 'collie', 'curly-coated_retriever', 'dandie_dinmont', 'dhole', 'dingo',
 'doberman', 'english_foxhound', 'english_setter', 'english_springer',
 'entlebucher', 'eskimo_dog', 'flat-coated_retriever', 'french_bulldog',
 'german_shepherd', 'german_short-haired_pointer', 'giant_schnauzer',
 'golden_retriever', 'gordon_setter', 'great_dane', 'great_pyrenees',
 'greater_swiss_mountain_dog', 'groenendael', 'ibizan_hound', 'irish_setter',
 'irish_terrier', 'irish_water_spaniel', 'irish_wolfhound',
 'italian_greyhound', 'japanese_spaniel', 'keeshond', 'kelpie',
 'kerry_blue_terrier', 'komondor', 'kuvasz', 'labrador_retriever',
 'lakeland_terrier', 'leonberg', 'lhasa', 'malamute', 'malinois', 'maltese_dog',
 'mexican_hairless', 'miniature_pinscher', 'miniature_poodle',
 'miniature_schnauzer', 'newfoundland', 'norfolk_terrier',
 'norwegian_elkhound', 'norwich_terrier', 'old_english_sheepdog',
 'otterhound', 'papillon', 'pekinese', 'pembroke' ,'pomeranian', 'pug',
 'redbone' ,'rhodesian_ridgeback', 'rottweiler', 'saint_bernard', 'saluki',
 'samoyed', 'schipperke', 'scotch_terrier', 'scottish_deerhound',
 'sealyham_terrier', 'shetland_sheepdog', 'shih-tzu', 'siberian_husky',
 'silky_terrier', 'soft-coated_wheaten_terrier', 'staffordshire_bullterrier',
 'standard_poodle', 'standard_schnauzer', 'sussex_spaniel', 'tibetan_mastiff',
 'tibetan_terrier', 'toy_poodle', 'toy_terrier', 'vizsla', 'walker_hound',
 'weimaraner', 'welsh_springer_spaniel', 'west_highland_white_terrier',
 'whippet', 'wire-haired_fox_terrier' ,'yorkshire_terrier']
    file = request.files['file']
    filename = secure_filename(file.filename)
    mymodel=os.path.join(app.config['UPLOAD_FOLDER'], filename)
    #print("heloooooooooooooo")
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    #flash('Image successfully uploaded and displayed below')
    #with open(os.path.join(app.config['UPLOAD_FOLDER'], filename), 'rb') as f:
        #image = Image.open(io.BytesIO(f.read()))
    global model
    #global graph
	  #graph = tf.get_default_graph()
    model=pickle.load(open("model.pkl","rb"))
    pickle.dump(model, open("/content/model.pkl","wb"))
    model=pickle.load(open("model.pkl","rb"))
    custom_data = create_data_batches([os.path.join(app.config['UPLOAD_FOLDER'], filename)], test_data=True)
    custom_preds = model.predict(custom_data)
    print(np.max(custom_preds[0]))
    lab=unique_labels[np.argmax(custom_preds[0])]
    if(np.max(custom_preds[0])<0.93):
      lab="I don't think you are a doggie."
    return render_template('index.html',name=filename,label=lab)

    
app.run()