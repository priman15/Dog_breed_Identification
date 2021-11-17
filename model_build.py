import tensorflow as tf
import tensorflow_hub as hub
import os
import pickle
import datetime
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from IPython.display import display, Image  
%load_ext tensorboard
IMG_SIZE = 224
BATCH_SIZE = 32

def show_25_images(images, labels):
  plt.figure(figsize=(12, 12))

  for i in range(25):
    ax = plt.subplot(5, 5, i+1)
    plt.imshow(images[i])
    plt.axis("off")
    plt.title(unique_labels[labels[i].argmax()])

def process_image(image_path):
  image = tf.io.read_file(image_path)
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.convert_image_dtype(image, tf.float32)
  image = tf.image.resize(image, size=[IMG_SIZE, IMG_SIZE])
  return image

def train_model():
  model = create_model()
  tensorboard = create_tensorboard_callback()
  model.fit(x=train_data,
            epochs=30,
            validation_data=val_data,
            validation_freq=1,
            callbacks=[tensorboard, early_stopping])
  
  return model

def create_model(input_shape, output_shape, model_url):
  model = tf.keras.Sequential([
    hub.KerasLayer(MODEL_URL),
    tf.keras.layers.Dense(units=OUTPUT_SHAPE,   activation="softmax") 
  ])

 
  model.compile(
      loss=tf.keras.losses.CategoricalCrossentropy(), 
      optimizer=tf.keras.optimizers.Adam(), 
      metrics=["accuracy"] 
  )

  
  model.build(INPUT_SHAPE) 
  
  return model

def unbatchify(data):
    data_as_list = list(data.unbatch().as_numpy_iterator())
    images = [ ]
    true_labels = [ ]
    for image, label in data_as_list:
      images.append(image)
      true_labels.append(get_pred_label(label))
    return images, true_labels

def get_image_label(image_path, label):
  """
  Takes an image file path name and the associated label,
  processes the image and returns a tuple of (image, label).
  """
  image = process_image(image_path)
  return image, label

def get_pred_label(prediction_probabilities):  
  """  Turns an array of prediction probabilities into a label. """
  return unique_labels[np.argmax(prediction_probabilities)]


def plot_pred_conf(prediction_probabilities, labels, n=1):
  pred_prob, true_label = prediction_probabilities[n], labels[n]

  pred_label = get_pred_label(pred_prob)

  top_10_pred_indexes = pred_prob.argsort()[-10:][::-1]
  top_10_pred_values = pred_prob[top_10_pred_indexes]
  top_10_pred_labels = unique_labels[top_10_pred_indexes]

  top_plot = plt.bar(np.arange(len(top_10_pred_labels)), 
                     top_10_pred_values, 
                     color="grey")
  plt.xticks(np.arange(len(top_10_pred_labels)),
             labels=top_10_pred_labels,
             rotation="vertical")

  if np.isin(true_label, top_10_pred_labels):
    top_plot[np.argmax(top_10_pred_labels == true_label)].set_color("green")
  else:
    pass

def create_tensorboard_callback():
  logdir = os.path.join("/content/drive/My Drive/Colab Notebooks", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
  return tf.keras.callbacks.TensorBoard(logdir)
  
print("TF version is : ", tf.__version__)
print("TF Hub version is : ", hub.__version__)

def plot_predicted_truth_labels(image_list, true_labels_list, predictions, index):
    true_label = true_labels_list[index]
    predicted_label = get_pred_label(predictions[index])
    prediction_probabilty = np.max(predictions[index]) 
    plt.imshow(image_list[index])
    plt.xticks([])
    plt.yticks([])
    color = 'green' if true_label == predicted_label else 'red'
    plt.title("{} / {} / {}".format(true_label,   
                                    predicted_label, 
                                    prediction_probabilty),
                                    color=color)
    
def create_data_batches(x, y=None, batch_size=BATCH_SIZE, valid_data=False, test_data=False):
  if test_data:
    data = tf.data.Dataset.from_tensor_slices((tf.constant(x)))
    data = data.map(process_image)
    data_batch = data.batch(BATCH_SIZE)
    return data_batch
  
  elif valid_data:
    data = tf.data.Dataset.from_tensor_slices((tf.constant(x), tf.constant(y)))
    data = data.map(get_image_label)
    data_batch = data.batch(BATCH_SIZE)
    return data_batch

  else:
    data = tf.data.Dataset.from_tensor_slices((tf.constant(x), tf.constant(y)))
    data = data.shuffle(buffer_size=len(x))
    data = data.map(get_image_label)
    data_batch = data.batch(BATCH_SIZE)
  return data_batch

if __name__=='__main__':
  if tf.config.list_physical_devices("GPU"):
    print('GPU Used')

  labels_csv = pd.read_csv('/content/labels.csv')
  filenames = ['/content/train/'+file+'.jpg' for file in labels_csv.id]

  string_labels = np.array(labels_csv.breed)

  unique_labels = np.unique(string_labels)

  string_labels[0] == unique_labels

  boolean_labels = [label == np.array(unique_labels) for label in string_labels]

  X = filenames
  y = boolean_labels
  NUM_IMAGES = 1000
  NUM_IMAGES
  X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)


  train_data = create_data_batches(X_train, y_train)
  val_data = create_data_batches(X_valid, y_valid, valid_data=True)
  train_images, train_labels = next(train_data.as_numpy_iterator())
  show_25_images(train_images, train_labels)
  INPUT_SHAPE = [None, IMG_SIZE, IMG_SIZE, 3] 

  OUTPUT_SHAPE = len(unique_labels) 

  MODEL_URL = "https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/4"
  model = create_model(INPUT_SHAPE,OUTPUT_SHAPE,MODEL_URL)
  model.summary()
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy",
                                                  patience=3)
  predictions  = model.predict(val_data)
  index = 0

  images, true_labels = unbatchify(val_data)
  plot_predicted_truth_labels(images, true_labels, predictions, 7)
  print(unique_labels)
  plot_pred_conf(prediction_probabilities=predictions, labels=true_labels,   n=9)
  model_path = "/content/drive/My Drive/Colab Notebooks/mobilnetV2-9000-images1.h5" 
  print(f"Saving model to: {model_path}...")
  model.save(model_path)