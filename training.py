import numpy as np
from tensorflow.keras.backend import gradients
import keras
from keras import layers
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.image as mpimg
from IPython.display import Image
#import IPython
from tensorflow import data as tf_data
import tensorflow as tf
import os
from PIL import Image,ImageOps 
from keras import backend as K
from tensorflow.python.framework.ops import disable_eager_execution
import tensorflow.compat.v1 as tf1
from tensorflow.python.framework import ops
#from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input, decode_predictions
#from tensorflow.keras.applications.efficientnet_v2 import preprocess_input, decode_predictions
#from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input, decode_predictions
#from tensorflow.keras.applications.nasnet import preprocess_input, decode_predictions
#import cv2
from keras.layers import Lambda
from tensorflow.python.framework.ops import disable_eager_execution
import random
import datetime
from matplotlib import colormaps
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from sklearn.metrics import ConfusionMatrixDisplay
import tensorflow_io as tfio
import argparse
import csv
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import mixed_precision
#import keras as k

aPath = '--xla_gpu_cuda_data_dir=/usr/lib/cuda'
os.environ['XLA_FLAGS'] = aPath
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#mixed_precision.set_global_policy('mixed_float16')

parser = argparse.ArgumentParser() 
#group = parser.add_mutually_exclusive_group(required=True)
group=parser
group.add_argument('-sz','--size_img',type=int,help = 'Tailles des images passées au modèle')
group.add_argument('-b','--batch',type=int,help = 'Taille du batch (min:10)')
group.add_argument('-cm','--color_mode',type=str,help = 'Définie le mode de représentation des images (rgb/grayscale)')
group.add_argument('-d','--dropout',type=float,help = 'Valeur du Dropout (entre 0 et 1)')
group.add_argument('-bl','--base_learning_rate',type=float,help = 'Taux apprentissage de base')
group.add_argument('-ie','--initial_epochs',type=int,help = 'Nombres inital itérations')
group.add_argument('-fe','--fine_epochs',type=int,help = 'Nombres finition itérations')
group.add_argument('-fl','--freeze_layer',type=int,help = 'Nombre de couches à geler')
group.add_argument('-wa','--weight_archi',type=str,help = 'Poid à utiliser avec architecture (imagenet, None ou un chemin de fichier)')
group.add_argument('-vs','--valid_split',type=float,help = 'Pourcentage de la base de données entrainement à utiliser pour la validation (entre 0 et 1)')
group.add_argument('-ts','--test_split',type=int,help = 'Pourcentage de la base de données validation à utiliser pour la base de données test (entre 0 et 100), pointe vers un dossier test dans --data_path si égal à 0')
group.add_argument('-dp','--data_path',type=str,help = 'Chemin pour la base de donnée (le chemin doit contenir /train/fichier_séparé_dans_des_dossiers_de_chaque_classes)')
group.add_argument('-sp','--save_path',type=str,help = 'Chemin pour sauvegarder')
group.add_argument('-sn','--save_filename',type=str,help = 'Nom du fichier de sauvegarde')
group.add_argument('-sh','--show_plot',action='store_true',help = 'Montre les figures à la fin')
args = parser.parse_args()
#parser.print_help()

IMG_HW=args.size_img
BATCH_SIZE = args.batch
NB_CHANNEL_IMG=3
COLOR_MODE_IMG=args.color_mode
IMG_SIZE = (IMG_HW, IMG_HW)
DROPOUT_VAL=args.dropout
BASE_LEARNIG_RATE=args.base_learning_rate
FINE_TUNE_LEARNING_RATE=BASE_LEARNIG_RATE/10
WEIGHTS_ARCHI=args.weight_archi
NB_FREEZED_LAYERS=args.freeze_layer
FINE_TUNE_EPOCHS = args.fine_epochs
INITIAL_EPOCHS = args.initial_epochs
VALIDATION_SPLIT=args.valid_split
TEST_SPLIT_FROM_VALIDATION=args.test_split
PATH=args.data_path
SAVE_PATH=args.save_path
CHECKPOINT_FILENAME=args.save_filename
CHECKPOINT_PATH = os.path.join(SAVE_PATH,CHECKPOINT_FILENAME)
SEED=5465189


try:
    os.mkdir(SAVE_PATH)
except OSError as error:
    print("Path already existing: ", error)   

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=CHECKPOINT_PATH,
                                                 save_weights_only=True,
                                                 monitor='val_categorical_accuracy',
                                                 save_best_only=True,
                                                 mode='max',
                                                 verbose=1)

#if not os.path.exists(os.path.join(SAVE_PATH,'logs')):
#                       os.makedirs(os.path.join(SAVE_PATH,'logs'))

tb_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(SAVE_PATH,'logs'), update_freq=1, histogram_freq=1)

'''
class LRRecorder(Callback):
    """Record current learning rate. """
    def on_epoch_begin(self, epoch, logs=None):
        lr = self.model.optimizer.lr
        print("The current learning rate is {}".format(lr.numpy()))
'''

train_dir =os.path.join(PATH,'train')
test_dataset=0



train_dataset, validation_dataset = tf.keras.utils.image_dataset_from_directory(train_dir,
                                                            shuffle=True,
                                                            batch_size=BATCH_SIZE,
                                                            image_size=IMG_SIZE,
                                                            label_mode="categorical",
                                                            validation_split=VALIDATION_SPLIT,
                                                            subset="both",
                                                            seed=SEED,
                                                            color_mode='rgb')
                                                            #pad_to_aspect_ratio=True)

if TEST_SPLIT_FROM_VALIDATION == 0:
  test_dir = os.path.join(PATH, 'test')
  test_dataset = tf.keras.utils.image_dataset_from_directory(test_dir,
                                                                 shuffle=True,
                                                                 batch_size=BATCH_SIZE,
                                                                 image_size=IMG_SIZE,
                                                                 label_mode="categorical",
                                                                # pad_to_aspect_ratio=True,
                                                                 color_mode='rgb',
                                                                 seed=SEED)
                                                               

class_names = train_dataset.class_names
if BATCH_SIZE >= 10:
  plt.figure(figsize=(10, 10))
  for images, labels in train_dataset.take(1):
    for i in range(9):
      ax = plt.subplot(3, 3, i + 1)
      plt.imshow(images[i].numpy().astype("uint8"))
      #plt.title(class_names[labels[i]])
      plt.axis("off")
  plt.savefig(os.path.join(SAVE_PATH,"example-"+str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))+".jpg"))
val_batches = tf.data.experimental.cardinality(validation_dataset)
if test_dataset == 0:
  test_dataset = validation_dataset.take(val_batches // TEST_SPLIT_FROM_VALIDATION)
  validation_dataset = validation_dataset.skip(val_batches // TEST_SPLIT_FROM_VALIDATION)
print('Number of training batches: %d' % tf.data.experimental.cardinality(train_dataset))
print('Number of validation batches: %d' % tf.data.experimental.cardinality(validation_dataset))
print('Number of test batches: %d' % tf.data.experimental.cardinality(test_dataset))
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)
layer=[tf.keras.layers.RandomFlip('horizontal'),
  tf.keras.layers.RandomRotation(0.2),
  tf.keras.layers.RandomContrast(0.2),
  tf.keras.layers.RandomBrightness(0.2)]
  #tf.keras.layers.RandomCrop ( height=int(IMG_HW/2), width=int(IMG_HW/2),seed=85),
  #tf.keras.layers.Resizing(height=IMG_HW, width=IMG_HW)]
if COLOR_MODE_IMG == 'grayscale':
   layer.append(tf.keras.layers.Lambda(lambda x: tf.repeat(tf.reduce_sum(x*tf.constant([0.21, 0.72, 0.07]), axis=-1, keepdims=True),3,axis=-1)))
data_augmentation = tf.keras.Sequential(
  layer
)
if BATCH_SIZE >= 10:
  plt.figure(figsize=(10, 10))
  for image, _ in train_dataset.take(1):
    first_image = image[0]
    for i in range(9):
      ax = plt.subplot(3, 3, i + 1)
      augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
      plt.imshow(augmented_image[0] / 255)
      plt.axis('off')
  plt.savefig(os.path.join(SAVE_PATH,"example_augmented-"+str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))+".jpg"))
#preprocess_input = tf.keras.applications.efficientnet.preprocess_input
# Create the base model from the pre-trained model MobileNet V2
IMG_SHAPE = IMG_SIZE + (NB_CHANNEL_IMG,)
base_model = tf.keras.applications.MobileNetV3Large(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights=WEIGHTS_ARCHI)

image_batch, label_batch = next(iter(train_dataset))
feature_batch = base_model(image_batch)
print(feature_batch.shape)
base_model.trainable = False
#base_model.summary()
global_average_layer = tf.keras.layers.GlobalMaxPool2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)
prediction_layer = tf.keras.layers.Dense(len(class_names),activation='softmax')
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)
inputs = tf.keras.Input(shape=(IMG_HW, IMG_HW, NB_CHANNEL_IMG))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(DROPOUT_VAL)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)
base_learning_rate = BASE_LEARNIG_RATE 
METRICS=['categorical_accuracy', 
         tf.keras.metrics.AUC(name='auc',multi_label=False),
         tf.keras.metrics.TopKCategoricalAccuracy(name="top_3_categorical_accuracy",k=3)]
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=base_learning_rate,weight_decay=False),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
              metrics=METRICS)
model.summary(show_trainable=True)
len(model.trainable_variables)

try:
  print("Dans le statement try")
  #ckpt = tf.train.Checkpoint(step=tf.Variable(1),model=model,optimizer=optimizer)
  #manager = tf.train.CheckpointManager(ckpt, './tf_ckpts', max_to_keep=3)
  model.load_weights(CHECKPOINT_PATH)
except FileNotFoundError:
  print("Pas de fichier de sauvegarde présent")
try:
 #  validation_dataset=tf.keras.utils.to_categorical(validation_dataset,3,dtype=int)
  loss0, accuracy0, auc0, top_3_categorical_accuracy0 = model.evaluate(validation_dataset)
  print("initial loss: {:.2f}".format(loss0))
  print("initial accuracy: {:.2f}".format(accuracy0))
  print("initial auc: {:.2f}".format(auc0))
  print("initial top_3: {:.2f}".format(top_3_categorical_accuracy0))
  history = model.fit(train_dataset,
                      epochs=INITIAL_EPOCHS,
                      validation_data=validation_dataset,
                      callbacks=[cp_callback,tb_callback],
                      verbose=1)
  acc = history.history['categorical_accuracy']
  val_acc = history.history['val_categorical_accuracy']

  auc= history.history['auc']
  val_auc= history.history['val_auc']

  loss = history.history['loss']
  val_loss = history.history['val_loss']

  top_3=history.history['top_3_categorical_accuracy']

  plt.figure(figsize=(7.5, 7.5))
  plt.plot(auc, label='Training AUC-ROC')
  plt.plot(val_auc, label='Validation AUC-ROC')
  plt.legend(loc='lower right')
  plt.ylabel('AUC-ROC')
  plt.ylim([min(plt.ylim()),1])
  plt.title('Training and Validation AUC-ROC')
  plt.savefig(os.path.join(SAVE_PATH,"base_auc_curve-"+str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))+".jpg"))

  plt.figure(figsize=(8, 8))
  plt.subplot(2, 1, 1)
  plt.plot(acc, label='Training Accuracy')
  plt.plot(val_acc, label='Validation Accuracy')
  plt.legend(loc='lower right')
  plt.ylabel('Accuracy')
  plt.ylim([min(plt.ylim()),1])
  plt.title('Training and Validation Accuracy')
  plt.subplot(2, 1, 2)
  plt.plot(loss, label='Training Loss')
  plt.plot(val_loss, label='Validation Loss')
  plt.legend(loc='upper right')
  plt.ylabel('Cross Entropy')
  plt.ylim([0,3.0])
  plt.title('Training and Validation Loss')
  plt.xlabel('epoch')
  plt.savefig(os.path.join(SAVE_PATH,"base_training_curve-"+str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))+".jpg"))
  path_csv=os.path.join(SAVE_PATH,str(CHECKPOINT_FILENAME)+"_history_base"+str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))+".csv")
  with open(path_csv, 'w', newline='') as file:
    writer = csv.writer(file, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    #print(history.history)
    writer.writerows(["Accuracy",])
    writer.writerows(map(lambda x: [x], acc))
    writer.writerows(["Validdation_Accuracy",])
    writer.writerows(map(lambda x: [x], val_acc))
    writer.writerows(["Loss",])
    writer.writerows(map(lambda x: [x], loss))
    writer.writerows(["Validation_Loss",])
    writer.writerows(map(lambda x: [x], val_loss))
    writer.writerows(["Auc_ROC",])
    writer.writerows(map(lambda x: [x], auc))
    writer.writerows(["Validation_Auc_ROC",])
    writer.writerows(map(lambda x: [x], val_auc))
    writer.writerows(["Top_3_Accuracy",])
    writer.writerows(map(lambda x: [x], top_3))
except KeyboardInterrupt:
  print("Skip by user")

base_model.trainable = True
print("Number of layers in the base model: ", len(base_model.layers))

# Fine-tune from this layer onwards
fine_tune_at = NB_FREEZED_LAYERS

print("Number of freezed layers: ", fine_tune_at)

# Freeze all the layers before the `fine_tune_at` layer
if fine_tune_at != 0:
  for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
              optimizer = tf.keras.optimizers.RMSprop(learning_rate=FINE_TUNE_LEARNING_RATE,weight_decay=False),
              metrics=METRICS)
model.summary(show_trainable=True)
len(model.trainable_variables)
total_epochs =  INITIAL_EPOCHS + FINE_TUNE_EPOCHS
try:
  print("Dans le statement try")
  model.load_weights(CHECKPOINT_PATH)
except FileNotFoundError:
  print("Pas de fichier de sauvegarde présent")

try:
  history_fine = model.fit(train_dataset,
                           epochs=total_epochs,
                           initial_epoch=INITIAL_EPOCHS,
                           validation_data=validation_dataset,
                           callbacks=[cp_callback,tb_callback])
  acc += history_fine.history['categorical_accuracy']
  val_acc += history_fine.history['val_categorical_accuracy']
  loss += history_fine.history['loss']
  val_loss += history_fine.history['val_loss']
  auc += history_fine.history['auc']
  val_auc += history_fine.history['val_auc']
  top_3 += history_fine.history['top_3_categorical_accuracy']

  plt.figure(figsize=(7.5, 7.5))
  plt.plot(auc, label='Training AUC-ROC')
  plt.plot(val_auc, label='Validation AUC-ROC')
  plt.ylim([0, 1])
  plt.plot([INITIAL_EPOCHS-1,INITIAL_EPOCHS-1],
            plt.ylim(), label='Start Fine Tuning')
  plt.ylim([min(plt.ylim()),1])
  plt.legend(loc='lower right')
  plt.title('Training and Validation AUC-ROC')
  plt.savefig(os.path.join(SAVE_PATH,"fine_tunning_auc_curve-"+str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))+".jpg"))

  plt.figure(figsize=(8, 8))
  plt.subplot(2, 1, 1)
  plt.plot(acc, label='Training Accuracy')
  plt.plot(val_acc, label='Validation Accuracy')
  plt.ylim([0, 1])
  plt.plot([INITIAL_EPOCHS-1,INITIAL_EPOCHS-1],
            plt.ylim(), label='Start Fine Tuning')
  plt.legend(loc='lower right')
  plt.title('Training and Validation Accuracy')
  plt.subplot(2, 1, 2)
  plt.plot(loss, label='Training Loss')
  plt.plot(val_loss, label='Validation Loss')
  plt.ylim([0, 3.0])
  plt.plot([INITIAL_EPOCHS-1,INITIAL_EPOCHS-1],
           plt.ylim(), label='Start Fine Tuning')
  plt.legend(loc='upper right')
  plt.title('Training and Validation Loss')
  plt.xlabel('epoch')
  plt.savefig(os.path.join(SAVE_PATH,"fine_tunning_curve-"+str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))+".jpg"))
  path_csv=os.path.join(SAVE_PATH,str(CHECKPOINT_FILENAME)+"_history_fine"+str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))+".csv")
  with open(path_csv, 'w', newline='') as file:
    writer = csv.writer(file, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    #print(history.history)
    writer.writerows(["Accuracy",])
    writer.writerows(map(lambda x: [x], acc))
    writer.writerows(["Validdation_Accuracy",])
    writer.writerows(map(lambda x: [x], val_acc))
    writer.writerows(["Loss",])
    writer.writerows(map(lambda x: [x], loss))
    writer.writerows(["Validation_Loss",])
    writer.writerows(map(lambda x: [x], val_loss))
    writer.writerows(["Auc_ROC",])
    writer.writerows(map(lambda x: [x], auc))
    writer.writerows(["Validation_Auc_ROC",])
    writer.writerows(map(lambda x: [x], val_auc))
    writer.writerows(["Top_3_Accuracy",])
    writer.writerows(map(lambda x: [x], top_3))
  #plt.show()
except KeyboardInterrupt:
  print("Skip by user")

loss, accuracy, auc, top_3 = model.evaluate(test_dataset)
print('Test accuracy :', accuracy)
print('Test loss :', loss)
print('Test auc :', auc)
print('Top 3 accuracy:',top_3)
image_batch, label_batch = test_dataset.as_numpy_iterator().next()
predictions = model.predict_on_batch(image_batch).flatten()
label_names=[]
for i in label_batch:
   e=0
   for j in i:
      if j == 1:
         label_names.append(class_names[e])
      e=e+1
   e=0
label_batch=label_names
# Apply a sigmoid since our model returns logits
predictions = tf.nn.sigmoid(predictions)
#predictions = tf.where(predictions < 0.5, 0, 1)
div=len(class_names)
div_pred=int(len(predictions)/div)
e=0
pred_true=0.0
label_predict=[]
for i in range(div_pred):
   for j in range(div):
      it=(len(class_names)*i)+j
      pred=predictions[it].numpy()
      if float(pred_true) < pred.item(0):
         pred_lab=class_names[j]
         pred_true=pred.item(0)
   pred_true=0.0
   label_predict.append(pred_lab)
   e=e+1
predictions=label_predict
# fonction to use in future :tf.math.top_k 
print('Predictions:\n', predictions)
print('Labels:\n', label_batch)

plt.figure(figsize=(10, 10))
for i in range(9):
  ax = plt.subplot(3, 3, i + 1)
  plt.imshow(image_batch[i].astype("uint8"))
  plt.title(predictions[i])
  plt.axis("off")
plt.savefig(os.path.join(SAVE_PATH,"predictions-"+str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))+".jpg"))
#plt.show()

if args.show_plot == True:
  plt.show()
