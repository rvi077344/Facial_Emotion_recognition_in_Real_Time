#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install opencv-contrib-python imutils


# In[2]:


#import all the dependencies
import csv
import glob
import pathlib

import cv2
import imutils
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import *
from tensorflow.keras.utils import to_categorical


# In[3]:


#Define a list of all possible emotions in dataset along with a color associated with each one:

EMOTIONS = ['angry', 'scared', 'happy', 'sad', 'surprised', 'neutral']


# In[4]:


#define color with each emotion
COLORS = {
    'angry': (0, 0, 255),
    'scared': (0, 128, 255),
    'happy': (0, 255, 255),
    'sad': (255, 0, 0),
    'surprised': (178, 255, 102),
    'neutral': (160, 160, 160)
}


# In[5]:


def build_network(input_shape, classes):
    input = Input(shape=input_shape)
    x = Conv2D(filters=32,
               kernel_size=(3, 3),
               padding='same',
               kernel_initializer='he_normal')(input)
    
    x = ELU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(filters=32,
               kernel_size=(3, 3),
               kernel_initializer='he_normal',
               padding='same')(x)
    x = ELU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(rate=0.25)(x)

    x = Conv2D(filters=64,
               kernel_size=(3, 3),
               kernel_initializer='he_normal',
               padding='same')(x)
    x = ELU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(filters=64,
               kernel_size=(3, 3),
               kernel_initializer='he_normal',
               padding='same')(x)
    x = ELU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(rate=0.25)(x)

    x = Conv2D(filters=128,
               kernel_size=(3, 3),
               kernel_initializer='he_normal',
               padding='same')(x)
    x = ELU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(filters=128,
               kernel_size=(3, 3),
               kernel_initializer='he_normal',
               padding='same')(x)
    x = ELU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(rate=0.25)(x)

    x = Flatten()(x)
    x = Dense(units=64,
              kernel_initializer='he_normal')(x)
    x = ELU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = Dropout(rate=0.5)(x)

    x = Dense(units=64,
              kernel_initializer='he_normal')(x)
    x = ELU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = Dropout(rate=0.5)(x)

    x = Dense(units=classes,
              kernel_initializer='he_normal')(x)
    output = Softmax()(x)

    return Model(input, output)


# In[6]:


def load_dataset(dataset_path, classes):
    train_images = []
    train_labels = []
    val_images = []
    val_labels = []
    test_images = []
    test_labels = []

    with open(dataset_path, 'r') as f:
        reader = csv.DictReader(f)
        for line in reader:
            label = int(line['emotion'])

            if label <= 1:
                label = 0  # This merges classes 1 and 0.

            if label > 0:
                label -= 1  # All classes start from 0.

            image = np.array(line['pixels'].split(' '),
                             dtype='uint8')
            image = image.reshape((48, 48))
            image = img_to_array(image)

            if line['Usage'] == 'Training':
                train_images.append(image)
                train_labels.append(label)
            elif line['Usage'] == 'PrivateTest':
                val_images.append(image)
                val_labels.append(label)
            else:
                test_images.append(image)
                test_labels.append(label)

    train_images = np.array(train_images)
    val_images = np.array(val_images)
    test_images = np.array(test_images)

    train_labels = to_categorical(np.array(train_labels),
                                  classes)
    val_labels = to_categorical(np.array(val_labels), classes)
    test_labels = to_categorical(np.array(test_labels),
                                 classes)

    return (train_images, train_labels),            (val_images, val_labels),            (test_images, test_labels)


# In[7]:


def rectangle_area(r):
    return (r[2] - r[0]) * (r[3] - r[1])


def plot_emotion(emotions_plot, emotion, probability, index):
    w = int(probability * emotions_plot.shape[1])
    cv2.rectangle(emotions_plot,
                  (5, (index * 35) + 5),
                  (w, (index * 35) + 35),
                  color=COLORS[emotion],
                  thickness=-1)

    white = (255, 255, 255)
    text = f'{emotion}: {probability * 100:.2f}%'
    cv2.putText(emotions_plot,
                text,
                (10, (index * 35) + 23),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=0.45,
                color=white,
                thickness=2)

    return emotions_plot


# In[8]:


def plot_face(image, emotion, detection):
    frame_x, frame_y, frame_width, frame_height = detection
    cv2.rectangle(image,
                  (frame_x, frame_y),
                  (frame_x + frame_width,
                   frame_y + frame_height),
                  color=COLORS[emotion],
                  thickness=2)
    cv2.putText(image,
                emotion,
                (frame_x, frame_y - 10),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=0.45,
                color=COLORS[emotion],
                thickness=2)

    return image


# In[9]:



def predict_emotion(model, roi):
    roi = cv2.resize(roi, (48, 48))
    roi = roi.astype('float') / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=0)

    predictions = model.predict(roi)[0]
    return predictions


# In[10]:


import pandas as pd
path = 'fer2013.csv'
data = pd.read_csv(path)
data.head(10)


# In[11]:


classes = len(EMOTIONS)

(train_images, train_labels), (val_images, val_labels), (test_images, test_labels) = load_dataset(path,classes)


# In[12]:


model = build_network((48, 48, 1), classes)
model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.003),metrics=['accuracy'])
checkpoint_pattern = ('model-ep{epoch:03d}-loss{loss:.3f}'
                          '-val_loss{val_loss:.3f}.h5')
checkpoint = ModelCheckpoint(checkpoint_pattern,monitor='val_loss',verbose=1,save_best_only=True,mode='min')


# In[13]:


BATCH_SIZE = 128
train_augmenter = ImageDataGenerator(rotation_range=10,zoom_range=0.1,
                              horizontal_flip=True,
                                    rescale=1. / 255.,
                                fill_mode='nearest')
train_gen = train_augmenter.flow(train_images,
                                     train_labels,
                                 batch_size=BATCH_SIZE)
train_steps = len(train_images) // BATCH_SIZE
val_augmenter = ImageDataGenerator(rescale=1. / 255.)
val_gen = val_augmenter.flow(val_images,val_labels,
                         batch_size=BATCH_SIZE)


# In[14]:


EPOCHS = 9
model.fit(train_gen,steps_per_epoch=train_steps,
              validation_data=val_gen,
              epochs=EPOCHS,
              verbose=1,
              callbacks=[checkpoint])
test_augmenter = ImageDataGenerator(rescale=1. / 255.)
test_gen = test_augmenter.flow(test_images,
                                   test_labels,
                                   batch_size=BATCH_SIZE)
test_steps = len(test_images)
BATCH_SIZE_, accuracy = model.evaluate(test_gen,steps=test_steps)

print(f'Accuracy: {accuracy * 100}%')


# In[16]:


from keras.preprocessing.image import img_to_array
from keras.models import load_model
import imutils
import sys
import cv2

detection_model_path = 'haarcascade_frontalface_default.xml'
emotion_recognition_model_path =  '_mini_xception.100_0.65.hdf5'
image_path = 'women smiling.jpg'

face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_recognition_model_path)
emotions = ['angry', 'disgust', 'scared', 'happy', 'sad', 'surprised', 'neutral']
color_frame = cv2.imread(image_path)
gray_frame = cv2.imread(image_path, 0)

cv2.imshow('Input test image', color_frame)
cv2.waitKey(1000)
cv2.destroyAllWindows()


detected_faces = face_detection.detectMultiScale(color_frame, scaleFactor=1.1, minNeighbors=5, 
                                        minSize=(30,30), flags=cv2.CASCADE_SCALE_IMAGE)
print('Number of faces detected : ', len(detected_faces))

if len(detected_faces)>0:
    
    detected_faces = sorted(detected_faces, reverse=True, key=lambda x: (x[2]-x[0])*(x[3]-x[1]))[0] # if more than one faces
    (fx, fy, fw, fh) = detected_faces
    
    im = gray_frame[fy:fy+fh, fx:fx+fw]
    im = cv2.resize(im, (48,48))  # the model is trained on 48*48 pixel image 
    im = im.astype("float")/255.0
    im = img_to_array(im)
    im = np.expand_dims(im, axis=0)
    
    preds = emotion_classifier.predict(im)[0]
    emotion_probability = np.max(preds)
    label = emotions[preds.argmax()]
    
    cv2.putText(color_frame, label, (fx, fy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    cv2.rectangle(color_frame, (fx, fy), (fx + fw, fy + fh),(0, 0, 255), 2)

cv2.imshow('Input test image', color_frame)
cv2.imwrite('output_'+image_path.split('/')[-1], color_frame)
cv2.waitKey(10000)
cv2.destroyAllWindows()


# In[ ]:


cv2.namedWindow('emotion_recognition')
camera = cv2.VideoCapture(0)  
#camera = cv2.VideoCapture('various_emotions.mp4')  

sz = (int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))

fourcc = cv2.VideoWriter_fourcc(*'mpeg')

out = cv2.VideoWriter()
out.open('output_various_emotions.mp4',fourcc, 15, sz, True) # initialize the writer


# while True: # when reading from a video camera, use this while condition
while(camera.read()[0]):  # when reading from a video file, use this while condition
    color_frame = camera.read()[1]
    color_frame = imutils.resize(color_frame,width=min(700, color_frame.shape[1]))
    
    
    gray_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)
    detected_faces = face_detection.detectMultiScale(gray_frame,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
    
    
    canvas = np.zeros((250, 300, 3), dtype="uint8")
    frameClone = color_frame.copy()    

    
    if len(detected_faces)>0:

        detected_faces = sorted(detected_faces, reverse=True, key=lambda x: (x[2]-x[0])*(x[3]-x[1]))[0] # if more than one faces
        (fx, fy, fw, fh) = detected_faces

        im = gray_frame[fy:fy+fh, fx:fx+fw]
        im = cv2.resize(im, (48,48))  # the model is trained on 48*48 pixel image 
        im = im.astype("float")/255.0
        im = img_to_array(im)
        im = np.expand_dims(im, axis=0)

        preds = emotion_classifier.predict(im)[0]
        emotion_probability = np.max(preds)
        label = emotions[preds.argmax()]

        cv2.putText(color_frame, label, (fx, fy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.rectangle(color_frame, (fx, fy), (fx + fw, fy + fh),(0, 0, 255), 2)

    
    for (i, (emotion, prob)) in enumerate(zip(emotions, preds)):
        # construct the label text
        text = "{}: {:.2f}%".format(emotion, prob * 100)
        w = int(prob * 300)
        
        cv2.rectangle(canvas, (7, (i * 35) + 5), (w, (i * 35) + 35), (0, 50, 100), -1)
        cv2.putText(canvas, text, (10, (i * 35) + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(frameClone, label, (fx, fy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 150, 100), 2)
        cv2.rectangle(frameClone, (fx, fy), (fx + fw, fy + fh), (100, 100, 100), 2)
    
    out.write(frameClone)
    out.write(canvas)
    
    cv2.imshow('emotion_recognition', frameClone)
    cv2.imshow("Probabilities", canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
camera.release()
out.release()
cv2.waitKey(10000)
cv2.destroyAllWindows()


# In[34]:





# In[ ]:




