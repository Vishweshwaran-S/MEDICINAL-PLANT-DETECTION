import cv2
import os
import tensorflow as tf
import PIL
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


data_dir="C:\leaf dataset"
import pathlib
data_dir=pathlib.Path(data_dir)
list(data_dir.glob('*/*.jpg'))
neem=list(data_dir.glob('neem/*'))
image_dir={
    'eucalyptus':list(data_dir.glob('eucalyptus/*')),
    'neem':list(data_dir.glob('neem/*')),
    'tulsi':list(data_dir.glob('tulsi/*')),
    'mint':list(data_dir.glob('mint/*')),
    'Aloe vera':list(data_dir.glob('Aloe vera/*')),
    'Turmeric':list(data_dir.glob('Turmeric/*'))

}
image_label={
    'eucalyptus':0,
    'neem':1,
    'tulsi':2,
    'mint':3,
    'Aloe vera':4,
    'Turmeric':5
}
x=[]
y=[]
for name,images in image_dir.items():
    for i in images:
        img=cv2.imread(str(i))
        if img is None:
            continue
        reshaped=cv2.resize(img,(220,220))
        x.append(reshaped)
        y.append(image_label[name])
x=np.array(x)
y=np.array(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=15)
x_trainscaled=x_train/255
x_test_scaled=x_test/255
num=6
augmentation=keras.Sequential([
    layers.experimental.preprocessing.RandomFlip('horizontal'),
    layers.experimental.preprocessing.RandomRotation(1.0),
    layers.experimental.preprocessing.RandomZoom(1.0)
])

model=keras.Sequential([
    augmentation,
   layers.Conv2D(16,3 ,padding='same',activation='relu'),
   layers.MaxPooling2D(),
   layers.Conv2D(32,3,padding='same',activation='relu'),
   layers.MaxPooling2D(),
   layers.Conv2D(64,3,padding='same',activation='relu'),
   layers.MaxPooling2D(),
   layers.Flatten(),
   layers.Dense(128,activation='relu'),
   layers.Dense(num)
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.fit(x_trainscaled,y_train,epochs=100)
model.evaluate(x_test_scaled,y_test)
z=model.predict(x_test_scaled)
f=np.argmax(z[36])
target_key = next(key for key, value in image_label.items() if value == f)
print(target_key)
plt.imshow(x_trainscaled[0])
model_path = 'C:\data\gen ai projects/leaf_detection.h5'
model.save(model_path)





