import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
print(tf.__version__)

model  = tf.keras.Sequential()

#posicion del fantasma neuronas X y Y 
#distancia de los fantasma y distancia  posicion de pacman
#y cuatro neuronas mas de la direccion
#2 neuronas de salida para decidir que direccion tomar
model.add(keras.layers.Dense(13 , activation="relu"))
model.add(keras.layers.Dense(2 , activation="sigmoid"))
model.compile(optimizer="adam" , loss="sparse_categorical_crossentropy" , metrics=["accuracy"])

model.fit(train , labels , epochs=5)

