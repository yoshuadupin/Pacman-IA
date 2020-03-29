import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import re
ftrain = open('training.txt' , 'r')
flabel = open('label.txt' , 'r')
class_names=['abajo' ,'arriba' , 'izquierda' , 'derecha']
set = re.split('\n' , ftrain.read())
label = re.split('\n' , flabel.read())

temp=[]
trainSet=[]
labelSet=[]
for a in set:
    temp.append(re.split('\s', a))
for i in range(len(temp)-1):
    temp1 = []
    for j in range(len(temp[i])):
        if j==len(temp[i])-1:
            temp1.append(int(float(temp[i][j])))
        else:
            temp1.append(int(temp[i][j]))
    trainSet.append(temp1)

temp=[]
for b in label:
    temp.append(re.split('\s', b))
for i in range(len(temp)-1):
    temp1 = []
    for j in range(len(temp[i])):
        temp1.append(int(temp[i][j]))
    labelSet.append(temp1)

print(labelSet)

ftrain.close()
flabel.close()

model  = tf.keras.Sequential()

#posicion del fantasma neuronas X y Y 
#distancia de los fantasma y distancia  posicion de pacman
#y cuatro neuronas mas de la direccion
#2 neuronas de salida para decidir que direccion tomar
model.add(keras.layers.Dense(13 ,activation="relu" ))
model.add(keras.layers.Dense(4 , activation="sigmoid"))
model.compile(optimizer="adam" , loss="binary_crossentropy" , metrics=["accuracy"])

#model.fit(trainSet , labelSet , epochs=5)

#import numpy as np
#data = np.random.random((1000, 13))
#labels = np.random.randint(2, size=(1000, 2))

# Train the model, iterating on the data in batches of 32 samples
model.fit(trainSet, labelSet, epochs=10, batch_size=1)
x_test = np.random.randint(272, size=(100, 13))
y_test = np.random.randint(2, size=(100, 4))

results = model.predict(trainSet)
print(results[0])

print(class_names[np.argmax(results[0])])
