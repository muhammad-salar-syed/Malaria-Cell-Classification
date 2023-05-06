import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image
import keras
from ResNet50 import model_Resnet50
np.random.seed(10)


SIZE = 224
dataset = []  
label = []  

parasitized_images = os.listdir('./cell_images/Parasitized/')
for i, name in enumerate(parasitized_images):    
    
    if (name.split('.')[1] == 'png'):
        image = cv2.imread('./cell_images/Parasitized/' + name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((SIZE, SIZE))
        dataset.append(np.array(image))
        label.append(0)

uninfected_images = os.listdir('./cell_images/Uninfected/')
for i, name in enumerate(uninfected_images):   
    
    if (name.split('.')[1] == 'png'):
        image = cv2.imread('./cell_images/Uninfected/' + name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((SIZE, SIZE))
        dataset.append(np.array(image))
        label.append(1)
        
# model       
inp = keras.layers.Input(shape=(224,224,3))

conv1 = keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(inp)
pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
norm1 = keras.layers.BatchNormalization(axis = -1)(pool1)
drop1 = keras.layers.Dropout(rate=0.2)(norm1)

conv2 = keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(drop1)
pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
norm2 = keras.layers.BatchNormalization(axis = -1)(pool2)
drop2 = keras.layers.Dropout(rate=0.2)(norm2)

flat = keras.layers.Flatten()(drop2) 

dense1 = keras.layers.Dense(512, activation='relu')(flat)
norm3 = keras.layers.BatchNormalization(axis = -1)(dense1)
drop3 = keras.layers.Dropout(rate=0.2)(norm3)

dense2 = keras.layers.Dense(256, activation='relu')(drop3)
norm4 = keras.layers.BatchNormalization(axis = -1)(dense1)
drop4 = keras.layers.Dropout(rate=0.2)(norm4)

out = keras.layers.Dense(2, activation='sigmoid')(drop4)  

model = keras.Model(inputs=inp, outputs=out)
print(model.summary())

from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

X_train, X_test, y_train, y_test = train_test_split(dataset, to_categorical(np.array(label)), test_size = 0.20, random_state = 0)

model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])

#Fit the model
history = model.fit(np.array(X_train), 
                         y_train, 
                         batch_size = 8, 
                         verbose = 1, 
                         epochs = 20,     
                         validation_split = 0.1,
                         shuffle = False
                     )
model.save('./malaria_cnn.h5')

print("Test_Accuracy: {:.2f}%".format(model.evaluate(np.array(X_test), np.array(y_test))[1]*100))

labels = ['Parasitized', 'Uninfected']
Images=np.array(X_test)
y_pred = model.predict(Images)
y_pred[y_pred>0.5] = 1
y_pred[y_pred<0.5] = 0

predicted_class = np.argmax(y_pred, axis = 1)
plt.figure(figsize = (8 , 10))
for i in range(12):
    plt.subplot(4 , 3, i+1)
    plt.subplots_adjust(hspace = 0.3 , wspace = 0.1)
    plt.imshow(Images[i, :, :, :])
    plt.title('Class: {}'.format(labels[int(predicted_class[i])]),fontsize = 12)
    plt.axis('off')
    
from sklearn.metrics import  confusion_matrix
tn, fp, fn, tp = confusion_matrix(np.argmax(y_test, axis = 1),np.argmax(y_pred, axis = 1)).ravel()
precision = tp/(tp+fp)
recall = tp/(tp+fn)
F1_Score = 2 * (precision * recall) / (precision + recall)

print('precision: ',precision*100)
print('recall: ',recall*100)
print('F1 Score: ',F1_Score*100)

import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
plt.figure(figsize=(8,5))
sns.heatmap(confusion_matrix(np.argmax(y_test, axis = 1),np.argmax(y_pred, axis = 1)),annot=True,fmt='.3g',xticklabels=['Parasitized', 'Uninfected'],yticklabels=['Parasitized', 'Uninfected'])
plt.show()


