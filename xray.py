from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
import tensorflow
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
classifier = Sequential()
classifier.add(Conv2D(32,(3,3),input_shape=(64,64,3),activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(2,2),strides=2)) #if stride not given it equal to pool filter size
classifier.add(Conv2D(32,(3,3),activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(2,2),strides=2))
classifier.add(Flatten())
classifier.add(Dense(units=128,activation='relu'))
classifier.add(Dense(units=1,activation='sigmoid'))

classifier.compile(optimizer='adam',loss='mse',metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.1,
                                   zoom_range=0.1,
                                   horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

#Training Set
train_set = train_datagen.flow_from_directory('C:\\Users\\ahmed\\PycharmProjects\\untitled\\xray\\chest_xray\\train\\',
                                             target_size=(64,64),
                                             batch_size=32,
                                             class_mode='binary')
#Validation Set

#Test Set /no output available


classifier.fit_generator(train_set,
                        steps_per_epoch=40,
                        epochs = 10,

                        validation_steps = 20,
                        )


img1 = image.load_img('C:\\Users\\ahmed\\PycharmProjects\\untitled\\xray\\chest_xray\\val\\PNEUMONIA\\person1949_bacteria_4880.jpeg', target_size=(64, 64))
img = image.img_to_array(img1)
img = img/255
img=np.expand_dims(img,axis=0)
# create a batch of size 1 [N,H,W,C]

prediction = classifier.predict_classes(img, batch_size=None) #gives all class prob.
print(prediction)
plt.text(20, 62, prediction, color='red', fontsize=18, bbox=dict(facecolor='white', alpha=0.8))
plt.imshow(img1)
plt.show()
img1 = image.load_img('C:\\Users\\ahmed\\PycharmProjects\\untitled\\xray\\chest_xray\\val\\PNEUMONIA\\person1946_bacteria_4875.jpeg', target_size=(64, 64))
img = image.img_to_array(img1)
img = img/255
img=np.expand_dims(img,axis=0)
# create a batch of size 1 [N,H,W,C]

prediction = classifier.predict_classes(img, batch_size=None) #gives all class prob.
print(prediction)
plt.text(20, 62, prediction, color='red', fontsize=18, bbox=dict(facecolor='white', alpha=0.8))
plt.imshow(img1)
plt.show()
img1 = image.load_img('C:\\Users\\ahmed\\PycharmProjects\\untitled\\xray\\chest_xray\\val\\PNEUMONIA\\person1954_bacteria_4886.jpeg', target_size=(64, 64))
img = image.img_to_array(img1)
img = img/255
img=np.expand_dims(img,axis=0)
# create a batch of size 1 [N,H,W,C]

prediction = classifier.predict_classes(img, batch_size=None) #gives all class prob.
print(prediction)
plt.text(20, 62, prediction, color='red', fontsize=18, bbox=dict(facecolor='white', alpha=0.8))
plt.imshow(img1)
plt.show()
## normal
img1 = image.load_img('C:\\Users\\ahmed\\PycharmProjects\\untitled\\xray\\chest_xray\\val\\NORMAL\\NORMAL2-IM-1427-0001.jpeg', target_size=(64, 64))
img = image.img_to_array(img1)
img = img/255
img=np.expand_dims(img,axis=0)
# create a batch of size 1 [N,H,W,C]

prediction = classifier.predict_classes(img, batch_size=None) #gives all class prob.
print(prediction)
plt.text(20, 62, prediction, color='red', fontsize=18, bbox=dict(facecolor='white', alpha=0.8))
plt.imshow(img1)
plt.show()
img1 = image.load_img('C:\\Users\\ahmed\\PycharmProjects\\untitled\\xray\\chest_xray\\val\\NORMAL\\NORMAL2-IM-1431-0001.jpeg', target_size=(64, 64))
img = image.img_to_array(img1)
img = img/255
img=np.expand_dims(img,axis=0)
# create a batch of size 1 [N,H,W,C]

prediction = classifier.predict_classes(img, batch_size=None) #gives all class prob.
print(prediction)
plt.text(20, 62, prediction, color='red', fontsize=18, bbox=dict(facecolor='white', alpha=0.8))
plt.imshow(img1)
plt.show()
img1 = image.load_img('C:\\Users\\ahmed\\PycharmProjects\\untitled\\xray\\chest_xray\\val\\NORMAL\\NORMAL2-IM-1442-0001.jpeg', target_size=(64, 64))
img = image.img_to_array(img1)
img = img/255
img=np.expand_dims(img,axis=0)
# create a batch of size 1 [N,H,W,C]

prediction = classifier.predict_classes(img, batch_size=None) #gives all class prob.
print(prediction)
plt.text(20, 62, prediction, color='red', fontsize=18, bbox=dict(facecolor='white', alpha=0.8))
plt.imshow(img1)
plt.show()








