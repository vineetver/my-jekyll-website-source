---
layout: posts
title:  "Multi-Class Classification with Tensorflow and Keras"
share: true
comments: true
author_profile: false
permalink: /Work/multi-class-neural-nets
header:
  teaser: /assets/images/posts/work/neural.jpg
  og_image: /assets/images/posts/work/neural.jpge
excerpt: "Simple Multi-Class Neural Net model."
related: true

---

# Multi-Class Classification
<img src="/assets/images/posts/work/neural.jpg">

## Objectives:

  * Create a deep neural network that performs multi-class classification.
  * Tune the deep neural network.


## About the Dataset.
  
This MNIST dataset contains a lot of examples:

* The MNIST training set contains 60,000 examples.
* The MNIST test set contains 10,000 examples.

Each example contains a pixel map showing how a person wrote a digit. For example, the following images shows how a person wrote the digit `1` and how that digit might be represented in a 14x14 pixel map (after the input data is normalized). 

![Two images. The first image shows a somewhat fuzzy digit one. The second image shows a 14x14 floating-point array in which most of the cells contain 0 but a few cells contain values between 0.0 and 1.0. The pattern of nonzero values corresponds to the image of the fuzzy digit in the first image.](https://www.tensorflow.org/images/MNIST-Matrix.png)

Each example in the MNIST dataset consists of:

* A label specified by a rater.  Each label must be an integer from 0 to 9.  
* A 28x28 pixel map, where each pixel is an integer between 0 and 255. The pixel values are on a gray scale in which 0 represents white, 255 represents black, and values between 0 and 255 represent various shades of gray.  

This is a multi-class classification problem with 10 output classes.

## Importing modules


```python
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from matplotlib import pyplot as plt

# adjust the granularity
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

# improves formatting when ouputting NumPy arrays
np.set_printoptions(linewidth = 200)
```

## Loading the dataset

In `tf.keras` the function for importing MNIST dataset is called `mnist.load_data()`:


```python
(x_train, y_train),(x_test, y_test) = tf.keras.datasets.mnist.load_data()

```

* `x_train` contains the training set's features.
* `y_train` contains the training set's labels.
* `x_test` contains the test set's features.
* `y_test` contains the test set's labels.

#### MNIST dataset is already shuffled.

## Exploring the dataset

The .csv file for MNIST does not contain column names. Instead of column names, you use ordinal numbers to access different subsets of the MNIST dataset. 


```python
# example #1923 of the training set.
x_train[1918]
```




    array([[  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   1, 171, 253, 133,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0, 125, 252, 252, 247,  93,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   6, 160, 252, 252, 252, 211,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [  0,   0,   0,   0,   0,   0,   0,  37, 252, 252, 210, 108, 190,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [  0,   0,   0,   0,   0,   0,   0,  37, 252, 252, 179,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [  0,   0,   0,   0,   0,   0,   0,  37, 252, 252, 179,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [  0,   0,   0,   0,   0,   0,   0,  37, 252, 252, 179,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [  0,   0,   0,   0,   0,   0,   0,  37, 252, 252, 179,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [  0,   0,   0,   0,   0,   0,   0,  37, 253, 253, 180,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [  0,   0,   0,   0,   0,   0,   0,  37, 252, 252, 242,  62,   0,   0,   0,  11, 155, 218, 217, 196,  73,   0,   0,   0,   0,   0,   0,   0],
           [  0,   0,   0,   0,   0,   0,   0,  32, 236, 252, 252,  71,   0,   0,  73, 191, 252, 253, 252, 252, 252, 105,   0,   0,   0,   0,   0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0, 144, 252, 252,  71,   0, 145, 237, 252, 252, 253, 252, 252, 252, 144,   0,   0,   0,   0,   0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,  84, 253, 253, 232,  47, 255, 253, 253, 253,  84,   0, 182, 253, 255,  35,   0,   0,   0,   0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   1, 128, 252, 252, 232, 253, 252, 179,  35,   0,   0,  57, 252, 253,  35,   0,   0,   0,   0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,  94, 247, 252, 252, 253, 200,  20,   0,   0,   0, 140, 252, 253,  35,   0,   0,   0,   0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 134, 252, 252, 253, 179,   0,   0,   0,   0, 181, 252, 253,  35,   0,   0,   0,   0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  73, 253, 253, 255, 253, 232, 109,   0,  94, 212, 253, 145,   0,   0,   0,   0,   0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  73, 252, 252, 253, 252, 252, 252, 218, 247, 252, 252,  20,   0,   0,   0,   0,   0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  63, 241, 252, 253, 231, 252, 252, 253, 252, 246, 132,   0,   0,   0,   0,   0,   0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 181, 252, 108,  46, 108, 108, 108, 108,  92,   0,   0,   0,   0,   0,   0,   0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0]], dtype=uint8)




```python
x_train.shape 
```




    (60000, 28, 28)



using `matplotlib.pyplot.imshow` to interpret the array as an image. 


```python
plt.imshow(x_train[1918])
```




    <matplotlib.image.AxesImage at 0x7f3f044ff430>




![png](/assets/images/notebook/nb7/output_12_1.png)



```python
# 5th row of example 1918
x_train[1918][5]
```




    array([  0,   0,   0,   0,   0,   0,   0,   6, 160, 252, 252, 252, 211,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0], dtype=uint8)




```python
# 19th pixel of row 5 of example 1918
x_train[2917][10][19]
```




    11



## Normalizing feature values


```python
x_train_normalized = x_train / 255.00
x_test_normalized = x_test / 255.00
print(x_train_normalized[1918][5]) 
```

    [0.         0.         0.         0.         0.         0.         0.         0.02352941 0.62745098 0.98823529 0.98823529 0.98823529 0.82745098 0.         0.         0.         0.         0.
     0.         0.         0.         0.         0.         0.         0.         0.         0.         0.        ]


## Defining a plotting function


```python
""" curve of one or more classification metrics vs epoch """ 
def plot_curve(epochs, hist, list_of_metrics): 

# METRICS = [
#      keras.metrics.TruePositives(name='tp'),
#      keras.metrics.FalsePositives(name='fp'),
#      keras.metrics.TrueNegatives(name='tn'),
#      keras.metrics.FalseNegatives(name='fn'), 
#      keras.metrics.BinaryAccuracy(name='accuracy'),
#      keras.metrics.Precision(name='precision'),
#      keras.metrics.Recall(name='recall'),
#      keras.metrics.AUC(name='auc'),
#   ]

  plt.figure()
  plt.xlabel("Epoch")
  plt.ylabel("Value")

  for m in list_of_metrics:
    x = hist[m]
    plt.plot(epochs[1:], x[1:], label=m)

  plt.legend()
```

## Creating a deep neural net model

The `create_model` function defines the topography of the model:

* The number of layers in the deep neural net.
* The number of nodes in each layer.
* regularization of layers.

The `create_model` function also defines the activation function of each layer.  The activation function of the output layer is softmax, which will give 10 different outputs for each example.


```python
def create_model(my_learning_rate):
  
  model = tf.keras.models.Sequential()

    
  # converting two-dimensional 28x28 array into a one-dimensional 
  # 784 element array.
  model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))


  # first hidden layer  
  model.add(tf.keras.layers.Dense(units=20, activation='relu'))
  
  # second hidden layer
  model.add(tf.keras.layers.Dense(units=10, activation='relu'))
    
  # dropout regularization layer 
  model.add(tf.keras.layers.Dropout(rate=0.2))

  # output layer
  model.add(tf.keras.layers.Dense(units=10, activation='softmax'))     
                           
  # compiling the model  
  model.compile(optimizer=tf.keras.optimizers.Adam(lr=my_learning_rate),
                loss="sparse_categorical_crossentropy",
                metrics=['accuracy'])
  
  return model    


""" Train the model """
def train_model(model, train_features, train_label, epochs,
                batch_size=None, validation_split=0.1):
  

  history = model.fit(x=train_features, y=train_label, batch_size=batch_size,
                      epochs=epochs, shuffle=True, 
                      validation_split=validation_split)
 
  
  epochs = history.epoch
  hist = pd.DataFrame(history.history)

  return epochs, hist    
```

## Training the model


```python
import warnings
warnings.filterwarnings('ignore')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

#suppressing warnings
```


```python
learning_rate = 0.001
epochs = 50
batch_size = 4000
validation_split = 0.2

my_model = None


# model's topography.
my_model = create_model(learning_rate)


epochs, hist = train_model(my_model, x_train_normalized, y_train, 
                           epochs, batch_size, validation_split)


list_of_metrics_to_plot = ['accuracy']
plot_curve(epochs, hist, list_of_metrics_to_plot)


print("\n Evaluating the new model against the test set:")
my_model.evaluate(x=x_test_normalized, y=y_test, batch_size=batch_size)
```

    Epoch 1/50
    12/12 [==============================] - 0s 13ms/step - loss: 2.2236 - accuracy: 0.1914 - val_loss: 2.0753 - val_accuracy: 0.2909
    Epoch 2/50
    12/12 [==============================] - 0s 5ms/step - loss: 2.0115 - accuracy: 0.2918 - val_loss: 1.8384 - val_accuracy: 0.4392
    Epoch 3/50
    12/12 [==============================] - 0s 5ms/step - loss: 1.8145 - accuracy: 0.3811 - val_loss: 1.6119 - val_accuracy: 0.5561
    Epoch 4/50
    12/12 [==============================] - 0s 5ms/step - loss: 1.6535 - accuracy: 0.4578 - val_loss: 1.4225 - val_accuracy: 0.6425
    Epoch 5/50
    12/12 [==============================] - 0s 5ms/step - loss: 1.5132 - accuracy: 0.5259 - val_loss: 1.2548 - val_accuracy: 0.7176
    Epoch 6/50
    12/12 [==============================] - 0s 5ms/step - loss: 1.3767 - accuracy: 0.5722 - val_loss: 1.1034 - val_accuracy: 0.7773
    Epoch 7/50
    12/12 [==============================] - 0s 5ms/step - loss: 1.2727 - accuracy: 0.6187 - val_loss: 0.9875 - val_accuracy: 0.7993
    Epoch 8/50
    12/12 [==============================] - 0s 5ms/step - loss: 1.1808 - accuracy: 0.6536 - val_loss: 0.8910 - val_accuracy: 0.8188
    Epoch 9/50
    12/12 [==============================] - 0s 5ms/step - loss: 1.1116 - accuracy: 0.6748 - val_loss: 0.8146 - val_accuracy: 0.8329
    Epoch 10/50
    12/12 [==============================] - 0s 5ms/step - loss: 1.0408 - accuracy: 0.6968 - val_loss: 0.7504 - val_accuracy: 0.8452
    Epoch 11/50
    12/12 [==============================] - 0s 6ms/step - loss: 0.9852 - accuracy: 0.7128 - val_loss: 0.6901 - val_accuracy: 0.8565
    Epoch 12/50
    12/12 [==============================] - 0s 5ms/step - loss: 0.9389 - accuracy: 0.7207 - val_loss: 0.6421 - val_accuracy: 0.8636
    Epoch 13/50
    12/12 [==============================] - 0s 5ms/step - loss: 0.8957 - accuracy: 0.7305 - val_loss: 0.6017 - val_accuracy: 0.8702
    Epoch 14/50
    12/12 [==============================] - 0s 5ms/step - loss: 0.8627 - accuracy: 0.7370 - val_loss: 0.5662 - val_accuracy: 0.8738
    Epoch 15/50
    12/12 [==============================] - 0s 5ms/step - loss: 0.8342 - accuracy: 0.7427 - val_loss: 0.5369 - val_accuracy: 0.8779
    Epoch 16/50
    12/12 [==============================] - 0s 5ms/step - loss: 0.8071 - accuracy: 0.7532 - val_loss: 0.5129 - val_accuracy: 0.8818
    Epoch 17/50
    12/12 [==============================] - 0s 5ms/step - loss: 0.7873 - accuracy: 0.7561 - val_loss: 0.4968 - val_accuracy: 0.8854
    Epoch 18/50
    12/12 [==============================] - 0s 5ms/step - loss: 0.7708 - accuracy: 0.7643 - val_loss: 0.4747 - val_accuracy: 0.8890
    Epoch 19/50
    12/12 [==============================] - 0s 5ms/step - loss: 0.7530 - accuracy: 0.7678 - val_loss: 0.4614 - val_accuracy: 0.8908
    Epoch 20/50
    12/12 [==============================] - 0s 6ms/step - loss: 0.7321 - accuracy: 0.7760 - val_loss: 0.4463 - val_accuracy: 0.8928
    Epoch 21/50
    12/12 [==============================] - 0s 6ms/step - loss: 0.7225 - accuracy: 0.7785 - val_loss: 0.4344 - val_accuracy: 0.8945
    Epoch 22/50
    12/12 [==============================] - 0s 7ms/step - loss: 0.7043 - accuracy: 0.7810 - val_loss: 0.4244 - val_accuracy: 0.8972
    Epoch 23/50
    12/12 [==============================] - 0s 5ms/step - loss: 0.6974 - accuracy: 0.7845 - val_loss: 0.4152 - val_accuracy: 0.8994
    Epoch 24/50
    12/12 [==============================] - 0s 5ms/step - loss: 0.6902 - accuracy: 0.7870 - val_loss: 0.4076 - val_accuracy: 0.9018
    Epoch 25/50
    12/12 [==============================] - 0s 6ms/step - loss: 0.6760 - accuracy: 0.7924 - val_loss: 0.3981 - val_accuracy: 0.9014
    Epoch 26/50
    12/12 [==============================] - 0s 5ms/step - loss: 0.6675 - accuracy: 0.7945 - val_loss: 0.3903 - val_accuracy: 0.9036
    Epoch 27/50
    12/12 [==============================] - 0s 5ms/step - loss: 0.6547 - accuracy: 0.7976 - val_loss: 0.3815 - val_accuracy: 0.9054
    Epoch 28/50
    12/12 [==============================] - 0s 5ms/step - loss: 0.6540 - accuracy: 0.7988 - val_loss: 0.3759 - val_accuracy: 0.9071
    Epoch 29/50
    12/12 [==============================] - 0s 5ms/step - loss: 0.6383 - accuracy: 0.8030 - val_loss: 0.3661 - val_accuracy: 0.9093
    Epoch 30/50
    12/12 [==============================] - 0s 6ms/step - loss: 0.6392 - accuracy: 0.8013 - val_loss: 0.3627 - val_accuracy: 0.9091
    Epoch 31/50
    12/12 [==============================] - 0s 6ms/step - loss: 0.6283 - accuracy: 0.8052 - val_loss: 0.3555 - val_accuracy: 0.9122
    Epoch 32/50
    12/12 [==============================] - 0s 6ms/step - loss: 0.6218 - accuracy: 0.8091 - val_loss: 0.3506 - val_accuracy: 0.9127
    Epoch 33/50
    12/12 [==============================] - 0s 5ms/step - loss: 0.6098 - accuracy: 0.8105 - val_loss: 0.3437 - val_accuracy: 0.9143
    Epoch 34/50
    12/12 [==============================] - 0s 5ms/step - loss: 0.6029 - accuracy: 0.8112 - val_loss: 0.3384 - val_accuracy: 0.9148
    Epoch 35/50
    12/12 [==============================] - 0s 5ms/step - loss: 0.5938 - accuracy: 0.8155 - val_loss: 0.3337 - val_accuracy: 0.9150
    Epoch 36/50
    12/12 [==============================] - 0s 6ms/step - loss: 0.5907 - accuracy: 0.8154 - val_loss: 0.3278 - val_accuracy: 0.9168
    Epoch 37/50
    12/12 [==============================] - 0s 5ms/step - loss: 0.5841 - accuracy: 0.8181 - val_loss: 0.3256 - val_accuracy: 0.9172
    Epoch 38/50
    12/12 [==============================] - 0s 5ms/step - loss: 0.5781 - accuracy: 0.8179 - val_loss: 0.3173 - val_accuracy: 0.9181
    Epoch 39/50
    12/12 [==============================] - 0s 6ms/step - loss: 0.5694 - accuracy: 0.8182 - val_loss: 0.3150 - val_accuracy: 0.9201
    Epoch 40/50
    12/12 [==============================] - 0s 5ms/step - loss: 0.5680 - accuracy: 0.8180 - val_loss: 0.3106 - val_accuracy: 0.9195
    Epoch 41/50
    12/12 [==============================] - 0s 5ms/step - loss: 0.5629 - accuracy: 0.8210 - val_loss: 0.3077 - val_accuracy: 0.9203
    Epoch 42/50
    12/12 [==============================] - 0s 5ms/step - loss: 0.5567 - accuracy: 0.8228 - val_loss: 0.3032 - val_accuracy: 0.9210
    Epoch 43/50
    12/12 [==============================] - 0s 6ms/step - loss: 0.5491 - accuracy: 0.8254 - val_loss: 0.2975 - val_accuracy: 0.9227
    Epoch 44/50
    12/12 [==============================] - 0s 5ms/step - loss: 0.5422 - accuracy: 0.8287 - val_loss: 0.2957 - val_accuracy: 0.9227
    Epoch 45/50
    12/12 [==============================] - 0s 5ms/step - loss: 0.5380 - accuracy: 0.8295 - val_loss: 0.2900 - val_accuracy: 0.9242
    Epoch 46/50
    12/12 [==============================] - 0s 6ms/step - loss: 0.5306 - accuracy: 0.8341 - val_loss: 0.2895 - val_accuracy: 0.9243
    Epoch 47/50
    12/12 [==============================] - 0s 5ms/step - loss: 0.5306 - accuracy: 0.8357 - val_loss: 0.2840 - val_accuracy: 0.9247
    Epoch 48/50
    12/12 [==============================] - 0s 5ms/step - loss: 0.5212 - accuracy: 0.8406 - val_loss: 0.2829 - val_accuracy: 0.9256
    Epoch 49/50
    12/12 [==============================] - 0s 5ms/step - loss: 0.5177 - accuracy: 0.8433 - val_loss: 0.2785 - val_accuracy: 0.9262
    Epoch 50/50
    12/12 [==============================] - 0s 4ms/step - loss: 0.5168 - accuracy: 0.8428 - val_loss: 0.2756 - val_accuracy: 0.9262
    
     Evaluating the new model against the test set:
    3/3 [==============================] - 0s 2ms/step - loss: 0.2806 - accuracy: 0.9273





    [0.28058871626853943, 0.927299976348877]




![png](/assets/images/notebook/nb7/output_23_2.png)


## Optimizing the model

Let's try to reach at least 98% accuracy against the test set


```python
def create_model(my_learning_rate):
  
  model = tf.keras.models.Sequential()

  model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))


  """ Increasing nodes increased accuracy """            # <--------------------  
  model.add(tf.keras.layers.Dense(units=256, activation='relu'))
  
  """ Adding a second layer decreased loss """           # <--------------------
  model.add(tf.keras.layers.Dense(units=127, activation='relu'))
    
 
  model.add(tf.keras.layers.Dropout(rate=0.2))


  model.add(tf.keras.layers.Dense(units=10, activation='softmax'))     
                           

  model.compile(optimizer=tf.keras.optimizers.Adam(lr=my_learning_rate),
                loss="sparse_categorical_crossentropy",
                metrics=['accuracy'])
  
  return model    


""" Train the model """
def train_model(model, train_features, train_label, epochs,
                batch_size=None, validation_split=0.1):
  

  history = model.fit(x=train_features, y=train_label, batch_size=batch_size,
                      epochs=epochs, shuffle=True, 
                      validation_split=validation_split)
 
  
  epochs = history.epoch
  hist = pd.DataFrame(history.history)

  return epochs, hist    
```


```python
learning_rate = 0.001
epochs = 60
batch_size = 4000
validation_split = 0.2

my_model = None


# model's topography.
my_model = create_model(learning_rate)


epochs, hist = train_model(my_model, x_train_normalized, y_train, 
                           epochs, batch_size, validation_split)


list_of_metrics_to_plot = ['accuracy']
plot_curve(epochs, hist, list_of_metrics_to_plot)


print("\n Evaluating the new model against the test set:")
my_model.evaluate(x=x_test_normalized, y=y_test, batch_size=batch_size)
```

    Epoch 1/60
    12/12 [==============================] - 0s 13ms/step - loss: 1.5717 - accuracy: 0.5836 - val_loss: 0.7117 - val_accuracy: 0.8388
    Epoch 2/60
    12/12 [==============================] - 0s 6ms/step - loss: 0.5979 - accuracy: 0.8247 - val_loss: 0.3529 - val_accuracy: 0.9001
    Epoch 3/60
    12/12 [==============================] - 0s 6ms/step - loss: 0.3901 - accuracy: 0.8834 - val_loss: 0.2781 - val_accuracy: 0.9217
    Epoch 4/60
    12/12 [==============================] - 0s 6ms/step - loss: 0.3114 - accuracy: 0.9085 - val_loss: 0.2380 - val_accuracy: 0.9308
    Epoch 5/60
    12/12 [==============================] - 0s 6ms/step - loss: 0.2666 - accuracy: 0.9235 - val_loss: 0.2095 - val_accuracy: 0.9400
    Epoch 6/60
    12/12 [==============================] - 0s 6ms/step - loss: 0.2309 - accuracy: 0.9330 - val_loss: 0.1892 - val_accuracy: 0.9454
    Epoch 7/60
    12/12 [==============================] - 0s 6ms/step - loss: 0.2068 - accuracy: 0.9402 - val_loss: 0.1708 - val_accuracy: 0.9501
    Epoch 8/60
    12/12 [==============================] - 0s 6ms/step - loss: 0.1862 - accuracy: 0.9457 - val_loss: 0.1584 - val_accuracy: 0.9542
    Epoch 9/60
    12/12 [==============================] - 0s 6ms/step - loss: 0.1690 - accuracy: 0.9513 - val_loss: 0.1476 - val_accuracy: 0.9578
    Epoch 10/60
    12/12 [==============================] - 0s 6ms/step - loss: 0.1553 - accuracy: 0.9552 - val_loss: 0.1402 - val_accuracy: 0.9586
    Epoch 11/60
    12/12 [==============================] - 0s 6ms/step - loss: 0.1427 - accuracy: 0.9588 - val_loss: 0.1333 - val_accuracy: 0.9606
    Epoch 12/60
    12/12 [==============================] - 0s 6ms/step - loss: 0.1292 - accuracy: 0.9629 - val_loss: 0.1251 - val_accuracy: 0.9621
    Epoch 13/60
    12/12 [==============================] - 0s 6ms/step - loss: 0.1205 - accuracy: 0.9651 - val_loss: 0.1199 - val_accuracy: 0.9637
    Epoch 14/60
    12/12 [==============================] - 0s 6ms/step - loss: 0.1119 - accuracy: 0.9682 - val_loss: 0.1140 - val_accuracy: 0.9657
    Epoch 15/60
    12/12 [==============================] - 0s 6ms/step - loss: 0.1055 - accuracy: 0.9694 - val_loss: 0.1106 - val_accuracy: 0.9663
    Epoch 16/60
    12/12 [==============================] - 0s 6ms/step - loss: 0.0979 - accuracy: 0.9718 - val_loss: 0.1064 - val_accuracy: 0.9677
    Epoch 17/60
    12/12 [==============================] - 0s 6ms/step - loss: 0.0912 - accuracy: 0.9742 - val_loss: 0.1025 - val_accuracy: 0.9693
    Epoch 18/60
    12/12 [==============================] - 0s 6ms/step - loss: 0.0858 - accuracy: 0.9754 - val_loss: 0.1006 - val_accuracy: 0.9699
    Epoch 19/60
    12/12 [==============================] - 0s 7ms/step - loss: 0.0815 - accuracy: 0.9770 - val_loss: 0.0981 - val_accuracy: 0.9689
    Epoch 20/60
    12/12 [==============================] - 0s 7ms/step - loss: 0.0768 - accuracy: 0.9781 - val_loss: 0.0973 - val_accuracy: 0.9693
    Epoch 21/60
    12/12 [==============================] - 0s 6ms/step - loss: 0.0704 - accuracy: 0.9795 - val_loss: 0.0941 - val_accuracy: 0.9710
    Epoch 22/60
    12/12 [==============================] - 0s 6ms/step - loss: 0.0675 - accuracy: 0.9808 - val_loss: 0.0908 - val_accuracy: 0.9718
    Epoch 23/60
    12/12 [==============================] - 0s 6ms/step - loss: 0.0634 - accuracy: 0.9818 - val_loss: 0.0877 - val_accuracy: 0.9729
    Epoch 24/60
    12/12 [==============================] - 0s 6ms/step - loss: 0.0602 - accuracy: 0.9826 - val_loss: 0.0897 - val_accuracy: 0.9733
    Epoch 25/60
    12/12 [==============================] - 0s 6ms/step - loss: 0.0565 - accuracy: 0.9841 - val_loss: 0.0858 - val_accuracy: 0.9732
    Epoch 26/60
    12/12 [==============================] - 0s 6ms/step - loss: 0.0528 - accuracy: 0.9853 - val_loss: 0.0857 - val_accuracy: 0.9730
    Epoch 27/60
    12/12 [==============================] - 0s 6ms/step - loss: 0.0500 - accuracy: 0.9853 - val_loss: 0.0842 - val_accuracy: 0.9741
    Epoch 28/60
    12/12 [==============================] - 0s 6ms/step - loss: 0.0471 - accuracy: 0.9870 - val_loss: 0.0841 - val_accuracy: 0.9736
    Epoch 29/60
    12/12 [==============================] - 0s 6ms/step - loss: 0.0444 - accuracy: 0.9872 - val_loss: 0.0817 - val_accuracy: 0.9747
    Epoch 30/60
    12/12 [==============================] - 0s 7ms/step - loss: 0.0419 - accuracy: 0.9887 - val_loss: 0.0821 - val_accuracy: 0.9741
    Epoch 31/60
    12/12 [==============================] - 0s 6ms/step - loss: 0.0408 - accuracy: 0.9889 - val_loss: 0.0809 - val_accuracy: 0.9753
    Epoch 32/60
    12/12 [==============================] - 0s 6ms/step - loss: 0.0392 - accuracy: 0.9890 - val_loss: 0.0807 - val_accuracy: 0.9749
    Epoch 33/60
    12/12 [==============================] - 0s 7ms/step - loss: 0.0366 - accuracy: 0.9897 - val_loss: 0.0799 - val_accuracy: 0.9758
    Epoch 34/60
    12/12 [==============================] - 0s 6ms/step - loss: 0.0342 - accuracy: 0.9908 - val_loss: 0.0789 - val_accuracy: 0.9764
    Epoch 35/60
    12/12 [==============================] - 0s 6ms/step - loss: 0.0325 - accuracy: 0.9909 - val_loss: 0.0788 - val_accuracy: 0.9762
    Epoch 36/60
    12/12 [==============================] - 0s 6ms/step - loss: 0.0316 - accuracy: 0.9919 - val_loss: 0.0786 - val_accuracy: 0.9769
    Epoch 37/60
    12/12 [==============================] - 0s 6ms/step - loss: 0.0290 - accuracy: 0.9924 - val_loss: 0.0784 - val_accuracy: 0.9768
    Epoch 38/60
    12/12 [==============================] - 0s 6ms/step - loss: 0.0283 - accuracy: 0.9925 - val_loss: 0.0779 - val_accuracy: 0.9769
    Epoch 39/60
    12/12 [==============================] - 0s 6ms/step - loss: 0.0268 - accuracy: 0.9931 - val_loss: 0.0780 - val_accuracy: 0.9769
    Epoch 40/60
    12/12 [==============================] - 0s 6ms/step - loss: 0.0257 - accuracy: 0.9934 - val_loss: 0.0773 - val_accuracy: 0.9767
    Epoch 41/60
    12/12 [==============================] - 0s 7ms/step - loss: 0.0247 - accuracy: 0.9933 - val_loss: 0.0770 - val_accuracy: 0.9774
    Epoch 42/60
    12/12 [==============================] - 0s 7ms/step - loss: 0.0227 - accuracy: 0.9939 - val_loss: 0.0762 - val_accuracy: 0.9775
    Epoch 43/60
    12/12 [==============================] - 0s 6ms/step - loss: 0.0216 - accuracy: 0.9945 - val_loss: 0.0765 - val_accuracy: 0.9770
    Epoch 44/60
    12/12 [==============================] - 0s 6ms/step - loss: 0.0212 - accuracy: 0.9947 - val_loss: 0.0777 - val_accuracy: 0.9776
    Epoch 45/60
    12/12 [==============================] - 0s 6ms/step - loss: 0.0200 - accuracy: 0.9949 - val_loss: 0.0775 - val_accuracy: 0.9773
    Epoch 46/60
    12/12 [==============================] - 0s 6ms/step - loss: 0.0183 - accuracy: 0.9955 - val_loss: 0.0780 - val_accuracy: 0.9771
    Epoch 47/60
    12/12 [==============================] - 0s 6ms/step - loss: 0.0186 - accuracy: 0.9952 - val_loss: 0.0781 - val_accuracy: 0.9778
    Epoch 48/60
    12/12 [==============================] - 0s 7ms/step - loss: 0.0169 - accuracy: 0.9961 - val_loss: 0.0771 - val_accuracy: 0.9777
    Epoch 49/60
    12/12 [==============================] - 0s 6ms/step - loss: 0.0168 - accuracy: 0.9961 - val_loss: 0.0781 - val_accuracy: 0.9773
    Epoch 50/60
    12/12 [==============================] - 0s 6ms/step - loss: 0.0162 - accuracy: 0.9959 - val_loss: 0.0783 - val_accuracy: 0.9777
    Epoch 51/60
    12/12 [==============================] - 0s 6ms/step - loss: 0.0152 - accuracy: 0.9963 - val_loss: 0.0775 - val_accuracy: 0.9785
    Epoch 52/60
    12/12 [==============================] - 0s 6ms/step - loss: 0.0143 - accuracy: 0.9965 - val_loss: 0.0788 - val_accuracy: 0.9777
    Epoch 53/60
    12/12 [==============================] - 0s 6ms/step - loss: 0.0136 - accuracy: 0.9971 - val_loss: 0.0792 - val_accuracy: 0.9783
    Epoch 54/60
    12/12 [==============================] - 0s 6ms/step - loss: 0.0130 - accuracy: 0.9973 - val_loss: 0.0797 - val_accuracy: 0.9775
    Epoch 55/60
    12/12 [==============================] - 0s 6ms/step - loss: 0.0125 - accuracy: 0.9972 - val_loss: 0.0799 - val_accuracy: 0.9775
    Epoch 56/60
    12/12 [==============================] - 0s 6ms/step - loss: 0.0119 - accuracy: 0.9973 - val_loss: 0.0796 - val_accuracy: 0.9775
    Epoch 57/60
    12/12 [==============================] - 0s 6ms/step - loss: 0.0115 - accuracy: 0.9977 - val_loss: 0.0791 - val_accuracy: 0.9778
    Epoch 58/60
    12/12 [==============================] - 0s 6ms/step - loss: 0.0108 - accuracy: 0.9977 - val_loss: 0.0784 - val_accuracy: 0.9785
    Epoch 59/60
    12/12 [==============================] - 0s 7ms/step - loss: 0.0103 - accuracy: 0.9979 - val_loss: 0.0794 - val_accuracy: 0.9784
    Epoch 60/60
    12/12 [==============================] - 0s 5ms/step - loss: 0.0098 - accuracy: 0.9980 - val_loss: 0.0794 - val_accuracy: 0.9780
    
     Evaluating the new model against the test set:
    3/3 [==============================] - 0s 2ms/step - loss: 0.0724 - accuracy: 0.9811





    [0.07239113003015518, 0.9811000227928162]




![png](/assets/images/notebook/nb7/output_26_2.png)


### Reached 98.1% test accuracy with the following configuration:
   * First hidden layer of 256nodes 
   * second hidden layer of 127 nodes
   * dropout regularization rate of 0.2
   * epochs = 60, batch_size = 4000


