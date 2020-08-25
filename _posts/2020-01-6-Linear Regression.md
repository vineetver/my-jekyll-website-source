---
layout: posts
title:  "Linear Regression with Tensorflow and Keras"
share: true
comments: true
author_profile: false
permalink: /Work/linear-regression
header:
  teaser: /assets/images/posts/work/lr.png
  og_image: /assets/images/posts/work/lr.png
excerpt: "This jupyter notebook uses a real dataset to predict the prices of houses in United States."
related: true

---




# Linear Regression with a Real Dataset

<img src="https://www.ecmwf.int/sites/default/files/Ai-image-deep-net-v3-690px.jpg">

------------------------------------------

This jupyter notebook uses a real dataset to predict the prices of houses in United States.

## Objectives

* Read a .csv file into a pandas DataFrame.
* Examining the Dataset.
* Experiment with different features in building a model.
* Tune the model's hyperparameters.

## About the Dataset
  
The [dataset for this notebook](https://www2.cdc.gov/nceh/lead/census90/house11/download.htm) is based on 1990 census data from California.

## Import relevant modules

```python
# Import relevant modules
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

# The following lines adjust the granularity
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format
```

### Loading the .csv file into a pandas DataFrame

The dataset has already been preprocessed. The following code cells imports the .csv file into a pandas DataFrame and scales the values in the label (`median_house_value`)

```python
# Import the dataset
training_df = pd.read_csv(filepath_or_buffer="data/california_housing_train.csv")

training_df.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>median_house_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-114.3</td>
      <td>34.2</td>
      <td>15.0</td>
      <td>5612.0</td>
      <td>1283.0</td>
      <td>1015.0</td>
      <td>472.0</td>
      <td>1.5</td>
      <td>66900.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-114.5</td>
      <td>34.4</td>
      <td>19.0</td>
      <td>7650.0</td>
      <td>1901.0</td>
      <td>1129.0</td>
      <td>463.0</td>
      <td>1.8</td>
      <td>80100.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-114.6</td>
      <td>33.7</td>
      <td>17.0</td>
      <td>720.0</td>
      <td>174.0</td>
      <td>333.0</td>
      <td>117.0</td>
      <td>1.7</td>
      <td>85700.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-114.6</td>
      <td>33.6</td>
      <td>14.0</td>
      <td>1501.0</td>
      <td>337.0</td>
      <td>515.0</td>
      <td>226.0</td>
      <td>3.2</td>
      <td>73400.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-114.6</td>
      <td>33.6</td>
      <td>20.0</td>
      <td>1454.0</td>
      <td>326.0</td>
      <td>624.0</td>
      <td>262.0</td>
      <td>1.9</td>
      <td>65500.0</td>
    </tr>
  </tbody>
</table>
</div>

```python
# scaling the label (median_house_value)
training_df["median_house_value"] /= 1000.0
```

Scaling `median_house_value` puts the value of each house in units of thousands. Scaling will keep loss values and learning rates in a friendlier range.  

Although scaling a label is usually not essential, scaling features in a multi-feature model usually is essential

## Examine the dataset

A large part of machine learning projects is getting to know your data. Pandas module provides a `describe` function that outputs the most common statistics about the dataset:

* `count`, which is the number of rows in that column.  

* `mean` and `std`, which contain the mean and standard deviation of the values in each column.

* `min` and `max`, which contain the lowest and highest values in each column.

```python
training_df.describe()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>median_house_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>17000.0</td>
      <td>17000.0</td>
      <td>17000.0</td>
      <td>17000.0</td>
      <td>17000.0</td>
      <td>17000.0</td>
      <td>17000.0</td>
      <td>17000.0</td>
      <td>17000.0</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-119.6</td>
      <td>35.6</td>
      <td>28.6</td>
      <td>2643.7</td>
      <td>539.4</td>
      <td>1429.6</td>
      <td>501.2</td>
      <td>3.9</td>
      <td>207.3</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.0</td>
      <td>2.1</td>
      <td>12.6</td>
      <td>2179.9</td>
      <td>421.5</td>
      <td>1147.9</td>
      <td>384.5</td>
      <td>1.9</td>
      <td>116.0</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-124.3</td>
      <td>32.5</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>0.5</td>
      <td>15.0</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-121.8</td>
      <td>33.9</td>
      <td>18.0</td>
      <td>1462.0</td>
      <td>297.0</td>
      <td>790.0</td>
      <td>282.0</td>
      <td>2.6</td>
      <td>119.4</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-118.5</td>
      <td>34.2</td>
      <td>29.0</td>
      <td>2127.0</td>
      <td>434.0</td>
      <td>1167.0</td>
      <td>409.0</td>
      <td>3.5</td>
      <td>180.4</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>-118.0</td>
      <td>37.7</td>
      <td>37.0</td>
      <td>3151.2</td>
      <td>648.2</td>
      <td>1721.0</td>
      <td>605.2</td>
      <td>4.8</td>
      <td>265.0</td>
    </tr>
    <tr>
      <th>max</th>
      <td>-114.3</td>
      <td>42.0</td>
      <td>52.0</td>
      <td>37937.0</td>
      <td>6445.0</td>
      <td>35682.0</td>
      <td>6082.0</td>
      <td>15.0</td>
      <td>500.0</td>
    </tr>
  </tbody>
</table>
</div>

### Identify anomalies in the dataset

The maximum value (max) of several columns is very
high compared to the other quantities. For example,
example the total_rooms column.

Its better not to use total_rooms as a feature.

## Defining functions that build and train a model

* `build_model(my_learning_rate)`, which builds a randomly-initialized model.
* `train_model(model, feature, label, epochs)`, which trains the model from the feature and label.
  
```python
    """ build the model """
def build_model(my_learning_rate):
  # most simple tf.keras models are sequential.
  model = tf.keras.models.Sequential()


  # The topography of a simple linear regression model
  # is a single node in a single layer.
  model.add(tf.keras.layers.Dense(units=1,
                                  input_shape=(1,)))

  # compiling the model topography into code that TensorFlow can execute
  # configuring training to minimize the model's mean squared error.
  model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=my_learning_rate),
                loss="mean_squared_error",
                metrics=[tf.keras.metrics.RootMeanSquaredError()])

  return model


    """train the model"""
def train_model(model, df, feature, label, epochs, batch_size):

  # input feature and the label
  # The model will train for the specified number of epochs.
  history = model.fit(x=df[feature],
                      y=df[label],
                      batch_size=batch_size,
                      epochs=epochs)

  # gathering the trained model's weight and bias.
  trained_weight = model.get_weights()[0]
  trained_bias = model.get_weights()[1]


  epochs = history.epoch
  
  # isolating the error for each epoch
  hist = pd.DataFrame(history.history)

  # To track the progression of training we're going to take a snapshot
  # of the model's RMS error at each epoch
  rmse = hist["root_mean_squared_error"]

  return trained_weight, trained_bias, epochs, rmse
```

## Defining plotting functions

The following matplotlib functions create the following plots:

* a scatter plot of the feature vs. the label, and a line showing the output of the trained model
* a loss curve

```python
def plot_the_model(trained_weight, trained_bias, feature, label):
  """Plot the trained model against 200 random training examples."""

  plt.xlabel(feature)
  plt.ylabel(label)

  # creating a scatter plot from 200 random points of the dataset.
  random_examples = training_df.sample(n=200)
  plt.scatter(random_examples[feature], random_examples[label])

  # creating a red line representing the model. The red line starts
  # at coordinates (x0, y0) and ends at coordinates (x1, y1).
  x0 = 0
  y0 = trained_bias
  x1 = 10000
  y1 = trained_bias + (trained_weight * x1)
  plt.plot([x0, x1], [y0, y1], c='r')

  plt.show()


def plot_the_loss_curve(epochs, rmse):

  plt.figure()
  plt.xlabel("Epoch")
  plt.ylabel("Root Mean Squared Error")

  plt.plot(epochs, rmse, label="Loss")
  plt.legend()
  plt.ylim([rmse.min()*0.97, rmse.max()])
  plt.show()  
```

## Calling the model functions

An important part of machine learning is determining which features correlate with the label

```python
import warnings
warnings.filterwarnings('ignore')  #to hide warnings
```

```python
# Hyperparameters.
learning_rate = 0.01
epochs = 30
batch_size = 30


# Specifing the feature and the label
my_feature = "total_rooms"  
my_label="median_house_value"

  
# discarding any pre-existing version of the model
my_model = None


# (y = mx + c) or (y = wx + b)
# where w, c are weight and bias
my_model = build_model(learning_rate)
weight, bias, epochs, rmse = train_model(my_model, training_df,
                                         my_feature, my_label,
                                         epochs, batch_size)


print("\nThe learned weight for your model is %.4f" % weight)
print("The learned bias for your model is %.4f\n" % bias )

plot_the_model(weight, bias, my_feature, my_label)
plot_the_loss_curve(epochs, rmse)
```

    Epoch 1/30
    567/567 [==============================] - 1s 887us/step - loss: 36366.2656 - root_mean_squared_error: 190.6994
    Epoch 2/30
    567/567 [==============================] - 0s 882us/step - loss: 27792.7734 - root_mean_squared_error: 166.7117
    Epoch 3/30
    567/567 [==============================] - 0s 863us/step - loss: 26641.8633 - root_mean_squared_error: 163.2234
    Epoch 4/30
    567/567 [==============================] - 0s 870us/step - loss: 26251.0996 - root_mean_squared_error: 162.0219
    Epoch 5/30
    567/567 [==============================] - 0s 864us/step - loss: 25680.3770 - root_mean_squared_error: 160.2510
    Epoch 6/30
    567/567 [==============================] - 1s 898us/step - loss: 25000.4492 - root_mean_squared_error: 158.1153
    Epoch 7/30
    567/567 [==============================] - 1s 923us/step - loss: 24215.3672 - root_mean_squared_error: 155.6129
    Epoch 8/30
    567/567 [==============================] - 0s 872us/step - loss: 23718.8691 - root_mean_squared_error: 154.0093
    Epoch 9/30
    567/567 [==============================] - 0s 819us/step - loss: 23048.5605 - root_mean_squared_error: 151.8175
    Epoch 10/30
    567/567 [==============================] - 0s 816us/step - loss: 22598.2910 - root_mean_squared_error: 150.3273
    Epoch 11/30
    567/567 [==============================] - 0s 821us/step - loss: 21807.9102 - root_mean_squared_error: 147.6750
    Epoch 12/30
    567/567 [==============================] - 0s 820us/step - loss: 21319.5508 - root_mean_squared_error: 146.0122
    Epoch 13/30
    567/567 [==============================] - 0s 814us/step - loss: 20787.4492 - root_mean_squared_error: 144.1785
    Epoch 14/30
    567/567 [==============================] - 0s 822us/step - loss: 20333.2754 - root_mean_squared_error: 142.5948
    Epoch 15/30
    567/567 [==============================] - 0s 820us/step - loss: 20026.3965 - root_mean_squared_error: 141.5146
    Epoch 16/30
    567/567 [==============================] - 0s 809us/step - loss: 19520.8301 - root_mean_squared_error: 139.7170
    Epoch 17/30
    567/567 [==============================] - 0s 826us/step - loss: 18882.7129 - root_mean_squared_error: 137.4144
    Epoch 18/30
    567/567 [==============================] - 0s 826us/step - loss: 18752.4629 - root_mean_squared_error: 136.9396
    Epoch 19/30
    567/567 [==============================] - 0s 818us/step - loss: 18289.2812 - root_mean_squared_error: 135.2379
    Epoch 20/30
    567/567 [==============================] - 0s 829us/step - loss: 17959.8184 - root_mean_squared_error: 134.0143
    Epoch 21/30
    567/567 [==============================] - 0s 813us/step - loss: 17427.4102 - root_mean_squared_error: 132.0129
    Epoch 22/30
    567/567 [==============================] - 0s 823us/step - loss: 17346.3828 - root_mean_squared_error: 131.7057
    Epoch 23/30
    567/567 [==============================] - 0s 833us/step - loss: 16797.7637 - root_mean_squared_error: 129.6062
    Epoch 24/30
    567/567 [==============================] - 0s 821us/step - loss: 16428.5098 - root_mean_squared_error: 128.1738
    Epoch 25/30
    567/567 [==============================] - 0s 824us/step - loss: 16316.2559 - root_mean_squared_error: 127.7351
    Epoch 26/30
    567/567 [==============================] - 0s 831us/step - loss: 16120.5986 - root_mean_squared_error: 126.9669
    Epoch 27/30
    567/567 [==============================] - 0s 828us/step - loss: 15964.0830 - root_mean_squared_error: 126.3491
    Epoch 28/30
    567/567 [==============================] - 0s 823us/step - loss: 15677.7637 - root_mean_squared_error: 125.2109
    Epoch 29/30
    567/567 [==============================] - 0s 815us/step - loss: 15493.4092 - root_mean_squared_error: 124.4725
    Epoch 30/30
    567/567 [==============================] - 0s 820us/step - loss: 15312.2725 - root_mean_squared_error: 123.7428
    
    The learned weight for your model is 0.0306
    The learned bias for your model is 132.3446

![png](/assets/images/notebook/nb4/output_18_1.png)

![png](/assets/images/notebook/nb4/output_18_2.png)

As you can tell, the trained model is not that good. This means that the trained model doesn't have much predictive power. The `total_rooms` is not a good feature.

```python
""" Function to predict label Values """
def predict_house_values(n, feature, label):

  batch = training_df[feature][10000:10000 + n]
  predicted_values = my_model.predict_on_batch(x=batch)

  print("feature   label          predicted")
  print("  value   value          value")
  print("          in thousand   in thousand")
  print("--------------------------------------")
  for i in range(n):
    print ("%5.0f %6.0f %15.0f" % (training_df[feature][10000 + i],
                                   training_df[label][10000 + i],
                                   predicted_values[i][0] ))
```

## Trying a different feature

The `total_rooms` feature had little predictive power. Maybe a different feature have greater predictive power like `total_bedrooms`.

When you change features, you might also need to tweak the hyperparameters.

```python
my_feature = "total_bedrooms"   # change feature here

learning_rate = 0.02
epochs = 30
batch_size = 40


""" train model """
my_model = build_model(learning_rate)
weight, bias, epochs, rmse = train_model(my_model, training_df,
                                         my_feature, my_label,
                                         epochs, batch_size)

print("\nThe learned weight for your model is %.4f" % weight)
print("The learned bias for your model is %.4f\n" % bias )

""" plot """
plot_the_model(weight, bias, my_feature, my_label)
plot_the_loss_curve(epochs, rmse)


""" predict label values """
predict_house_values(15, my_feature, my_label)


```

    Epoch 1/30
    425/425 [==============================] - 0s 610us/step - loss: 69943.5625 - root_mean_squared_error: 264.4684
    Epoch 2/30
    425/425 [==============================] - 0s 621us/step - loss: 27669.8262 - root_mean_squared_error: 166.3425
    Epoch 3/30
    425/425 [==============================] - 0s 615us/step - loss: 26469.1719 - root_mean_squared_error: 162.6935
    Epoch 4/30
    425/425 [==============================] - 0s 617us/step - loss: 25447.7637 - root_mean_squared_error: 159.5236
    Epoch 5/30
    425/425 [==============================] - 0s 617us/step - loss: 24447.3398 - root_mean_squared_error: 156.3564
    Epoch 6/30
    425/425 [==============================] - 0s 616us/step - loss: 23428.7598 - root_mean_squared_error: 153.0646
    Epoch 7/30
    425/425 [==============================] - 0s 625us/step - loss: 22520.5488 - root_mean_squared_error: 150.0685
    Epoch 8/30
    425/425 [==============================] - 0s 618us/step - loss: 21617.3809 - root_mean_squared_error: 147.0285
    Epoch 9/30
    425/425 [==============================] - 0s 607us/step - loss: 20779.3750 - root_mean_squared_error: 144.1505
    Epoch 10/30
    425/425 [==============================] - 0s 613us/step - loss: 20024.9219 - root_mean_squared_error: 141.5094
    Epoch 11/30
    425/425 [==============================] - 0s 610us/step - loss: 19321.2090 - root_mean_squared_error: 139.0007
    Epoch 12/30
    425/425 [==============================] - 0s 607us/step - loss: 18657.0117 - root_mean_squared_error: 136.5907
    Epoch 13/30
    425/425 [==============================] - 0s 617us/step - loss: 17961.7051 - root_mean_squared_error: 134.0213
    Epoch 14/30
    425/425 [==============================] - 0s 616us/step - loss: 17361.1133 - root_mean_squared_error: 131.7616
    Epoch 15/30
    425/425 [==============================] - 0s 601us/step - loss: 16930.2598 - root_mean_squared_error: 130.1163
    Epoch 16/30
    425/425 [==============================] - 0s 606us/step - loss: 16393.5762 - root_mean_squared_error: 128.0374
    Epoch 17/30
    425/425 [==============================] - 0s 619us/step - loss: 16014.4414 - root_mean_squared_error: 126.5482
    Epoch 18/30
    425/425 [==============================] - 0s 612us/step - loss: 15613.5254 - root_mean_squared_error: 124.9541
    Epoch 19/30
    425/425 [==============================] - 0s 603us/step - loss: 15281.8428 - root_mean_squared_error: 123.6198
    Epoch 20/30
    425/425 [==============================] - 0s 614us/step - loss: 14995.5625 - root_mean_squared_error: 122.4564
    Epoch 21/30
    425/425 [==============================] - 0s 609us/step - loss: 14708.7314 - root_mean_squared_error: 121.2796
    Epoch 22/30
    425/425 [==============================] - 0s 620us/step - loss: 14489.1475 - root_mean_squared_error: 120.3709
    Epoch 23/30
    425/425 [==============================] - 0s 621us/step - loss: 14299.5195 - root_mean_squared_error: 119.5806
    Epoch 24/30
    425/425 [==============================] - 0s 611us/step - loss: 14136.0869 - root_mean_squared_error: 118.8953
    Epoch 25/30
    425/425 [==============================] - 0s 607us/step - loss: 14031.5479 - root_mean_squared_error: 118.4548
    Epoch 26/30
    425/425 [==============================] - 0s 600us/step - loss: 13925.2305 - root_mean_squared_error: 118.0052
    Epoch 27/30
    425/425 [==============================] - 0s 617us/step - loss: 13840.8740 - root_mean_squared_error: 117.6472
    Epoch 28/30
    425/425 [==============================] - 0s 616us/step - loss: 13789.2070 - root_mean_squared_error: 117.4275
    Epoch 29/30
    425/425 [==============================] - 0s 612us/step - loss: 13734.6035 - root_mean_squared_error: 117.1947
    Epoch 30/30
    425/425 [==============================] - 0s 613us/step - loss: 13676.8311 - root_mean_squared_error: 116.9480
    
    The learned weight for your model is 0.0228
    The learned bias for your model is 184.1888

![png](/assets/images/notebook/nb4/output_22_1.png)

![png](/assets/images/notebook/nb4/output_22_2.png)

    feature   label          predicted
      value   value          value
              in thousand   in thousand
    --------------------------------------
      393     53             193
      618     92             198
      863     69             204
      471     62             195
      483     80             195
     1313    295             214
      441    500             194
      443    342             194
      282    118             191
      675    128             200
      363    187             192
      166     80             188
     1075    112             209
      741     95             201
      663     69             199

`total_bedrooms` also failed to produce better predictions

## A better way to find features whose values correlate with the label

So far, we've relied on trial-and-error to identify possible features for the model.  Let's rely on statistics.
A **correlation matrix** indicates how each attribute's values relate to the other attribute's values. Correlation values have the following meanings:

* `1.0`: perfect positive correlation; when one attribute rises, the other attribute rises.
* `-1.0`: perfect negative correlation; when one attribute rises, the other attribute falls.
* `0.0`: no correlation; the two column's are not linearly related

In general, the higher the absolute value of a correlation value, the greater its predictive power. For example, a correlation value of -0.8 implies far more predictive power than a correlation of -0.2.

The following code cell generates the correlation matrix for attributes of the Dataset:

```python
training_df.corr()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>median_house_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>longitude</th>
      <td>1.0</td>
      <td>-0.9</td>
      <td>-0.1</td>
      <td>0.0</td>
      <td>0.1</td>
      <td>0.1</td>
      <td>0.1</td>
      <td>-0.0</td>
      <td>-0.0</td>
    </tr>
    <tr>
      <th>latitude</th>
      <td>-0.9</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>-0.0</td>
      <td>-0.1</td>
      <td>-0.1</td>
      <td>-0.1</td>
      <td>-0.1</td>
      <td>-0.1</td>
    </tr>
    <tr>
      <th>housing_median_age</th>
      <td>-0.1</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>-0.4</td>
      <td>-0.3</td>
      <td>-0.3</td>
      <td>-0.3</td>
      <td>-0.1</td>
      <td>0.1</td>
    </tr>
    <tr>
      <th>total_rooms</th>
      <td>0.0</td>
      <td>-0.0</td>
      <td>-0.4</td>
      <td>1.0</td>
      <td>0.9</td>
      <td>0.9</td>
      <td>0.9</td>
      <td>0.2</td>
      <td>0.1</td>
    </tr>
    <tr>
      <th>total_bedrooms</th>
      <td>0.1</td>
      <td>-0.1</td>
      <td>-0.3</td>
      <td>0.9</td>
      <td>1.0</td>
      <td>0.9</td>
      <td>1.0</td>
      <td>-0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>population</th>
      <td>0.1</td>
      <td>-0.1</td>
      <td>-0.3</td>
      <td>0.9</td>
      <td>0.9</td>
      <td>1.0</td>
      <td>0.9</td>
      <td>-0.0</td>
      <td>-0.0</td>
    </tr>
    <tr>
      <th>households</th>
      <td>0.1</td>
      <td>-0.1</td>
      <td>-0.3</td>
      <td>0.9</td>
      <td>1.0</td>
      <td>0.9</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.1</td>
    </tr>
    <tr>
      <th>median_income</th>
      <td>-0.0</td>
      <td>-0.1</td>
      <td>-0.1</td>
      <td>0.2</td>
      <td>-0.0</td>
      <td>-0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.7</td>
    </tr>
    <tr>
      <th>median_house_value</th>
      <td>-0.0</td>
      <td>-0.1</td>
      <td>0.1</td>
      <td>0.1</td>
      <td>0.0</td>
      <td>-0.0</td>
      <td>0.1</td>
      <td>0.7</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>

 The `median_income` correlates 0.7 with the label
 `(median_house_value)`, so `median_income` might be a
 good feature. The other seven potential features
 all have a correlation close to 0.

```python
my_feature = "median_income"


learning_rate = 0.06
epochs = 24
batch_size = 30

my_model = build_model(learning_rate)
weight, bias, epochs, rmse = train_model(my_model, training_df,
                                         my_feature, my_label,
                                         epochs, batch_size)

print("\nThe learned weight for your model is %.4f" % weight)
print("The learned bias for your model is %.4f\n" % bias )

plot_the_model(weight, bias, my_feature, my_label)
plot_the_loss_curve(epochs, rmse)

predict_house_values(15, my_feature, my_label)
```

    Epoch 1/24
    567/567 [==============================] - 0s 811us/step - loss: 25467.0820 - root_mean_squared_error: 159.5841
    Epoch 2/24
    567/567 [==============================] - 0s 802us/step - loss: 7227.4463 - root_mean_squared_error: 85.0144
    Epoch 3/24
    567/567 [==============================] - 0s 811us/step - loss: 7017.9907 - root_mean_squared_error: 83.7734
    Epoch 4/24
    567/567 [==============================] - 0s 812us/step - loss: 7015.0405 - root_mean_squared_error: 83.7558
    Epoch 5/24
    567/567 [==============================] - 0s 801us/step - loss: 7016.5010 - root_mean_squared_error: 83.7646
    Epoch 6/24
    567/567 [==============================] - 0s 799us/step - loss: 7016.2896 - root_mean_squared_error: 83.7633
    Epoch 7/24
    567/567 [==============================] - 0s 819us/step - loss: 7016.9043 - root_mean_squared_error: 83.7670
    Epoch 8/24
    567/567 [==============================] - 0s 812us/step - loss: 7015.9893 - root_mean_squared_error: 83.7615
    Epoch 9/24
    567/567 [==============================] - 0s 804us/step - loss: 7016.6973 - root_mean_squared_error: 83.7657
    Epoch 10/24
    567/567 [==============================] - 0s 805us/step - loss: 7014.6167 - root_mean_squared_error: 83.7533
    Epoch 11/24
    567/567 [==============================] - 0s 810us/step - loss: 7015.4897 - root_mean_squared_error: 83.7585
    Epoch 12/24
    567/567 [==============================] - 0s 809us/step - loss: 7016.2539 - root_mean_squared_error: 83.7631
    Epoch 13/24
    567/567 [==============================] - 0s 795us/step - loss: 7015.9028 - root_mean_squared_error: 83.7610
    Epoch 14/24
    567/567 [==============================] - 0s 816us/step - loss: 7016.2871 - root_mean_squared_error: 83.7633
    Epoch 15/24
    567/567 [==============================] - 0s 803us/step - loss: 7015.3462 - root_mean_squared_error: 83.7577
    Epoch 16/24
    567/567 [==============================] - 0s 810us/step - loss: 7016.3677 - root_mean_squared_error: 83.7638
    Epoch 17/24
    567/567 [==============================] - 0s 794us/step - loss: 7016.2402 - root_mean_squared_error: 83.7630
    Epoch 18/24
    567/567 [==============================] - 0s 803us/step - loss: 7015.5117 - root_mean_squared_error: 83.7587
    Epoch 19/24
    567/567 [==============================] - 0s 815us/step - loss: 7015.4419 - root_mean_squared_error: 83.7582
    Epoch 20/24
    567/567 [==============================] - 0s 816us/step - loss: 7016.5742 - root_mean_squared_error: 83.7650
    Epoch 21/24
    567/567 [==============================] - 0s 810us/step - loss: 7015.1309 - root_mean_squared_error: 83.7564
    Epoch 22/24
    567/567 [==============================] - 0s 804us/step - loss: 7015.5859 - root_mean_squared_error: 83.7591
    Epoch 23/24
    567/567 [==============================] - 0s 797us/step - loss: 7016.6001 - root_mean_squared_error: 83.7652
    Epoch 24/24
    567/567 [==============================] - 0s 812us/step - loss: 7015.5850 - root_mean_squared_error: 83.7591
    
    The learned weight for your model is 42.3674
    The learned bias for your model is 43.0539

![png](/assets/images/notebook/nb4/output_27_1.png)

![png](/assets/images/notebook/nb4/output_27_2.png)

    feature   label          predicted
      value   value          value
              in thousand   in thousand
    --------------------------------------
        2     53             134
        4     92             212
        3     69             154
        2     62             132
        3     80             153
        2    295             148
       10    500             457
        5    342             260
        2    118             147
        4    128             221
        8    187             397
        3     80             163
        3    112             179
        4     95             222
        2     69             143

Based on the loss value, this feature produces a better model than the previous features. However, this feature still is not great at making predictions

Correlation matrices don't tell the entire story. Using `median_income` as a feature may raise some ethical issues
