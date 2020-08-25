---
layout: posts
share: true
comments: true
author_profile: false
permalink: /Work/classifcation
header:
  teaser: /assets/images/posts/work/class.png
  og_image: /assets/images/posts/work/class.png
excerpt: "Binary Classification with class imbalance dataset"
related: true

---

# Binary Classification

<img src="https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2020/01/Scatter-Plot-of-Binary-Classification-Dataset.png">

Creating and evaluating a binary classification model.

------------------------------------------

## Objectives

* Convert a regression question into a classification question.
* Modifing the classification threshold and determine how that modification influences the model
* Experiment with different classification metrics to determine model's effectiveness.

## The Dataset

Like the previous notebook, this notebook uses the California Housing Dataset.

The following code imports the modules

```python
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import feature_column
from tensorflow.keras import layers
from matplotlib import pyplot as plt

# The following lines adjust the granularity
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format
tf.keras.backend.set_floatx('float32')

```

## Importing the datasets

```python
train_df = pd.read_csv("data/california_housing_train.csv")
test_df = pd.read_csv("data/california_housing_test.csv")
train_df = train_df.reindex(np.random.permutation(train_df.index)) # shuffling the training set
```

```python
train_df
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
      <th>5079</th>
      <td>-118.1</td>
      <td>34.0</td>
      <td>38.0</td>
      <td>1301.0</td>
      <td>264.0</td>
      <td>877.0</td>
      <td>275.0</td>
      <td>4.6</td>
      <td>191300.0</td>
    </tr>
    <tr>
      <th>16101</th>
      <td>-122.5</td>
      <td>37.9</td>
      <td>35.0</td>
      <td>2492.0</td>
      <td>409.0</td>
      <td>812.0</td>
      <td>373.0</td>
      <td>8.8</td>
      <td>500001.0</td>
    </tr>
    <tr>
      <th>10085</th>
      <td>-119.8</td>
      <td>34.4</td>
      <td>22.0</td>
      <td>2845.0</td>
      <td>500.0</td>
      <td>1456.0</td>
      <td>454.0</td>
      <td>5.7</td>
      <td>276400.0</td>
    </tr>
    <tr>
      <th>3177</th>
      <td>-117.8</td>
      <td>33.8</td>
      <td>26.0</td>
      <td>2110.0</td>
      <td>409.0</td>
      <td>1146.0</td>
      <td>407.0</td>
      <td>4.4</td>
      <td>229600.0</td>
    </tr>
    <tr>
      <th>9714</th>
      <td>-119.6</td>
      <td>36.6</td>
      <td>42.0</td>
      <td>2311.0</td>
      <td>439.0</td>
      <td>1347.0</td>
      <td>436.0</td>
      <td>2.6</td>
      <td>69700.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3240</th>
      <td>-117.9</td>
      <td>33.9</td>
      <td>25.0</td>
      <td>3205.0</td>
      <td>409.0</td>
      <td>1291.0</td>
      <td>408.0</td>
      <td>7.2</td>
      <td>299200.0</td>
    </tr>
    <tr>
      <th>5402</th>
      <td>-118.2</td>
      <td>33.9</td>
      <td>36.0</td>
      <td>1435.0</td>
      <td>249.0</td>
      <td>606.0</td>
      <td>234.0</td>
      <td>4.1</td>
      <td>212600.0</td>
    </tr>
    <tr>
      <th>11192</th>
      <td>-121.0</td>
      <td>37.7</td>
      <td>18.0</td>
      <td>5129.0</td>
      <td>1171.0</td>
      <td>3622.0</td>
      <td>1128.0</td>
      <td>2.0</td>
      <td>92700.0</td>
    </tr>
    <tr>
      <th>3007</th>
      <td>-117.8</td>
      <td>33.7</td>
      <td>6.0</td>
      <td>1593.0</td>
      <td>371.0</td>
      <td>832.0</td>
      <td>379.0</td>
      <td>4.4</td>
      <td>239500.0</td>
    </tr>
    <tr>
      <th>445</th>
      <td>-117.0</td>
      <td>34.0</td>
      <td>12.0</td>
      <td>5876.0</td>
      <td>1222.0</td>
      <td>2992.0</td>
      <td>1151.0</td>
      <td>2.4</td>
      <td>112100.0</td>
    </tr>
  </tbody>
</table>
<p>17000 rows × 9 columns</p>
</div>

```python
train_df.describe()
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
      <td>207300.9</td>
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
      <td>115983.8</td>
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
      <td>14999.0</td>
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
      <td>119400.0</td>
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
      <td>180400.0</td>
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
      <td>265000.0</td>
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
      <td>500001.0</td>
    </tr>
  </tbody>
</table>
</div>

## Normalizing values

When creating a model with multiple features, the values of each feature should have roughly the same range.

A **Z-score** is the number of standard deviations from the mean for a particular value.

* mean is 60
* standard deviation is 10

The value 75 would have a Z-score of:

```
  Z-score = (75 - 60) / 10 = +1.5
```

```python
# Calculating the Z-scores of each column in the training set

train_df_mean = train_df.mean()
train_df_std = train_df.std()
train_df_norm = (train_df - train_df_mean)/train_df_std


train_df_norm
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
      <th>5079</th>
      <td>0.7</td>
      <td>-0.8</td>
      <td>0.7</td>
      <td>-0.6</td>
      <td>-0.7</td>
      <td>-0.5</td>
      <td>-0.6</td>
      <td>0.4</td>
      <td>-0.1</td>
    </tr>
    <tr>
      <th>16101</th>
      <td>-1.4</td>
      <td>1.1</td>
      <td>0.5</td>
      <td>-0.1</td>
      <td>-0.3</td>
      <td>-0.5</td>
      <td>-0.3</td>
      <td>2.6</td>
      <td>2.5</td>
    </tr>
    <tr>
      <th>10085</th>
      <td>-0.1</td>
      <td>-0.6</td>
      <td>-0.5</td>
      <td>0.1</td>
      <td>-0.1</td>
      <td>0.0</td>
      <td>-0.1</td>
      <td>0.9</td>
      <td>0.6</td>
    </tr>
    <tr>
      <th>3177</th>
      <td>0.9</td>
      <td>-0.9</td>
      <td>-0.2</td>
      <td>-0.2</td>
      <td>-0.3</td>
      <td>-0.2</td>
      <td>-0.2</td>
      <td>0.3</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>9714</th>
      <td>-0.0</td>
      <td>0.4</td>
      <td>1.1</td>
      <td>-0.2</td>
      <td>-0.2</td>
      <td>-0.1</td>
      <td>-0.2</td>
      <td>-0.7</td>
      <td>-1.2</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3240</th>
      <td>0.8</td>
      <td>-0.8</td>
      <td>-0.3</td>
      <td>0.3</td>
      <td>-0.3</td>
      <td>-0.1</td>
      <td>-0.2</td>
      <td>1.8</td>
      <td>0.8</td>
    </tr>
    <tr>
      <th>5402</th>
      <td>0.7</td>
      <td>-0.8</td>
      <td>0.6</td>
      <td>-0.6</td>
      <td>-0.7</td>
      <td>-0.7</td>
      <td>-0.7</td>
      <td>0.1</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>11192</th>
      <td>-0.7</td>
      <td>1.0</td>
      <td>-0.8</td>
      <td>1.1</td>
      <td>1.5</td>
      <td>1.9</td>
      <td>1.6</td>
      <td>-1.0</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>3007</th>
      <td>0.9</td>
      <td>-0.9</td>
      <td>-1.8</td>
      <td>-0.5</td>
      <td>-0.4</td>
      <td>-0.5</td>
      <td>-0.3</td>
      <td>0.3</td>
      <td>0.3</td>
    </tr>
    <tr>
      <th>445</th>
      <td>1.3</td>
      <td>-0.8</td>
      <td>-1.3</td>
      <td>1.5</td>
      <td>1.6</td>
      <td>1.4</td>
      <td>1.7</td>
      <td>-0.8</td>
      <td>-0.8</td>
    </tr>
  </tbody>
</table>
<p>17000 rows × 9 columns</p>
</div>

```python
# Calculating the Z-scores of each column in the test set

test_df_mean = test_df.mean()
test_df_std  = test_df.std()
test_df_norm = (test_df - test_df_mean)/test_df_std

test_df_norm
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
      <td>-1.2</td>
      <td>0.8</td>
      <td>-0.1</td>
      <td>0.6</td>
      <td>0.3</td>
      <td>0.1</td>
      <td>0.3</td>
      <td>1.5</td>
      <td>1.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.6</td>
      <td>-0.6</td>
      <td>1.1</td>
      <td>-0.5</td>
      <td>-0.5</td>
      <td>-0.6</td>
      <td>-0.6</td>
      <td>-0.1</td>
      <td>-0.3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.9</td>
      <td>-0.9</td>
      <td>-0.1</td>
      <td>0.5</td>
      <td>-0.1</td>
      <td>0.1</td>
      <td>0.0</td>
      <td>1.1</td>
      <td>0.6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.6</td>
      <td>-0.9</td>
      <td>-0.1</td>
      <td>-1.2</td>
      <td>-1.2</td>
      <td>-1.3</td>
      <td>-1.3</td>
      <td>1.3</td>
      <td>1.1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.0</td>
      <td>0.3</td>
      <td>-0.8</td>
      <td>-0.6</td>
      <td>-0.7</td>
      <td>-0.5</td>
      <td>-0.7</td>
      <td>-0.5</td>
      <td>-1.1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2995</th>
      <td>-0.1</td>
      <td>-0.6</td>
      <td>-0.5</td>
      <td>-0.5</td>
      <td>0.3</td>
      <td>-0.1</td>
      <td>0.3</td>
      <td>-1.4</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>2996</th>
      <td>0.7</td>
      <td>-0.7</td>
      <td>-0.1</td>
      <td>1.2</td>
      <td>1.3</td>
      <td>2.0</td>
      <td>1.5</td>
      <td>-0.2</td>
      <td>0.3</td>
    </tr>
    <tr>
      <th>2997</th>
      <td>-0.1</td>
      <td>0.3</td>
      <td>-1.5</td>
      <td>-0.8</td>
      <td>-0.8</td>
      <td>-0.7</td>
      <td>-0.7</td>
      <td>-0.8</td>
      <td>-1.3</td>
    </tr>
    <tr>
      <th>2998</th>
      <td>1.2</td>
      <td>-0.7</td>
      <td>0.9</td>
      <td>-1.2</td>
      <td>-1.2</td>
      <td>-1.3</td>
      <td>-1.3</td>
      <td>-0.3</td>
      <td>-0.4</td>
    </tr>
    <tr>
      <th>2999</th>
      <td>-0.0</td>
      <td>-0.6</td>
      <td>1.0</td>
      <td>-0.4</td>
      <td>-0.6</td>
      <td>-0.6</td>
      <td>-0.6</td>
      <td>2.6</td>
      <td>2.6</td>
    </tr>
  </tbody>
</table>
<p>3000 rows × 9 columns</p>
</div>

## Creating a binary label

In classification, the label for every example must be binary. Tthe label in the California housing dataset `median_house_value`, contains floating-point values like 80,100 or 85,700 rather than 0s and 1s.

So we create a new column named `median_house_value_is_high` in both the training set and the test set. If the `median_house_value` is higher than a certain threshold, then we set `median_house_value_is_high` to 1. Otherwise, set `median_house_value_is_high` to 0.

```python
threshold = 265000 # 75% of median house values
train_df_norm["median_house_value_is_high"] = (train_df["median_house_value"] > threshold).astype(float)
test_df_norm["median_house_value_is_high"] =  (test_df["median_house_value"] > threshold).astype(float)

# astype(float) converts True and False to 1s and 0s

train_df_norm["median_house_value_is_high"]
```

    5079    0.0
    16101   1.0
    10085   1.0
    3177    0.0
    9714    0.0
             ..
    3240    1.0
    5402    0.0
    11192   0.0
    3007    0.0
    445     0.0
    Name: median_house_value_is_high, Length: 17000, dtype: float64

## Representing features in feature columns

```python
feature_columns = []

# numerical feature column to represent median_income.
median_income = tf.feature_column.numeric_column("median_income")
feature_columns.append(median_income)

# numerical feature column to represent total_rooms.
tr = tf.feature_column.numeric_column("total_rooms")
feature_columns.append(tr)

# feature columns into layers
feature_layer = layers.DenseFeatures(feature_columns)


feature_layer(dict(train_df_norm))
```

    <tf.Tensor: shape=(17000, 2), dtype=float32, numpy=
    array([[ 0.38855404, -0.6159161 ],
           [ 2.5967586 , -0.06957252],
           [ 0.931172  ,  0.09235802],
           ...,
           [-0.9728647 ,  1.1400899 ],
           [ 0.28562745, -0.48196787],
           [-0.760618  ,  1.4827588 ]], dtype=float32)>

## Defining functions that build and train model

* `create_model(my_learning_rate, feature_layer, my_metrics)`
* `train_model(model, dataset, epochs, label_name, batch_size, shuffle)`z

and using sigmoid as the activation function

```python
"""Creating a simple classification model"""
def create_model(my_learning_rate, feature_layer, my_metrics):

  model = tf.keras.models.Sequential()

  
  model.add(feature_layer)

  # sigmoid function.
  model.add(tf.keras.layers.Dense(units=1, input_shape=(1,),
                                  activation=tf.sigmoid),)

  # we will use a different loss function for classification
  model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=my_learning_rate),
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=my_metrics)

  return model



"""Feed a dataset into the model in order to train it."""
def train_model(model, dataset, epochs, label_name,
                batch_size=None, shuffle=True):
  

  
  features = {name:np.array(value) for name, value in dataset.items()}
  label = np.array(features.pop(label_name))
  history = model.fit(x=features, y=label, batch_size=batch_size,
                      epochs=epochs, shuffle=shuffle)
  
  
  epochs = history.epoch

  # isolating the classification metric for each epoch
  hist = pd.DataFrame(history.history)

  return epochs, hist
```

## Defining a plotting function

Shows how various classification metrics change with each epoch.

```python
 """Ploting a curve of one or more classification metrics vs epoch"""  
def plot_curve(epochs, hist, list_of_metrics):

  plt.figure()
  plt.xlabel("Epoch")
  plt.ylabel("Value")

  for m in list_of_metrics:
    x = hist[m]
    plt.plot(epochs[1:], x[1:], label=m)

  plt.legend()

```

## Invoking create, train, and plot function

```python
import warnings
warnings.filterwarnings('ignore')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

#suppressing warnings
```

```python
learning_rate = 0.001
epochs = 20
batch_size = 100
label_name = "median_house_value_is_high"
classification_threshold = 0.35

# metrics the model will measure
METRICS = [
           tf.keras.metrics.BinaryAccuracy(name='accuracy',
                                           threshold=classification_threshold),
           tf.keras.metrics.Recall(name='recall',
                                   thresholds=classification_threshold ),
           tf.keras.metrics.Precision(name='precision',
                                      thresholds=classification_threshold),
          ]


my_model = create_model(learning_rate, feature_layer, METRICS)


epochs, hist = train_model(my_model, train_df_norm, epochs,
                           label_name, batch_size)

# Plot a graph of the metrics vs epochs
list_of_metrics_to_plot = ['accuracy', 'precision', 'recall']

plot_curve(epochs, hist, list_of_metrics_to_plot)
```

    Epoch 1/20
    170/170 [==============================] - 0s 1ms/step - loss: 0.6526 - accuracy: 0.5022 - recall: 0.8333 - precision: 0.3134
    Epoch 2/20
    170/170 [==============================] - 0s 1ms/step - loss: 0.5842 - accuracy: 0.5825 - recall: 0.8126 - precision: 0.3539
    Epoch 3/20
    170/170 [==============================] - 0s 2ms/step - loss: 0.5307 - accuracy: 0.6465 - recall: 0.7853 - precision: 0.3955
    Epoch 4/20
    170/170 [==============================] - 0s 1ms/step - loss: 0.4896 - accuracy: 0.6930 - recall: 0.7617 - precision: 0.4347
    Epoch 5/20
    170/170 [==============================] - 0s 2ms/step - loss: 0.4578 - accuracy: 0.7266 - recall: 0.7311 - precision: 0.4697
    Epoch 6/20
    170/170 [==============================] - 0s 1ms/step - loss: 0.4343 - accuracy: 0.7509 - recall: 0.7026 - precision: 0.5011
    Epoch 7/20
    170/170 [==============================] - 0s 1ms/step - loss: 0.4172 - accuracy: 0.7742 - recall: 0.6798 - precision: 0.5380
    Epoch 8/20
    170/170 [==============================] - 0s 1ms/step - loss: 0.4065 - accuracy: 0.7891 - recall: 0.6574 - precision: 0.5672
    Epoch 9/20
    170/170 [==============================] - 0s 1ms/step - loss: 0.4007 - accuracy: 0.7988 - recall: 0.6419 - precision: 0.5893
    Epoch 10/20
    170/170 [==============================] - 0s 1ms/step - loss: 0.3981 - accuracy: 0.8035 - recall: 0.6339 - precision: 0.6012
    Epoch 11/20
    170/170 [==============================] - 0s 1ms/step - loss: 0.3971 - accuracy: 0.8071 - recall: 0.6266 - precision: 0.6110
    Epoch 12/20
    170/170 [==============================] - 0s 1ms/step - loss: 0.3968 - accuracy: 0.8075 - recall: 0.6207 - precision: 0.6133
    Epoch 13/20
    170/170 [==============================] - 0s 1ms/step - loss: 0.3967 - accuracy: 0.8080 - recall: 0.6174 - precision: 0.6153
    Epoch 14/20
    170/170 [==============================] - 0s 1ms/step - loss: 0.3966 - accuracy: 0.8087 - recall: 0.6150 - precision: 0.6176
    Epoch 15/20
    170/170 [==============================] - 0s 1ms/step - loss: 0.3966 - accuracy: 0.8090 - recall: 0.6150 - precision: 0.6184
    Epoch 16/20
    170/170 [==============================] - 0s 1ms/step - loss: 0.3966 - accuracy: 0.8091 - recall: 0.6141 - precision: 0.6189
    Epoch 17/20
    170/170 [==============================] - 0s 1ms/step - loss: 0.3966 - accuracy: 0.8095 - recall: 0.6124 - precision: 0.6203
    Epoch 18/20
    170/170 [==============================] - 0s 1ms/step - loss: 0.3966 - accuracy: 0.8090 - recall: 0.6138 - precision: 0.6187
    Epoch 19/20
    170/170 [==============================] - 0s 1ms/step - loss: 0.3966 - accuracy: 0.8095 - recall: 0.6146 - precision: 0.6198
    Epoch 20/20
    170/170 [==============================] - 0s 1ms/step - loss: 0.3966 - accuracy: 0.8091 - recall: 0.6138 - precision: 0.6189

![png](/assets/images/notebook/nb5/output_22_1.png)

## Evaluate the model against the test set

```python
features = {name:np.array(value) for name, value in test_df_norm.items()}
label = np.array(features.pop(label_name))

my_model.evaluate(x = features, y = label, batch_size=batch_size)
```

    30/30 [==============================] - 0s 1ms/step - loss: 0.4069 - accuracy: 0.8010 - recall: 0.5965 - precision: 0.6005





    [0.4068722724914551,
     0.8009999990463257,
     0.5965147614479065,
     0.6005398035049438]

#### A model that always guesses median_house_value_is_high is False would be 75% accurate. Our model makes 80% accurate predictions, which is not that great

## Adjusting the classification threshold

```python
learning_rate = 0.001
epochs = 20
batch_size = 100
label_name = "median_house_value_is_high"
classification_threshold = 0.60

# metrics the model will measure
METRICS = [
           tf.keras.metrics.BinaryAccuracy(name='accuracy',
                                           threshold=classification_threshold),
           tf.keras.metrics.Recall(name='recall',
                                   thresholds=classification_threshold ),
           tf.keras.metrics.Precision(name='precision',
                                      thresholds=classification_threshold),
          ]


my_model = create_model(learning_rate, feature_layer, METRICS)


epochs, hist = train_model(my_model, train_df_norm, epochs,
                           label_name, batch_size)

# Plot a graph of the metrics vs epochs
list_of_metrics_to_plot = ['accuracy', 'precision', 'recall']

plot_curve(epochs, hist, list_of_metrics_to_plot)
```

    Epoch 1/20
    170/170 [==============================] - 0s 2ms/step - loss: 0.6549 - accuracy: 0.7381 - recall: 0.3847 - precision: 0.4705
    Epoch 2/20
    170/170 [==============================] - 0s 1ms/step - loss: 0.5889 - accuracy: 0.7679 - recall: 0.3911 - precision: 0.5500
    Epoch 3/20
    170/170 [==============================] - 0s 1ms/step - loss: 0.5332 - accuracy: 0.7886 - recall: 0.3927 - precision: 0.6219
    Epoch 4/20
    170/170 [==============================] - 0s 1ms/step - loss: 0.4890 - accuracy: 0.8038 - recall: 0.3904 - precision: 0.6894
    Epoch 5/20
    170/170 [==============================] - 0s 1ms/step - loss: 0.4551 - accuracy: 0.8174 - recall: 0.3935 - precision: 0.7595
    Epoch 6/20
    170/170 [==============================] - 0s 1ms/step - loss: 0.4309 - accuracy: 0.8244 - recall: 0.3909 - precision: 0.8066
    Epoch 7/20
    170/170 [==============================] - 0s 1ms/step - loss: 0.4151 - accuracy: 0.8287 - recall: 0.3850 - precision: 0.8450
    Epoch 8/20
    170/170 [==============================] - 0s 1ms/step - loss: 0.4062 - accuracy: 0.8314 - recall: 0.3838 - precision: 0.8675
    Epoch 9/20
    170/170 [==============================] - 0s 1ms/step - loss: 0.4014 - accuracy: 0.8323 - recall: 0.3862 - precision: 0.8705
    Epoch 10/20
    170/170 [==============================] - 0s 1ms/step - loss: 0.3989 - accuracy: 0.8325 - recall: 0.3899 - precision: 0.8657
    Epoch 11/20
    170/170 [==============================] - 0s 1ms/step - loss: 0.3978 - accuracy: 0.8330 - recall: 0.3984 - precision: 0.8563
    Epoch 12/20
    170/170 [==============================] - 0s 1ms/step - loss: 0.3972 - accuracy: 0.8322 - recall: 0.3986 - precision: 0.8499
    Epoch 13/20
    170/170 [==============================] - 0s 1ms/step - loss: 0.3969 - accuracy: 0.8329 - recall: 0.4036 - precision: 0.8477
    Epoch 14/20
    170/170 [==============================] - 0s 1ms/step - loss: 0.3968 - accuracy: 0.8331 - recall: 0.4048 - precision: 0.8476
    Epoch 15/20
    170/170 [==============================] - 0s 1ms/step - loss: 0.3967 - accuracy: 0.8332 - recall: 0.4059 - precision: 0.8468
    Epoch 16/20
    170/170 [==============================] - 0s 1ms/step - loss: 0.3967 - accuracy: 0.8335 - recall: 0.4092 - precision: 0.8437
    Epoch 17/20
    170/170 [==============================] - 0s 1ms/step - loss: 0.3966 - accuracy: 0.8332 - recall: 0.4069 - precision: 0.8454
    Epoch 18/20
    170/170 [==============================] - 0s 1ms/step - loss: 0.3966 - accuracy: 0.8334 - recall: 0.4083 - precision: 0.8442
    Epoch 19/20
    170/170 [==============================] - 0s 1ms/step - loss: 0.3966 - accuracy: 0.8334 - recall: 0.4099 - precision: 0.8423
    Epoch 20/20
    170/170 [==============================] - 0s 1ms/step - loss: 0.3966 - accuracy: 0.8332 - recall: 0.4078 - precision: 0.8436

![png](/assets/images/notebook/nb5/output_27_1.png)

#### Classification_threshold of 0.60 gives the highest accuracy (83.3%) and pecision (84.4%)

## Summarizing model performance with AUC

```python
learning_rate = 0.001
epochs = 20
batch_size = 100
label_name = "median_house_value_is_high"

METRICS = [
      tf.keras.metrics.AUC(num_thresholds=100, name='auc'),
]


my_model = create_model(learning_rate, feature_layer, METRICS)


epochs, hist = train_model(my_model, train_df_norm, epochs,
                           label_name, batch_size)

# Plot metrics vs epochs
list_of_metrics_to_plot = ['auc']
plot_curve(epochs, hist, list_of_metrics_to_plot)
```

    Epoch 1/20
    170/170 [==============================] - 0s 1ms/step - loss: 0.8612 - auc: 0.1643
    Epoch 2/20
    170/170 [==============================] - 0s 1ms/step - loss: 0.7613 - auc: 0.1791
    Epoch 3/20
    170/170 [==============================] - 0s 1ms/step - loss: 0.6796 - auc: 0.2272
    Epoch 4/20
    170/170 [==============================] - 0s 1ms/step - loss: 0.6133 - auc: 0.3605
    Epoch 5/20
    170/170 [==============================] - 0s 1ms/step - loss: 0.5598 - auc: 0.6874
    Epoch 6/20
    170/170 [==============================] - 0s 1ms/step - loss: 0.5179 - auc: 0.7951
    Epoch 7/20
    170/170 [==============================] - 0s 1ms/step - loss: 0.4856 - auc: 0.8194
    Epoch 8/20
    170/170 [==============================] - 0s 1ms/step - loss: 0.4610 - auc: 0.8277
    Epoch 9/20
    170/170 [==============================] - 0s 1ms/step - loss: 0.4421 - auc: 0.8321
    Epoch 10/20
    170/170 [==============================] - 0s 1ms/step - loss: 0.4275 - auc: 0.8341
    Epoch 11/20
    170/170 [==============================] - 0s 1ms/step - loss: 0.4168 - auc: 0.8352
    Epoch 12/20
    170/170 [==============================] - 0s 1ms/step - loss: 0.4089 - auc: 0.8358
    Epoch 13/20
    170/170 [==============================] - 0s 1ms/step - loss: 0.4036 - auc: 0.8362
    Epoch 14/20
    170/170 [==============================] - 0s 1ms/step - loss: 0.4004 - auc: 0.8363
    Epoch 15/20
    170/170 [==============================] - 0s 1ms/step - loss: 0.3986 - auc: 0.8369
    Epoch 16/20
    170/170 [==============================] - 0s 1ms/step - loss: 0.3977 - auc: 0.8367
    Epoch 17/20
    170/170 [==============================] - 0s 1ms/step - loss: 0.3972 - auc: 0.8367
    Epoch 18/20
    170/170 [==============================] - 0s 1ms/step - loss: 0.3969 - auc: 0.8368
    Epoch 19/20
    170/170 [==============================] - 0s 1ms/step - loss: 0.3968 - auc: 0.8367
    Epoch 20/20
    170/170 [==============================] - 0s 1ms/step - loss: 0.3967 - auc: 0.8369

![png](/assets/images/notebook/nb5/output_30_1.png)

```python

```
