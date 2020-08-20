---
layout: posts
title:  "Data Analysis using Python and Matplotlib"
share: true
comments: true
author_profile: false
permalink: /Work/data-analysis-py
header:
  teaser: /assets/images/posts/work/python-data-analysis.jpg
  og_image: /assets/images/posts/work/python-data-analysis.jpg
excerpt: "Data analysis workflow with python and matplotlib"
related: true

---

# Data Analysis using Python and Matplotlib

### About the data

All the datasets are available on [Kaggle](www.kaggle.com) for free.

#### Attribute information
  * Rainfall
  * Temperature

Missing attribute values: none

```python
import pandas as pd #import pandas
import matplotlib.pyplot as plt

#Creating the dataframe using the json file (read_json)
df = pd.read_json(r'rain.json')
```

```python
print(df)
```

            Month  Rainfall  Temperature
    0     January     1.650            7
    1    February     1.250           10
    2       March     1.940           15
    3       April     2.750           20
    4         May     2.750           25
    5        June     3.645           24
    6        July     5.500           30
    7      August     1.000           40
    8   September     1.300           33
    9     October     2.000           20
    10   November     0.500           32
    11   December     2.300           10

```python
print(df.describe())    #This shows the statistics only
```

            Rainfall  Temperature
    count  12.000000    12.000000
    mean    2.215417    22.166667
    std     1.349841    10.408330
    min     0.500000     7.000000
    25%     1.287500    13.750000
    50%     1.970000    22.000000
    75%     2.750000    30.500000
    max     5.500000    40.000000

```python
df.plot(x='Month', y='Temperature', label='Temperature')
df.plot(x='Month', y='Rainfall', label='Rainfall')
plt.show()
```

![png](/assets/images/notebook/nb2/output_5_0.png)

![png](/assets/images/notebook/nb2/output_5_1.png)

# MatplotLib PyPlot

### Using Matplotlib to make plots

```python
plt.plot( df['Month'], df['Temperature'], label = 'Temperature')
plt.show()

```

![png](/assets/images/notebook/nb2/output_8_0.png)

### Fixing overlapping months

```python
plt.figure(figsize=(15,5))   #first number is width, 2nd is height
plt.plot( df['Month'], df['Temperature'], label = 'Temperature')
plt.show()
```

![png](/assets/images/notebook/nb2/output_10_0.png)

```python
#Plot Rainfall
plt.figure(figsize=(17,5))
plt.plot( df['Month'], df['Rainfall'], label = 'Rainfall')
plt.show()
```

![png](/assets/images/notebook/nb2/output_11_0.png)

```python
#Plot Both temperature and Rainfall against Month
plt.figure(figsize=(17,5))
plt.plot( df['Month'], df['Rainfall'], label = 'Rainfall')
plt.plot( df['Month'], df['Temperature'], label = 'Temperature')
plt.legend()   #To show us different colors according to label
plt.show()
```

![png](/assets/images/notebook/nb2/output_12_0.png)

# Scatterplot Graph

```python
import seaborn as sns
```

```python
df = pd.read_csv('data/tempYearly.csv')
```

```python
df.head()
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
      <th>Temperature</th>
      <th>Year</th>
      <th>Rainfall</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>16.99</td>
      <td>1956</td>
      <td>1.01</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10.34</td>
      <td>1957</td>
      <td>1.66</td>
    </tr>
    <tr>
      <th>2</th>
      <td>21.01</td>
      <td>1958</td>
      <td>3.50</td>
    </tr>
    <tr>
      <th>3</th>
      <td>23.68</td>
      <td>1959</td>
      <td>3.31</td>
    </tr>
    <tr>
      <th>4</th>
      <td>24.59</td>
      <td>1960</td>
      <td>3.61</td>
    </tr>
  </tbody>
</table>
</div>

```python
sns.scatterplot(x = 'Year',
               y = 'Temperature',
               data = df)
plt.show()
```

![png](/assets/images/notebook/nb2/output_17_0.png)

```python
sns.set(rc={'figure.figsize':(12,6)})   #To increase width of plot
sns.scatterplot(x = 'Year',
               y = 'Temperature',
               data = df)
plt.show()
```

![png](/assets/images/notebook/nb2/output_18_0.png)

```python
sns.set(rc={'figure.figsize':(15,6)})
sns.regplot(x = 'Year',    #regplot to see regression line
               y = 'Temperature',
               data = df)
plt.show()
```

![png](/assets/images/notebook/nb2/output_19_0.png)

```python
sns.set(rc={'figure.figsize':(15,6)})
sns.regplot(x = 'Rainfall',     #regplot to see regression line with rainfall and temp
               y = 'Temperature',
               data = df)
plt.show()
```

![png](/assets/images/notebook/nb2/output_20_0.png)

Important analysis: The above shows positive effect, so it shows there is a correlation between rainfall and temperature.

# Seaborn Heatmap

To visualize large and small values quickly.

```python
data = pd.read_csv(r'data/birthYearly.csv')
data
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
      <th>month</th>
      <th>year</th>
      <th>births</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>January</td>
      <td>2000</td>
      <td>101</td>
    </tr>
    <tr>
      <th>1</th>
      <td>February</td>
      <td>2000</td>
      <td>168</td>
    </tr>
    <tr>
      <th>2</th>
      <td>March</td>
      <td>2000</td>
      <td>271</td>
    </tr>
    <tr>
      <th>3</th>
      <td>April</td>
      <td>2000</td>
      <td>229</td>
    </tr>
    <tr>
      <th>4</th>
      <td>May</td>
      <td>2000</td>
      <td>287</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>79</th>
      <td>August</td>
      <td>2006</td>
      <td>243</td>
    </tr>
    <tr>
      <th>80</th>
      <td>September</td>
      <td>2006</td>
      <td>174</td>
    </tr>
    <tr>
      <th>81</th>
      <td>October</td>
      <td>2006</td>
      <td>62</td>
    </tr>
    <tr>
      <th>82</th>
      <td>November</td>
      <td>2006</td>
      <td>175</td>
    </tr>
    <tr>
      <th>83</th>
      <td>December</td>
      <td>2006</td>
      <td>134</td>
    </tr>
  </tbody>
</table>
<p>84 rows × 3 columns</p>
</div>

```python
# We want to draw a heatmap but we will get an error because the data is not in correct format for the heat map
# We need to organize the data using pivots
sns.heatmap(data,
           annot = True,    #we want our numbers inside the heatmap
           fmt = "d"      #decimal format data
           )
```

    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-50-92057be0a01a> in <module>
          3 sns.heatmap(data,
          4            annot = True,    #we want our numbers inside the heatmap
    ----> 5            fmt = "d"      #decimal format data
          6            )


    ~/anaconda3/lib/python3.7/site-packages/seaborn/matrix.py in heatmap(data, vmin, vmax, cmap, center, robust, annot, fmt, annot_kws, linewidths, linecolor, cbar, cbar_kws, cbar_ax, square, xticklabels, yticklabels, mask, ax, **kwargs)
        510     plotter = _HeatMapper(data, vmin, vmax, cmap, center, robust, annot, fmt,
        511                           annot_kws, cbar, cbar_kws, xticklabels,
    --> 512                           yticklabels, mask)
        513 
        514     # Add the pcolormesh kwargs here


    ~/anaconda3/lib/python3.7/site-packages/seaborn/matrix.py in __init__(self, data, vmin, vmax, cmap, center, robust, annot, fmt, annot_kws, cbar, cbar_kws, xticklabels, yticklabels, mask)
        164         # Determine good default values for the colormapping
        165         self._determine_cmap_params(plot_data, vmin, vmax,
    --> 166                                     cmap, center, robust)
        167 
        168         # Sort out the annotations


    ~/anaconda3/lib/python3.7/site-packages/seaborn/matrix.py in _determine_cmap_params(self, plot_data, vmin, vmax, cmap, center, robust)
        195                                cmap, center, robust):
        196         """Use some heuristics to set good defaults for colorbar and range."""
    --> 197         calc_data = plot_data.data[~np.isnan(plot_data.data)]
        198         if vmin is None:
        199             vmin = np.percentile(calc_data, 2) if robust else calc_data.min()


    TypeError: ufunc 'isnan' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''

```python
#Pivoting the table
dataP = data.pivot("month", "year", "births")
print(dataP)
```

    year       2000  2001  2002  2003  2004  2005  2006
    month                                              
    April       229    75   289   214   261   115   299
    August      238   162    86    97   202   158   243
    December    152   286   175   142    58   269   134
    February    168    64   152    71   285   242   201
    January     101   291   274    70    84   279   299
    July        142   285   125   194    77   235   193
    June        183   112   257   103   199   204   136
    March       271   289   248    54    70   258   196
    May         287   141   112   258   175   125    70
    November    108    69   131   152    99   230   175
    October     263   148   274   235   157   224    62
    September   264    96   123   104    53   225   174

```python
sns.heatmap(dataP,
           annot = True,
           fmt = "d"
           )
plt.show()
```

![png](/assets/images/notebook/nb2/output_26_0.png)

#### You can quickly draw some conclusions from this heatmap like

1) Very less births in September 2004 and March 2003

2) Alot of births in April 2006

# Seaborn JointPlot

Good for looking at distribution and data points at the same time

```python
data = pd.read_csv(r'data/tempYearly.csv')
data.head()
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
      <th>Temperature</th>
      <th>Year</th>
      <th>Rainfall</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>16.99</td>
      <td>1956</td>
      <td>1.01</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10.34</td>
      <td>1957</td>
      <td>1.66</td>
    </tr>
    <tr>
      <th>2</th>
      <td>21.01</td>
      <td>1958</td>
      <td>3.50</td>
    </tr>
    <tr>
      <th>3</th>
      <td>23.68</td>
      <td>1959</td>
      <td>3.31</td>
    </tr>
    <tr>
      <th>4</th>
      <td>24.59</td>
      <td>1960</td>
      <td>3.61</td>
    </tr>
  </tbody>
</table>
</div>

```python
#We will draw a joint plot which will get scatterplot in middle and histograms on the side
sns.jointplot("Rainfall",
             "Temperature",
              data = data
             )
plt.show()
```

![png](/assets/images/notebook/nb2/output_30_0.png)

In the above jointplot
----------------------------------

Scatterplot can show us outliers, we can imagine a regression line and see the outliers

Histograms can show us the distibution of the data

```python
sns.jointplot("Rainfall",
             "Temperature",
              data = data,
              kind = "hex"   #Hexagram to visualize it abit better
             )
plt.show()
```

![png](/assets/images/notebook/nb2/output_32_0.png)

```python
sns.jointplot("Rainfall",
             "Temperature",
              data = data,
              kind = "reg"   #Regression Line, so we can see a possible positive correlation
             )
plt.show()
```

![png](/assets/images/notebook/nb2/output_33_0.png)

