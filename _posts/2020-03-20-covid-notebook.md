---
layout: posts
title:  "Covid Data Analysis using Python"
share: true
comments: true
author_profile: false
permalink: /Work/covid-notebook
header:
  teaser: /assets/images/posts/work/covidpost.jpg
  og_image: /assets/images/posts/work/covidpost.jpge
excerpt: "Data analysis, visualization of covid using python."
related: true

---

# Covid19 Data Analysis Notebook

<img src="https://www.fda.gov/files/covid19-1600x900.jpg">

------------------------------------------

### About the Dataset

The [European CDC](https://ourworldindata.org/coronavirus-source-data) publishes daily statistics on the COVID-19 pandemic. Not just for Europe, but for the entire world. We rely on the ECDC as they collect and harmonize data from around the world which allows us to compare what is happening in different countries. The European CDC data provides a global perspective on the evolving pandemic.

The European Centre for Disease Prevention and Control ECDC provides three statistical resources on the COVID-19 pandemic:

* COVID-19 Dashboard
* Situation reports
* The daily data tables
The ECDC makes all their data available in a daily updated clean downloadable file. This gets updated daily reflecting data collected up to 6:00 and 10:00 CET.

The European CDC collects and aggregates data from countries around the world. The most up-to-date data for any particular country is therefore typically available earlier via the national health agencies than via the ECDC. This lag between nationally available data and the ECDC data is not very long as the ECDC publishes new data daily. But it can be several hours.

### Import the modules

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
```

### Importing covid19 dataset

```python
confirmed_cases = pd.read_csv('data/covid19_Confirmed_dataset.csv', header=0, parse_dates=True)
confirmed_cases.head()
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
      <th>Province/State</th>
      <th>Country/Region</th>
      <th>Lat</th>
      <th>Long</th>
      <th>1/22/20</th>
      <th>1/23/20</th>
      <th>1/24/20</th>
      <th>1/25/20</th>
      <th>1/26/20</th>
      <th>1/27/20</th>
      <th>...</th>
      <th>4/21/20</th>
      <th>4/22/20</th>
      <th>4/23/20</th>
      <th>4/24/20</th>
      <th>4/25/20</th>
      <th>4/26/20</th>
      <th>4/27/20</th>
      <th>4/28/20</th>
      <th>4/29/20</th>
      <th>4/30/20</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>Afghanistan</td>
      <td>33.0000</td>
      <td>65.0000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1092</td>
      <td>1176</td>
      <td>1279</td>
      <td>1351</td>
      <td>1463</td>
      <td>1531</td>
      <td>1703</td>
      <td>1828</td>
      <td>1939</td>
      <td>2171</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>Albania</td>
      <td>41.1533</td>
      <td>20.1683</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>609</td>
      <td>634</td>
      <td>663</td>
      <td>678</td>
      <td>712</td>
      <td>726</td>
      <td>736</td>
      <td>750</td>
      <td>766</td>
      <td>773</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>Algeria</td>
      <td>28.0339</td>
      <td>1.6596</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>2811</td>
      <td>2910</td>
      <td>3007</td>
      <td>3127</td>
      <td>3256</td>
      <td>3382</td>
      <td>3517</td>
      <td>3649</td>
      <td>3848</td>
      <td>4006</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>Andorra</td>
      <td>42.5063</td>
      <td>1.5218</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>717</td>
      <td>723</td>
      <td>723</td>
      <td>731</td>
      <td>738</td>
      <td>738</td>
      <td>743</td>
      <td>743</td>
      <td>743</td>
      <td>745</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>Angola</td>
      <td>-11.2027</td>
      <td>17.8739</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>24</td>
      <td>25</td>
      <td>25</td>
      <td>25</td>
      <td>25</td>
      <td>26</td>
      <td>27</td>
      <td>27</td>
      <td>27</td>
      <td>27</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 104 columns</p>
</div>

#### Let's check the shape of the dataframe

```python
confirmed_cases.shape # 266 rows and 104 columns
```

    (266, 104)

### Dropping the unwanted columns

```python
col = confirmed_cases.columns  
print(col)  #Latitude and Longitude are not important features for us here
```

    Index(['Province/State', 'Country/Region', 'Lat', 'Long', '1/22/20', '1/23/20',
           '1/24/20', '1/25/20', '1/26/20', '1/27/20',
           ...
           '4/21/20', '4/22/20', '4/23/20', '4/24/20', '4/25/20', '4/26/20',
           '4/27/20', '4/28/20', '4/29/20', '4/30/20'],
          dtype='object', length=104)

```python
column_to_drop = ['Lat', 'Long', 'Province/State']
confirmed_cases.drop(column_to_drop, axis=1,
    inplace = True)  # will change the dataset too if True
```

### Aggregating the rows by the country

```python
covid_data_grouped = confirmed_cases.groupby('Country/Region').sum()
covid_data_grouped.head()

```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    <pn>
     .dataframe thead th {
    </pn>
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1/22/20</th>
      <th>1/23/20</th>
      <th>1/24/20</th>
      <th>1/25/20</th>
      <th>1/26/20</th>
      <th>1/27/20</th>
      <th>1/28/20</th>
      <th>1/29/20</th>
      <th>1/30/20</th>
      <th>1/31/20</th>
      <th>...</th>
      <th>4/21/20</th>
      <th>4/22/20</th>
      <th>4/23/20</th>
      <th>4/24/20</th>
      <th>4/25/20</th>
      <th>4/26/20</th>
      <th>4/27/20</th>
      <th>4/28/20</th>
      <th>4/29/20</th>
      <th>4/30/20</th>
    </tr>
    <tr>
      <th>Country/Region</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Afghanistan</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1092</td>
      <td>1176</td>
      <td>1279</td>
      <td>1351</td>
      <td>1463</td>
      <td>1531</td>
      <td>1703</td>
      <td>1828</td>
      <td>1939</td>
      <td>2171</td>
    </tr>
    <tr>
      <th>Albania</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>609</td>
      <td>634</td>
      <td>663</td>
      <td>678</td>
      <td>712</td>
      <td>726</td>
      <td>736</td>
      <td>750</td>
      <td>766</td>
      <td>773</td>
    </tr>
    <tr>
      <th>Algeria</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>2811</td>
      <td>2910</td>
      <td>3007</td>
      <td>3127</td>
      <td>3256</td>
      <td>3382</td>
      <td>3517</td>
      <td>3649</td>
      <td>3848</td>
      <td>4006</td>
    </tr>
    <tr>
      <th>Andorra</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>717</td>
      <td>723</td>
      <td>723</td>
      <td>731</td>
      <td>738</td>
      <td>738</td>
      <td>743</td>
      <td>743</td>
      <td>743</td>
      <td>745</td>
    </tr>
    <tr>
      <th>Angola</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>24</td>
      <td>25</td>
      <td>25</td>
      <td>25</td>
      <td>25</td>
      <td>26</td>
      <td>27</td>
      <td>27</td>
      <td>27</td>
      <td>27</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 100 columns</p>
</div>

```python
df = covid_data_grouped
df.shape # We now have 187 Countries and 100 dates
```

    (187, 100)

```python
df.filter(regex='India',axis = 0).head() # filter the column that you're looking for
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
      <th>1/22/20</th>
      <th>1/23/20</th>
      <th>1/24/20</th>
      <th>1/25/20</th>
      <th>1/26/20</th>
      <th>1/27/20</th>
      <th>1/28/20</th>
      <th>1/29/20</th>
      <th>1/30/20</th>
      <th>1/31/20</th>
      <th>...</th>
      <th>4/21/20</th>
      <th>4/22/20</th>
      <th>4/23/20</th>
      <th>4/24/20</th>
      <th>4/25/20</th>
      <th>4/26/20</th>
      <th>4/27/20</th>
      <th>4/28/20</th>
      <th>4/29/20</th>
      <th>4/30/20</th>
    </tr>
    <tr>
      <th>Country/Region</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>India</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>20080</td>
      <td>21370</td>
      <td>23077</td>
      <td>24530</td>
      <td>26283</td>
      <td>27890</td>
      <td>29451</td>
      <td>31324</td>
      <td>33062</td>
      <td>34863</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 100 columns</p>
</div>

### Visualizing data related to a country for example India

```python
df.loc['India'].plot(figsize=(15,5))
```

    <matplotlib.axes._subplots.AxesSubplot at 0x7fa30db166d0>

![png](/assets/images/notebook/nb3/output_16_1.png)

```python
df.loc['US'].plot()
df.loc['Brazil'].plot()
df.loc['India'].plot()
df.loc['Italy'].plot()
plt.legend()

```

    <matplotlib.legend.Legend at 0x7fa30d2c9310>

![png](/assets/images/notebook/nb3/output_17_1.png)

```python
#Spread of the virus in India for the first 20 dates only
df.loc['India'][:20].plot()
```

    <matplotlib.axes._subplots.AxesSubplot at 0x7fa30d264f10>

![png](/assets/images/notebook/nb3/output_18_1.png)

```python
df.loc['India'][:20].plot()
df.loc['US'][:20].plot()
df.loc['China'][:20].plot()
df.loc['Italy'][:20].plot()
plt.legend()
```

    <matplotlib.legend.Legend at 0x7fa30d1f0ed0>

![png](/assets/images/notebook/nb3/output_19_1.png)

### Caculating the first derivative of the curve

```python
df.loc['India'].diff().plot() #We want to find a measure for new cases, so either say average or maximum number of new cases
```

    <matplotlib.axes._subplots.AxesSubplot at 0x7fa30d155250>

![png](/assets/images/notebook/nb3/output_21_1.png)

### Find maxmimum infection rate for India

```python
df.loc['India'].diff().max()
```

    1893.0

```python
df.loc['US'].diff().max()
```

    36188.0

```python
df.loc['Italy'].diff().max()
```

    6557.0

```python
df.loc['United Kingdom'].diff().max()
```

    8733.0

### Find maximum infection rate for all of the countries

```python
countries = list(df.index)
max_infection_rates = []
for c in countries :
    max_infection_rates.append(df.loc[c].diff().max())
max_infection_rates
```

    [232.0,
     34.0,
     199.0,
     43.0,
     5.0,
     6.0,
     291.0,
     134.0,
     497.0,
     1321.0,
     105.0,
     7.0,
     301.0,
     641.0,
     12.0,
     1485.0,
     2454.0,
     4.0,
     19.0,
     1.0,
     104.0,
     92.0,
     7.0,
     7502.0,
     26.0,
     137.0,
     41.0,
     21.0,
     6.0,
     45.0,
     31.0,
     203.0,
     2778.0,
     31.0,
     21.0,
     1138.0,
     15136.0,
     353.0,
     1.0,
     57.0,
     81.0,
     37.0,
     113.0,
     96.0,
     63.0,
     58.0,
     381.0,
     391.0,
     99.0,
     156.0,
     5.0,
     371.0,
     11536.0,
     269.0,
     32.0,
     130.0,
     7.0,
     134.0,
     20.0,
     9.0,
     5.0,
     267.0,
     26849.0,
     38.0,
     5.0,
     42.0,
     6933.0,
     403.0,
     156.0,
     6.0,
     68.0,
     167.0,
     132.0,
     12.0,
     10.0,
     3.0,
     72.0,
     210.0,
     99.0,
     1893.0,
     436.0,
     3186.0,
     91.0,
     1515.0,
     1131.0,
     6557.0,
     52.0,
     1161.0,
     40.0,
     264.0,
     29.0,
     851.0,
     289.0,
     300.0,
     69.0,
     3.0,
     48.0,
     61.0,
     17.0,
     13.0,
     21.0,
     90.0,
     234.0,
     7.0,
     14.0,
     10.0,
     235.0,
     190.0,
     58.0,
     52.0,
     2.0,
     41.0,
     1425.0,
     222.0,
     12.0,
     13.0,
     30.0,
     281.0,
     19.0,
     3.0,
     14.0,
     1346.0,
     89.0,
     2.0,
     69.0,
     208.0,
     107.0,
     386.0,
     144.0,
     1292.0,
     357.0,
     5.0,
     27.0,
     3683.0,
     538.0,
     545.0,
     1516.0,
     957.0,
     523.0,
     7099.0,
     22.0,
     5.0,
     6.0,
     4.0,
     54.0,
     6.0,
     1351.0,
     87.0,
     2379.0,
     2.0,
     20.0,
     1426.0,
     114.0,
     70.0,
     73.0,
     354.0,
     28.0,
     9630.0,
     65.0,
     67.0,
     3.0,
     812.0,
     1321.0,
     6.0,
     27.0,
     15.0,
     181.0,
     188.0,
     10.0,
     14.0,
     40.0,
     82.0,
     5138.0,
     36188.0,
     11.0,
     578.0,
     552.0,
     8733.0,
     48.0,
     167.0,
     29.0,
     19.0,
     66.0,
     4.0,
     5.0,
     9.0,
     8.0]

```python
df['max_infection_rates'] = max_infection_rates
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
      <th>1/22/20</th>
      <th>1/23/20</th>
      <th>1/24/20</th>
      <th>1/25/20</th>
      <th>1/26/20</th>
      <th>1/27/20</th>
      <th>1/28/20</th>
      <th>1/29/20</th>
      <th>1/30/20</th>
      <th>1/31/20</th>
      <th>...</th>
      <th>4/22/20</th>
      <th>4/23/20</th>
      <th>4/24/20</th>
      <th>4/25/20</th>
      <th>4/26/20</th>
      <th>4/27/20</th>
      <th>4/28/20</th>
      <th>4/29/20</th>
      <th>4/30/20</th>
      <th>max_infection_rates</th>
    </tr>
    <tr>
      <th>Country/Region</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Afghanistan</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1176</td>
      <td>1279</td>
      <td>1351</td>
      <td>1463</td>
      <td>1531</td>
      <td>1703</td>
      <td>1828</td>
      <td>1939</td>
      <td>2171</td>
      <td>232.0</td>
    </tr>
    <tr>
      <th>Albania</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>634</td>
      <td>663</td>
      <td>678</td>
      <td>712</td>
      <td>726</td>
      <td>736</td>
      <td>750</td>
      <td>766</td>
      <td>773</td>
      <td>34.0</td>
    </tr>
    <tr>
      <th>Algeria</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>2910</td>
      <td>3007</td>
      <td>3127</td>
      <td>3256</td>
      <td>3382</td>
      <td>3517</td>
      <td>3649</td>
      <td>3848</td>
      <td>4006</td>
      <td>199.0</td>
    </tr>
    <tr>
      <th>Andorra</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>723</td>
      <td>723</td>
      <td>731</td>
      <td>738</td>
      <td>738</td>
      <td>743</td>
      <td>743</td>
      <td>743</td>
      <td>745</td>
      <td>43.0</td>
    </tr>
    <tr>
      <th>Angola</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>25</td>
      <td>25</td>
      <td>25</td>
      <td>25</td>
      <td>26</td>
      <td>27</td>
      <td>27</td>
      <td>27</td>
      <td>27</td>
      <td>5.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 101 columns</p>
</div>

### Creating a new dataframe with only needed column

```python
covid_data = pd.DataFrame(df['max_infection_rates'])
```

```python
covid_data.head()
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
      <th>max_infection_rates</th>
    </tr>
    <tr>
      <th>Country/Region</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Afghanistan</th>
      <td>232.0</td>
    </tr>
    <tr>
      <th>Albania</th>
      <td>34.0</td>
    </tr>
    <tr>
      <th>Algeria</th>
      <td>199.0</td>
    </tr>
    <tr>
      <th>Andorra</th>
      <td>43.0</td>
    </tr>
    <tr>
      <th>Angola</th>
      <td>5.0</td>
    </tr>
  </tbody>
</table>
</div>

###

- Importing the WorldHappinessReport.csv dataset
- selecting needed columns for our analysis
- join the datasets
- calculate the correlations as the result of our analysis

### Importing the dataset

```python
happy_data = pd.read_csv('data/worldwide_happiness_report.csv')
```

```python
happy_data.head()
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
      <th>Overall rank</th>
      <th>Country or region</th>
      <th>Score</th>
      <th>GDP per capita</th>
      <th>Social support</th>
      <th>Healthy life expectancy</th>
      <th>Freedom to make life choices</th>
      <th>Generosity</th>
      <th>Perceptions of corruption</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Finland</td>
      <td>7.769</td>
      <td>1.340</td>
      <td>1.587</td>
      <td>0.986</td>
      <td>0.596</td>
      <td>0.153</td>
      <td>0.393</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Denmark</td>
      <td>7.600</td>
      <td>1.383</td>
      <td>1.573</td>
      <td>0.996</td>
      <td>0.592</td>
      <td>0.252</td>
      <td>0.410</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Norway</td>
      <td>7.554</td>
      <td>1.488</td>
      <td>1.582</td>
      <td>1.028</td>
      <td>0.603</td>
      <td>0.271</td>
      <td>0.341</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Iceland</td>
      <td>7.494</td>
      <td>1.380</td>
      <td>1.624</td>
      <td>1.026</td>
      <td>0.591</td>
      <td>0.354</td>
      <td>0.118</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Netherlands</td>
      <td>7.488</td>
      <td>1.396</td>
      <td>1.522</td>
      <td>0.999</td>
      <td>0.557</td>
      <td>0.322</td>
      <td>0.298</td>
    </tr>
  </tbody>
</table>
</div>

### let's drop the useless columns

```python
cols = ["Overall rank", "Score", "Generosity", "Perceptions of corruption"]
```

```python
happy_data.drop(cols, axis = 1, inplace = True)
happy_data.head()
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
      <th>Country or region</th>
      <th>GDP per capita</th>
      <th>Social support</th>
      <th>Healthy life expectancy</th>
      <th>Freedom to make life choices</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Finland</td>
      <td>1.340</td>
      <td>1.587</td>
      <td>0.986</td>
      <td>0.596</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Denmark</td>
      <td>1.383</td>
      <td>1.573</td>
      <td>0.996</td>
      <td>0.592</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Norway</td>
      <td>1.488</td>
      <td>1.582</td>
      <td>1.028</td>
      <td>0.603</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Iceland</td>
      <td>1.380</td>
      <td>1.624</td>
      <td>1.026</td>
      <td>0.591</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Netherlands</td>
      <td>1.396</td>
      <td>1.522</td>
      <td>0.999</td>
      <td>0.557</td>
    </tr>
  </tbody>
</table>
</div>

### Changing the indices of the dataframe

```python
happy_data.set_index('Country or region', inplace = True)
happy_data.head()
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
      <th>GDP per capita</th>
      <th>Social support</th>
      <th>Healthy life expectancy</th>
      <th>Freedom to make life choices</th>
    </tr>
    <tr>
      <th>Country or region</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Finland</th>
      <td>1.340</td>
      <td>1.587</td>
      <td>0.986</td>
      <td>0.596</td>
    </tr>
    <tr>
      <th>Denmark</th>
      <td>1.383</td>
      <td>1.573</td>
      <td>0.996</td>
      <td>0.592</td>
    </tr>
    <tr>
      <th>Norway</th>
      <td>1.488</td>
      <td>1.582</td>
      <td>1.028</td>
      <td>0.603</td>
    </tr>
    <tr>
      <th>Iceland</th>
      <td>1.380</td>
      <td>1.624</td>
      <td>1.026</td>
      <td>0.591</td>
    </tr>
    <tr>
      <th>Netherlands</th>
      <td>1.396</td>
      <td>1.522</td>
      <td>0.999</td>
      <td>0.557</td>
    </tr>
  </tbody>
</table>
</div>

### Now let's join two dataset

#### Corona Dataset

```python
covid_data.shape    # 187 rows and 1 column
```

    (187, 1)

#### World happiness report Dataset

```python
happy_data.shape  # 156 rows and 4 columns
```

    (156, 4)

```python
data = covid_data.join(happy_data, how = 'inner')
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
      <th>max_infection_rates</th>
      <th>GDP per capita</th>
      <th>Social support</th>
      <th>Healthy life expectancy</th>
      <th>Freedom to make life choices</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Afghanistan</th>
      <td>232.0</td>
      <td>0.350</td>
      <td>0.517</td>
      <td>0.361</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>Albania</th>
      <td>34.0</td>
      <td>0.947</td>
      <td>0.848</td>
      <td>0.874</td>
      <td>0.383</td>
    </tr>
    <tr>
      <th>Algeria</th>
      <td>199.0</td>
      <td>1.002</td>
      <td>1.160</td>
      <td>0.785</td>
      <td>0.086</td>
    </tr>
    <tr>
      <th>Argentina</th>
      <td>291.0</td>
      <td>1.092</td>
      <td>1.432</td>
      <td>0.881</td>
      <td>0.471</td>
    </tr>
    <tr>
      <th>Armenia</th>
      <td>134.0</td>
      <td>0.850</td>
      <td>1.055</td>
      <td>0.815</td>
      <td>0.283</td>
    </tr>
  </tbody>
</table>
</div>

```python
data.count().isnull()
```

    max_infection_rates             False
    GDP per capita                  False
    Social support                  False
    Healthy life expectancy         False
    Freedom to make life choices    False
    dtype: bool

### Creating a correlation matrix

```python
corr = data.corr
print(corr)  # There is positive correlation between max_infection-rate and all other features
```

    <bound method DataFrame.corr of              max_infection_rates  GDP per capita  Social support  \
    Afghanistan                232.0           0.350           0.517   
    Albania                     34.0           0.947           0.848   
    Algeria                    199.0           1.002           1.160   
    Argentina                  291.0           1.092           1.432   
    Armenia                    134.0           0.850           1.055   
    ...                          ...             ...             ...   
    Venezuela                   29.0           0.960           1.427   
    Vietnam                     19.0           0.741           1.346   
    Yemen                        5.0           0.287           1.163   
    Zambia                       9.0           0.578           1.058   
    Zimbabwe                     8.0           0.366           1.114   
    
                 Healthy life expectancy  Freedom to make life choices  
    Afghanistan                    0.361                         0.000  
    Albania                        0.874                         0.383  
    Algeria                        0.785                         0.086  
    Argentina                      0.881                         0.471  
    Armenia                        0.815                         0.283  
    ...                              ...                           ...  
    Venezuela                      0.805                         0.154  
    Vietnam                        0.851                         0.543  
    Yemen                          0.463                         0.143  
    Zambia                         0.426                         0.431  
    Zimbabwe                       0.433                         0.361  
    
    [143 rows x 5 columns]>

### Visualizating the results

```python
x = data['GDP per capita']
y = data['Healthy life expectancy']
sns.regplot(x,y)  #GDP per capita is a good representation of a country's standard of living
```

    <matplotlib.axes._subplots.AxesSubplot at 0x7fa30d0884d0>

![png](/assets/images/notebook/nb3/output_52_1.png)

```python
a = data['GDP per capita']
b = data['max_infection_rates']
sns.regplot(a, b) # There is no positive correlation between GDP per capita and max infection rates of countries
                    # Although we could scale max_infection_rates but there should be no correlation between the two
```

    <matplotlib.axes._subplots.AxesSubplot at 0x7fa3067c7a50>

![png](/assets/images/notebook/nb3/output_53_1.png)

```python
n = data['max_infection_rates']
m = data['Healthy life expectancy']
sns.regplot(m,np.log(n))  # scaling max infection rates and there seems to be a postive correlation between the two
```

    <matplotlib.axes._subplots.AxesSubplot at 0x7fa306747e50>

![png](/assets/images/notebook/nb3/output_54_1.png)
