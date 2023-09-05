# Accuracy Metrics

## Imports


```python
import pandas as pd
from os import listdir, getcwd
from os.path import isfile, join
```

## Data import

This notebook can be used to compute and average results across multiple runs, simply adapt the run_folders.


```python
run_folders = ['run_1']
```


```python
dfs = []
for folder in run_folders:
    run_folder = folder + "/validation/"
    file_list = [f for f in listdir(run_folder) if isfile(join(run_folder, f))]
    print("Loaded", file_list, "from", run_folder)
    for i in range(len(file_list)):
        fname = run_folder + file_list[i]
        df_temp = pd.read_csv(fname)
        df_temp["src"] = file_list[i].replace(".csv", "")
        df_temp["run"] = folder
        dfs.append(df_temp)
df = pd.concat(dfs)
```

    Loaded ['ActiveFL.csv', 'CEP.csv', 'Random.csv', 'FedCS.csv', 'PowD.csv'] from run_1/validation/


## Preview of the imported data


```python
df.sample(5)
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
      <th>round</th>
      <th>client</th>
      <th>loss</th>
      <th>acc</th>
      <th>total</th>
      <th>correct</th>
      <th>class_accuracy</th>
      <th>src</th>
      <th>run</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>12199</th>
      <td>24</td>
      <td>humongous-rating</td>
      <td>0.031195</td>
      <td>0.725490</td>
      <td>51</td>
      <td>37</td>
      <td>[0.8571428656578064, 1.0, 0.5, 0.4000000059604...</td>
      <td>Random</td>
      <td>run_1</td>
    </tr>
    <tr>
      <th>4852</th>
      <td>9</td>
      <td>seething-tambourine</td>
      <td>0.056867</td>
      <td>0.450980</td>
      <td>51</td>
      <td>23</td>
      <td>[0.3333333432674408, 1.0, 1.0, 0.3333333432674...</td>
      <td>CEP</td>
      <td>run_1</td>
    </tr>
    <tr>
      <th>9348</th>
      <td>18</td>
      <td>square-lien</td>
      <td>0.042600</td>
      <td>0.568627</td>
      <td>51</td>
      <td>29</td>
      <td>[0.8333333134651184, 0.875, 1.0, 0.0, 0.333333...</td>
      <td>Random</td>
      <td>run_1</td>
    </tr>
    <tr>
      <th>8055</th>
      <td>16</td>
      <td>kinetic-kern</td>
      <td>0.035176</td>
      <td>0.745098</td>
      <td>51</td>
      <td>38</td>
      <td>[0.6666666865348816, 1.0, 0.800000011920929, 0...</td>
      <td>PowD</td>
      <td>run_1</td>
    </tr>
    <tr>
      <th>3443</th>
      <td>6</td>
      <td>few-body</td>
      <td>0.052351</td>
      <td>0.509804</td>
      <td>51</td>
      <td>26</td>
      <td>[0.75, 0.8333333134651184, 0.6666666865348816,...</td>
      <td>FedCS</td>
      <td>run_1</td>
    </tr>
  </tbody>
</table>
</div>



## Generate Mean, Min and Max Accuracy after the final round

### Aggregation per Run


```python
df_gen = df[df['round'] == max(df['round'])][['src', 'run', 'acc']]
df_gen_max = df_gen.groupby(['src', 'run']).max()
df_gen_max['op'] = 'Top-1'
df_gen_mean = df_gen.groupby(['src', 'run']).mean()
df_gen_mean['op'] = 'Mean'
df_gen_min = df_gen.groupby(['src', 'run']).min()
df_gen_min['op'] = 'Bottom-1'
df_gen = pd.concat([df_gen_max, df_gen_mean, df_gen_min])
df_gen = df_gen.pivot(columns="op")
df_gen
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th></th>
      <th colspan="3" halign="left">acc</th>
    </tr>
    <tr>
      <th></th>
      <th>op</th>
      <th>Bottom-1</th>
      <th>Mean</th>
      <th>Top-1</th>
    </tr>
    <tr>
      <th>src</th>
      <th>run</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ActiveFL</th>
      <th>run_1</th>
      <td>0.568627</td>
      <td>0.735216</td>
      <td>0.901961</td>
    </tr>
    <tr>
      <th>CEP</th>
      <th>run_1</th>
      <td>0.588235</td>
      <td>0.743725</td>
      <td>0.901961</td>
    </tr>
    <tr>
      <th>FedCS</th>
      <th>run_1</th>
      <td>0.549020</td>
      <td>0.740392</td>
      <td>0.921569</td>
    </tr>
    <tr>
      <th>PowD</th>
      <th>run_1</th>
      <td>0.568627</td>
      <td>0.731216</td>
      <td>0.882353</td>
    </tr>
    <tr>
      <th>Random</th>
      <th>run_1</th>
      <td>0.568627</td>
      <td>0.740000</td>
      <td>0.901961</td>
    </tr>
  </tbody>
</table>
</div>



### Final Aggregation


```python
df_output = df_gen.groupby('src').mean()
df_output
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="3" halign="left">acc</th>
    </tr>
    <tr>
      <th>op</th>
      <th>Bottom-1</th>
      <th>Mean</th>
      <th>Top-1</th>
    </tr>
    <tr>
      <th>src</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ActiveFL</th>
      <td>0.568627</td>
      <td>0.735216</td>
      <td>0.901961</td>
    </tr>
    <tr>
      <th>CEP</th>
      <td>0.588235</td>
      <td>0.743725</td>
      <td>0.901961</td>
    </tr>
    <tr>
      <th>FedCS</th>
      <td>0.549020</td>
      <td>0.740392</td>
      <td>0.921569</td>
    </tr>
    <tr>
      <th>PowD</th>
      <td>0.568627</td>
      <td>0.731216</td>
      <td>0.882353</td>
    </tr>
    <tr>
      <th>Random</th>
      <td>0.568627</td>
      <td>0.740000</td>
      <td>0.901961</td>
    </tr>
  </tbody>
</table>
</div>



## Time to Accuracy

### Generate Time to Accuracy (Mean)


```python
df_gen = df[['round', 'src', 'run', 'acc']]
accuracy_steps = [0.2,0.3,0.4,0.5,0.6,0.7,0.8]
df_res = []
for algorithm in df['src'].unique():
    result = {}
    for step in accuracy_steps:
        df_tmp = df_gen[df_gen['src'] == algorithm][['round', 'acc', 'run']].groupby(['round', 'run']).mean().reset_index()
        df_tmp = df_tmp[df_tmp['acc'] >= step]
        if len(df_tmp) == 0:
            result[step] = '-'
        else:
            df_temp = df_tmp[['round', 'run']].groupby('run').min().reset_index()
            result[step] = df_temp['round'].mean()
    df_res.append(pd.DataFrame(result,index=[algorithm]))
df_res = pd.concat(df_res)
df_res.columns = ["{:.2f}".format(x) for x in df_res.columns]
df_res
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
      <th>0.20</th>
      <th>0.30</th>
      <th>0.40</th>
      <th>0.50</th>
      <th>0.60</th>
      <th>0.70</th>
      <th>0.80</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ActiveFL</th>
      <td>4.0</td>
      <td>5.0</td>
      <td>7.0</td>
      <td>9.0</td>
      <td>13.0</td>
      <td>25.0</td>
      <td>-</td>
    </tr>
    <tr>
      <th>CEP</th>
      <td>4.0</td>
      <td>5.0</td>
      <td>7.0</td>
      <td>9.0</td>
      <td>14.0</td>
      <td>23.0</td>
      <td>-</td>
    </tr>
    <tr>
      <th>Random</th>
      <td>3.0</td>
      <td>4.0</td>
      <td>6.0</td>
      <td>7.0</td>
      <td>12.0</td>
      <td>22.0</td>
      <td>-</td>
    </tr>
    <tr>
      <th>FedCS</th>
      <td>3.0</td>
      <td>5.0</td>
      <td>6.0</td>
      <td>9.0</td>
      <td>13.0</td>
      <td>23.0</td>
      <td>-</td>
    </tr>
    <tr>
      <th>PowD</th>
      <td>4.0</td>
      <td>4.0</td>
      <td>6.0</td>
      <td>8.0</td>
      <td>14.0</td>
      <td>24.0</td>
      <td>-</td>
    </tr>
  </tbody>
</table>
</div>


