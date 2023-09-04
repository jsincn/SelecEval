# Loading Raw Files

## Overview
The output of each run contains raw files that can be included in further analytics. 
The following snippets allows the reading of multiple output files and stores them in a single pandas dataframe.

## Loading Validation Results
```python3
import pandas as pd
from os import listdir, getcwd
from os.path import isfile, join
run_folders = ['run_1', 'run_2', 'run_3']
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
df
```


## Loading Training Output
```python3
import pandas as pd
from os import listdir, getcwd
from os.path import isfile, join
import json
run_folders = ['run_1', 'run_2', 'run_3']
dfs = []
for folder in run_folders:
    run_folder = folder + "/client_output/"
    file_list = [f for f in listdir(run_folder) if isfile(join(run_folder, f))]
    print("Loaded", file_list, "from", run_folder)
    for i in range(len(file_list)):
        file_dfs = []
        with open(run_folder + file_list[i]) as f:
            for line in f.readlines():
                json_data = pd.json_normalize(json.loads(line))
                file_dfs.append(json_data)
        file_df = pd.concat(file_dfs)
        file_df["src"] = file_list[i].replace(".json", "")
        file_df["run"] = folder
        dfs.append(file_df)
df = pd.concat(dfs)
```

## Loading Data Distribution
```python3
import pandas as pd
from os import listdir
from os.path import isfile, join
file_list = [f for f in listdir("output") if isfile(join("output", f))]
dfs = []
for i in range(len(file_list)):
    fname = "output/" + file_list[i]
    df_temp = pd.read_csv(fname)
    df_temp.set_index(['client'], inplace=True)
    df_temp['run'] = file_list[i].replace(".csv", "")
    dfs.append(df_temp)
df = pd.concat(dfs).reset_index()
```