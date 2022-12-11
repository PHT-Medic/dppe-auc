# pp-auc
Privacy-preserving AUC computation with PHT-meDIC
Exact privacy preserving AUC computation using the PHT.

Concept of Mete: https://www.overleaf.com/project/5e39693bd5d8de0001399783


## Install requirements
Run `pip install -r requirements.txt` to ensure all requirments are fullfilled.

## Synthetic Data generation
During each run, synthetic data is generated.
To generate sampe data, specify the number of stations and subjects. Afterwards 30-50% of fake subjects are added randomly. Unencrypted data is stored 


## Key creation
Keys are generated and stored for the number of stations and the aggregator within the  `data/keys/` directory

## Execute
1. Define number of subjects and create dataframes and keys.
2. Execute over sample of stations


## Time for tasks

Most trivial implementation with pd.apply() --> vectorizing should increase speed!
With 100 subjects and between 30-50% fake subjects

```/usr/local/bin/python3.7 /Users/mariusherr/ukt/GitHub/pp-auc/code/main.py
Start encrypting table with 140 subjects from station 1
Encryption time 6.9425 seconds
Start encrypting table with 136 subjects from station 2
Encryption time 6.6865 seconds
Start encrypting table with 131 subjects from station 3
Encryption time 6.4835 seconds```

With 10000 subjects and between 30-50% fake subjects
```

And with 1000 subjects and same distribution of fake subjects:
```/usr/local/bin/python3.7 /Users/mariusherr/ukt/GitHub/pp-auc/code/main.py
Start encrypting table with 1471 subjects from station 1
Encryption time 76.0529 seconds
Start encrypting table with 1360 subjects from station 2
Encryption time 68.7387 seconds
Start encrypting table with 1464 subjects from station 3
Encryption time 71.8119 seconds
```



Removed parts: 
TP = []
    FP = []

    TP.insert(0, encrypt(agg_pk, 0))
    FP.insert(0, encrypt(agg_pk, 0))

    for i in range(1, length+ 1):
        TP.insert(i - 1, e_add(agg_pk, TP[i-1], dataframe['Label'][i-1]))
        FP.insert(i - 1, e_add(agg_pk, FP[i-1], add_const(agg_pk, mul_const(agg_pk, dataframe['Label'][i-1], -1), 1)))
