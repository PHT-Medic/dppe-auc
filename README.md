# pp-auc
Privacy-preserving AUC computation with SMPC protocol


Exact privacy preserving AUC computation for the PHT1.0

Concept of Mete: https://www.overleaf.com/project/5e39693bd5d8de0001399783

# Next steps
- Improve implemtation performance
- Continue to follow protocol and finalize code
- Include functionalities in train-container-library
- Test on real PHT Setup

## Install requirements
Run `pip install -r requirements.txt` to ensure all requirments are fullfilled.

## Data generation
To generate sampe data, specify the number of stations and subjects. Afterwards 30-50% of fake subjects are added randomly. Unencrypted data is stored 


## Key creation
Keys are generated and stored for the number of stations and the aggregator under `data/keys/`

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



