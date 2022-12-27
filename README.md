# DPPE-AUC
Distributed Privacy-Preserving exact - area under the curve; a novel method to compute the exact global AUC without revealing individual sensitive input data. It utilizes a combination of Paillier, symmetric, and asymmetric encryption with perfect randomized encoding to compute the exact measurement even with tie conditions. 
## Install requirements
Run `pip install -r requirements.txt` to ensure all requirements are fulfilled.

## Synthetic Data generation
Three different experiments are used to measure the performance.
To generate sample data, specify the number of stations and subjects. Afterwards 30-50% of fake subjects are added randomly.

## Experiments
The performance is evaluated against the commonly standard sklearn AUC library.

### Varying number of input parties
TODO- add plots

### Varying number of input samples
TODO- add plots
