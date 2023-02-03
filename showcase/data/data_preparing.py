import pandas as pd
import numpy as np

labels = pd.read_csv("Labels.csv")
df = pd.read_csv("Sequences.csv")

accessions = df.Accession
tmp = accessions.str.rpartition('.')[2]
sequences = df.assign(Accession=tmp)
merged_df = sequences.merge(labels, how='inner', on=['Accession'])
print(merged_df)
df_split = np.array_split(merged_df, 3)

for i in range(3):
    df_split[i].to_csv('sequences_s' + str(i+1) + '.txt', header=False, index=False, sep ='\t')
