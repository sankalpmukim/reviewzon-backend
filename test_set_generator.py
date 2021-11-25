import pandas as pd
import numpy as np

data = pd.read_json('Automotive_5.json', lines=True)
print(data.head())
print(data.info())
data.to_csv('Automotive_5.csv', index=False)
