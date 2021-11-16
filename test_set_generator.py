import pandas as pd
import numpy as np

data = pd.read_json('Patio_Lawn_and_Garden_5.json', lines=True)
print(data.head())
print(data.info())
data.to_csv('Patio_Lawn_and_Garden_5.csv', index=False)
