import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle

df = pd.read_csv('data.csv')
y = df['math score']
X = df.drop('math score', axis=1)
X = pd.get_dummies(X)

model = RandomForestRegressor()
model.fit(X, y)

pickle.dump(model, open('model.pkl', 'wb'))
print('Model trained and saved!')
