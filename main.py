###Regresssion Model
import pandas
import numpy,sklearn
#Definition
from sklearn.ensemble import DecisionTreeRegressor
house_model = DecisionTreeRegressor(random_state=1)


house_data = pandas.read_csv('melb_data.csv')
print(house_data.head())
house_data=house_data.dropna(axis=0)

#Target
y = house_data.Price

#Features
house_features=['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
x=house_data[house_features]



