import pandas
import numpy,sklearn

house_data = pandas.read_csv('melb_data.csv')
print(house_data.head())
house_data=house_data.dropna(axis=0)

#Target
y = house_data.Price

#Features
house_features=['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
x=house_data[house_features]

#Definition
from sklearn.tree import DecisionTreeRegressor
house_model = DecisionTreeRegressor(random_state=1)

#fit
house_model.fit(x,y)

#Predict
print("The prediction are :")
print(house_model.predict(x))
numpy.savetxt('model_prediction.csv',house_model.predict(x.head(5)))
