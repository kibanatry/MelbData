###DECISION TREE

import pandas
import numpy,sklearn

#For model validation 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
########################################################

house_data = pandas.read_csv('melb_data.csv')
print(house_data.head())
house_data=house_data.dropna(axis=0)

#Target
y = house_data.Price

#Features
house_features=['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
x=house_data[house_features]

#To split the files into train and test
train_x,test_x,train_y,test_y=train_test_split(x,y)
####################################################

#Definition
from sklearn.tree import DecisionTreeRegressor
house_model = DecisionTreeRegressor(random_state=1)

#To check over and underfitting#
def get_mae(max_leaf_node,train_x,test_x,train_y,test_y):
  model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
  house_model.fit(train_x, train_y)
  preds_val = house_model.predict(test_x)
  mae = mean_absolute_error(test_y, preds_val)
  return(mae)

##########################################
# compare MAE with differing values of max_leaf_nodes
for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_x, test_x, train_y, test_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))


#fit
#house_model.fit(train_x,train_y)

#Predict
#print("The prediction are :")
#prediction_value=(house_model.predict(test_x))
#print(house_model.predict(test_x))
#numpy.savetxt('model_prediction.csv',house_model.predict(test_x))
#print(mean_absolute_error(test_y,prediction_value))

############################################################