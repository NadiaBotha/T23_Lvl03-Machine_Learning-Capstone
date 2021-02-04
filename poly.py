#Import packages
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

#Data was obtained from - https://www.cdc.gov/growthcharts/html_charts/wtage.htm#males.

#Define the x and y datasets for the training (will be used in the actual model). It has to be in a matrix format for linearRegression and PolynomialFeatures.
x_train = [[120.5], [126.5], [132.5], [144.5], [150.5], [156.5], [168.5], [174.5], [180.5], [192.5], [198.5], [204.5], [216.5], [222.5], [234.5], [240]] 
y_train = [[24.19264], [25.34731], [26.59626], [29.47257], [31.13865], [32.96852], [37.07331], [39.28212], [41.5275], [45.79301], [47.66815], [49.28662], [51.69086], [52.53731], [53.74501], [54.00982]] 

#Define the x and y datasets for the testing (will not be used for the model, just as a check).
x_test = [[138.5], [162.5], [186.5], [210.5], [228.5]] 
y_test = [[27.9639], [34.95475], [43.71882], [50.62293], [53.21739]]

#---------------------------------------------------------LINEAR PLOT--------------------------------------------------------------------------------
#Define the linearRegression function by calling the LinearRegression() method.
linear_reg_function = LinearRegression()
#Pass the x and y data to the linear regression function.
linear_reg_function.fit(x_train, y_train)
#Create an array for all the x-values. This is based on the x-data. The lowest value is 120.5 and highest is 240, and we want to obtain increments of 12.
#The upper value has been changed from 240 tot 270 to allow for some prediction.
x_axis = np.linspace(120, 270, 300)
#Calculate/predict the y values for the x_axis array values.
y_axis = linear_reg_function.predict(x_axis.reshape(x_axis.shape[0], 1))
#Plot the prediction
plt.plot(x_axis, y_axis)

#--------------------------------------------------------QUADRATIC PLOT------------------------------------------------------------------------------
#Use the PolynomialFeatures method to transform all of the linear data to quadratic data.
#It creates a function with the highest power of 2.
quad_tranformation_object = PolynomialFeatures(degree=2)

#Tranform the data which has a degree of 1 to a degree of 2, which is suitable for a quadratic function.
x_train_quadratic = quad_tranformation_object.fit_transform(x_train)
x_test_quadratic = quad_tranformation_object.transform(x_test)

#Push the quadratic data to the linearRegression method.
quad_reg_function = LinearRegression()
quad_reg_function.fit(x_train_quadratic, y_train)
#Adjust the x_axis for quadratic data.
quad_x_axis = quad_tranformation_object.transform(x_axis.reshape(x_axis.shape[0], 1))


#---------------------------------------------------CUBIC PLOT-----------------------------------------------------------------------------
# The same logic applies as with the quadratic plot, the transformation function has just been set to accomodate for a degree of 3.
cubic_tranformation_object = PolynomialFeatures(degree=3)

x_train_quadratic = cubic_tranformation_object.fit_transform(x_train)
x_test_quadratic = cubic_tranformation_object.transform(x_test)

cubic_reg_function = LinearRegression()
cubic_reg_function.fit(x_train_quadratic, y_train)
cubic_x_axis = cubic_tranformation_object.transform(x_axis.reshape(x_axis.shape[0], 1))

plt.plot(x_axis, quad_reg_function.predict(quad_x_axis), c='green', linestyle='dotted')
plt.plot(x_axis, cubic_reg_function.predict(cubic_x_axis), c='r', linestyle='--')
plt.axis([120, 270, 24, 65])
plt.title('Weight per age')
plt.xlabel('Age (months)')
plt.ylabel('Weight (kg)')
plt.grid(True)
plt.scatter(x_train, y_train)
plt.legend(['Linear', 'Quadratic', 'Cubic'])
plt.show()



