#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression
import random
from random import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import style

data = pd.read_csv(r"E:\Projects\Students grade\student-mat.csv", sep=";")

data = data[["G1","G2","G3","studytime","health","absences"]]
predict = 'G3'

data.head()

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

best_accuracy = 0
for i in range(10):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)
    linear = LinearRegression().fit(x_train, y_train)
    accuracy = linear.score(x_test, y_test)
    if accuracy>best_accuracy:
        best_accuracy = accuracy

print('Accuracy: ', accuracy)
print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(round(predictions[x]), x_test[x], y_test[x])


# In[11]:


p = 'G1'
style.use("ggplot")
plt.scatter(data[p],data[predict])
plt.xlabel(p)
plt.ylabel("Final Grade")
plt.show()


# In[ ]:
