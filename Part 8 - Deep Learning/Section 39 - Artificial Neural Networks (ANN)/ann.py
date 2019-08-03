# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values  #everything excluding rownumber, id & surename
y = dataset.iloc[:, 13].values #last column is the results

# Encoding categorical data
# will need to encode columns 1, 2 since one of them is the nacionality and the other one is gender
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

onehotencoder = OneHotEncoder(categorical_features = [1]) #creating dummy variables since there are 3 countries in the dataset (for genders which are only two, no need for that)
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:] #removing one of the 3 dummy variables to avoid the dummy variable trap

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) #20% for test the remainder for training

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


################################################# UPWARD JUST DATA PREPROCESSING, HERE IS THE ACTUAL ANN #######################################################################

#importing Keras library and packs
import  keras
from keras.models import Sequential
from keras.layers import Dense

#initializing the artificial neural network
classifier = Sequential()

#building the input layer and the first hidden layer
classifier.add(Dense(units = 6, kernel_initializer='uniform', activation = 'relu'))

#second hidden layer
classifier.add(Dense(units = 6, kernel_initializer='uniform', activation = 'relu'))

#output layer. If i had 3 or 4 outcomes instead of only 2 possibilities (here its stay or leave, but could be stay active, go inactive or delete account for exemple), i would change units = 1 by units = 3 and sigmoid by softmax (which is basically the same but for more variables)
classifier.add(Dense(units = 1, kernel_initializer='uniform', activation = 'sigmoid'))

#compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#adam is a good stocastic gradient decent algorithim

#fitting the network to the training dataset
classifier.fit(X_train, y_train, batch_size = 10, epochs=100)


###############################################3 NOW JUST COMPARING TO THE TEST DATASET #######################################################################################


#Predicting and evaluating it

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5) #transforming from values between 0 and 1 to 0 or 1 (my treshold is at 50%, could be lower for sensitive cases like danger of mechanical failure)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)