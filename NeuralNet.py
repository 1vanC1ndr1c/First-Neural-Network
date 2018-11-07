# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 12:48:34 2018

@author: Ivan
"""

############################# 1. LOAD DATA ####################################
'''                                                                           #
In this tutorial, we are going to use the Pima Indians onset of               #
diabetes dataset.                                                             #
This is a standard machine learning dataset from the                          #
UCI Machine Learning repository.                                              #
It describes patient medical record data for Pima Indians                     #
 and whether they had an onset of diabetes within five years.                 #
                                                                              #
As such, it is a binary classification problem                                #
(onset of diabetes as 1 or not as 0).                                         #
All of the input variables that describe each patient are numerical.          #
This makes it easy to use directly with neural networks that                  #
expect numerical input and output values,                                     #
and ideal for our first neural network in Keras.                              #
'''                                                                           #
#You can initialize the random number generator with any seed you like        #
from keras.models import Sequential                                           #
from keras.layers import Dense                                                #
import numpy                                                                  #
# fix random seed for reproducibility                                         #
numpy.random.seed(7)                                                          #
                                                                              #
# load pima indians dataset                                                   #
dataset = numpy.loadtxt("pima-indians-diabetes.data.csv", delimiter=",")      #
                                                                              #
# split into input (X) and output (Y) variables                               #
#There are eight input variables and one output variable (the last column).   #
# Once loaded we can split the dataset into input variables (X)               #
#and the output class variable (Y).                                           #
#Note, the dataset has 9 columns and the range 0:8                            #
#will select columns from 0 to 7, stopping before index 8                     #
X = dataset[:,0:8]                                                            #
Y = dataset[:,8]                                                              #
###############################################################################




############################# 2. DEFINE MODEL #################################
'''                                                                           #
The first thing to get right is to ensure the input layer has the right number#
of inputs.                                                                    #
This can be specified when creating the first layer with the input_dim        #
argument and setting it to 8 for the 8 input variables.                       #
                                                                              #
In this example, we will use a fully-connected                                #
network structure with three layers.                                          #
                                                                              #
Fully connected layers are defined using the Dense class                      #
                                                                              #
We can specify the number of neurons in the layer as the first argument,      #
the initialization method as the second argument as init and specify          #
the activation function using the activation argument.                        #
                                                                              #
                                                                              #
In this case, we initialize the network weights to a small random number      #
generated from a uniform distribution (‘uniform‘),                            #
in this case between 0 and 0.05 because that is the default uniform weight    #
initialization in Keras.                                                      #
                                                                              #
                                                                              #
We will use the rectifier (‘relu‘) activation function on                     #
the first two layers and the sigmoid function in the output layer.            # 
Better performance is achieved using the rectifier activation function.       #
We use a sigmoid on the output layer to ensure our network output             #
is between 0 and 1 and easy to map to either a probability of class 1 or      #
snap to a hard classification of either class with a default threshold of 0.5.#
                                                                              #
                                                                              #
We can piece it all together by adding each layer.                            #
The first layer has 12 neurons and expects 8 input variables.                 #
The second hidden layer has 8 neurons and finally,                            #
the output layer has 1 neuron to predict the class (onset of diabetes or not).#
'''                                                                           #
                                                                              #
# create model                                                                #
model = Sequential()                                                          #
model.add(Dense(12, input_dim=8, activation='relu'))                          #
model.add(Dense(8, activation='relu'))                                        #
model.add(Dense(1, activation='sigmoid'))                                     #
###############################################################################




############################# 3. COMPILE MODEL ################################
'''                                                                           #
Compiling the model uses the efficient numerical libraries under the covers   #
(the so-called backend) such as Theano or TensorFlow.                         #
The backend automatically chooses the best way to represent                   #
the network for training and making predictions to run on your hardware,      #
such as CPU or GPU or even distributed.                                       #
                                                                              #
                                                                              #
When compiling, we must specify some additional properties required when      #
training the network.                                                         #
                                                                              #
Remember training a network means finding the best set of weights to          #
make predictions for this problem.                                            #
                                                                              #
We must specify the loss function to use to evaluate a set of weights,        #
the optimizer used to search through different weights for the network        #
and any optional metrics we would like to collect and report during training. #
                                                                              #
In this case, we will use logarithmic loss,                                   #
which for a binary classification problem is defined in Keras as              #
“binary_crossentropy“.                                                        #
We will also use the efficient gradient descent algorithm “adam”.             #
'''                                                                           #
                                                                              #
# Compile model                                                               #
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
###############################################################################




############################# 4. FIT MODEL ####################################
''''                                                                          #
Now it is time to execute the model on some data.'                            #
We can train or fit our model on our loaded data by calling                   #
the fit() function on the model.                                              #
'                                                                             #
The training process will run for a fixed number of iterations through        #
the dataset called epochs,                                                    #
that we must specify using the nepochs argument.                              #
We can also set the number of instances that are evaluated before a           #
weight update in the network is performed,                                    #
called the batch size and set using the batch_size argument.                  #
'''                                                                           #
                                                                              #
# Fit the model                                                               #
model.fit(X, Y, epochs=150, batch_size=10)                                    #
###############################################################################




############################# 5. EVALUATE MODEL ###############################
'''                                                                           #
We have trained our neural network on the entire dataset and we can evaluate  #
the performance of the network on the same dataset.                           #
                                                                              #
This will only give us an idea of how well we have modeled the dataset        #
(e.g. train accuracy), but no idea of how well the algorithm                  #
might perform on new data.                                                    #
We have done this for simplicity, but ideally,                                #
you could separate your data into train and test datasets                     #
for training and evaluation of your model.                                    #
                                                                              #
You can evaluate your model on your training dataset using the evaluate()     #
function on your model and pass it the same                                   #
input and output used to train the model.                                     #
                                                                              #
This will generate a prediction for each input and output pair                #
and collect scores, including the average loss                                #
and any metrics you have configured, such as accuracy.                        #
'''                                                                           #
                                                                              #
# evaluate the model                                                          #
scores = model.evaluate(X, Y)                                                 #
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))               #
###############################################################################


