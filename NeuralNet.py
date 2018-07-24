#####################################################################################################################
#   CS 6375.003 - Assignment 3, Neural Network Programming
#   All assumptions and modifications made to the supplied code are mentioned in the comments
#####################################################################################################################


import numpy as np
import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import Imputer

class NeuralNet:
    def __init__(self, train, option, activation, label = None, header = True, h1 = 4, h2 = 2):
        np.random.seed(1)

        # train refers to the training dataset
        # test refers to the testing dataset
        # h1 and h2 represent the number of nodes in 1st and 2nd hidden layers
        self.activation = activation
        if option == 1:
            raw_input = pd.read_csv(train)
            train_dataset = self.preprocess(raw_input)
            ncols = len(train_dataset.columns)
            nrows = len(train_dataset.index)
            self.X = train_dataset.iloc[:, 0:(ncols -1)].values.reshape(nrows, ncols-1)
            self.y = train_dataset.iloc[:, (ncols-1)].values.reshape(nrows, 1)
        elif option == 2 or option == 3 or option == 4:            
            self.X = train
            self.y = label

        # Find number of input and output layers from the dataset        
        input_layer_size = len(self.X[0])
        if not isinstance(self.y[0], np.ndarray):
            output_layer_size = 1
        else:
            output_layer_size = len(self.y[0])

        # assign random weights to matrices in network
        # number of weights connecting layers = (no. of nodes in previous layer) x (no. of nodes in following layer)
        self.w01 = 2 * np.random.random((input_layer_size, h1)) - 1
        self.X01 = self.X
        self.delta01 = np.zeros((input_layer_size, h1))
        self.w12 = 2 * np.random.random((h1, h2)) - 1
        self.X12 = np.zeros((len(self.X), h1))
        self.delta12 = np.zeros((h1, h2))
        self.w23 = 2 * np.random.random((h2, output_layer_size)) - 1
        self.X23 = np.zeros((len(self.X), h2))
        self.delta23 = np.zeros((h2, output_layer_size))
        self.deltaOut = np.zeros((output_layer_size, 1))
    
    #Activation function with sigmoid as default
    def __activation(self, x, activation="sigmoid"):
        if activation == "sigmoid":
            der = self.__sigmoid(x)
        elif activation == "tanh":
            der = self.__tanh(x)
        elif activation == "relu":
            der = self.__relu(x)
        return der
    
    # Activation function derivative with Sigmoid as default    
    def __activation_derivative(self, x, activation="sigmoid"):
        if activation == "sigmoid":
            der = self.__sigmoid_derivative(x)
        elif activation == "tanh":
            der = self.__tanh_derivative(x)
        elif activation == "relu":
            der = self.__relu_derivate(x)
        return der

    # sigmoid function definition
    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # derivative of sigmoid function, indicates confidence about existing weight
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    #tanh function
    def __tanh(self, x):        
        return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

    # derivative of tanh function
    def __tanh_derivative(self, x):
        return 1 - np.power(x, 2)

    # relu function 
    def __relu(self, x):
        x[x <= 0] = 0
        return x

    # derivative of relu function
    def __relu_derivative(self, x): 
        x[x <= 0] = 0
        x[x > 0] = 1
        return x

    #pre-process function for the dataSet
    def preprocess(self, X):   

        #Scaling of data   
        scaler = StandardScaler();
        scaler.fit(X);
        X = scaler.transform(X);
        #Normalizing the scaled data
        normalizer = Normalizer().fit(X)        
        X = pd.DataFrame(X)
        return X      

    # Below is the training function
    def train(self, max_iterations = 1000, learning_rate = 0.05):
        for iteration in range(max_iterations):
            out = self.forward_pass()
            error = 0.5 * np.power((out - self.y), 2)
            self.backward_pass(out, self.activation)
            update_layer2 = learning_rate * self.X23.T.dot(self.deltaOut)
            update_layer1 = learning_rate * self.X12.T.dot(self.delta23)
            update_input = learning_rate * self.X01.T.dot(self.delta12)

            self.w23 += update_layer2
            self.w12 += update_layer1
            self.w01 += update_input

        print("After " + str(max_iterations) + " iterations, the total error is " + str(np.sum(error)))
        print("The final weight vectors are (starting from input to output layers)")
        print(self.w01)
        print(self.w12)
        print(self.w23)
    
    def forward_pass(self):
        # pass our inputs through our neural network        
        in1 = np.dot(self.X, self.w01 )
        self.X12 = self.__activation(in1, self.activation)
        in2 = np.dot(self.X12, self.w12)
        self.X23 = self.__activation(in2, self.activation)
        in3 = np.dot(self.X23, self.w23)
        out = self.__activation(in3, self.activation)
        return out

    def backward_pass(self, out, activation):
        # pass our inputs through our neural network
        self.compute_output_delta(out, activation)
        self.compute_hidden_layer2_delta(activation)
        self.compute_hidden_layer1_delta(activation)

    def compute_output_delta(self, out, activation="sigmoid"):
        if activation == "sigmoid":
            delta_output = (self.y - out) * (self.__sigmoid_derivative(out))
        elif activation == "tanh":
            delta_output = (self.y - out) * (self.__tanh_derivative(out))
        elif activation == "relu":
            delta_output = (self.y - out) * (self.__relu_derivative(out))

        self.deltaOut = delta_output

    def compute_hidden_layer2_delta(self, activation="sigmoid"):
        if activation == "sigmoid":
            delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__sigmoid_derivative(self.X23))
        elif activation == "tanh":
            delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__tanh_derivative(self.X23))
        elif activation == "relu":
            delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__relu_derivative(self.X23))

        self.delta23 = delta_hidden_layer2

    def compute_hidden_layer1_delta(self, activation="sigmoid"):
        if activation == "sigmoid":
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__sigmoid_derivative(self.X12))
        elif activation == "tanh":
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__tanh_derivative(self.X12))
        elif activation == "relu":
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__relu_derivative(self.X12))

        self.delta12 = delta_hidden_layer1

   
    def compute_input_layer_delta(self, activation="sigmoid"):
        if activation == "sigmoid":
            delta_input_layer = np.multiply(self.__sigmoid_derivative(self.X01), self.delta01.dot(self.w01.T))
            self.delta01 = delta_input_layer

#predicting the error for the test data from csv file given

    def predict(self, test, header = True, h1 = 4, h2 = 2):       
        raw_input = pd.read_csv(test)               
        test_dataset = self.preprocess(raw_input)
        ncols = len(test_dataset.columns)
        nrows = len(test_dataset.index)
        self.X = test_dataset.iloc[:, 0:(ncols -1)].values.reshape(nrows, ncols-1)
        self.y = test_dataset.iloc[:, (ncols-1)].values.reshape(nrows, 1)         
        self.X01 =  self.X              
        self.X12 = np.zeros((len(self.X), h1))        
        self.X23 = np.zeros((len(self.X), h2))     
        out = self.forward_pass()        
        return ((np.sum(pow((self.y - out),2))) / 2)

#predicting the error for the test data for URL
    def predict_data(self, test, label, header = True, h1 = 4, h2 = 2):        
       	self.X = test
        self.y = label
        self.X01 =  self.X              
        self.X12 = np.zeros((len(self.X), h1))        
        self.X23 = np.zeros((len(self.X), h2))     
        out = self.forward_pass()   
        return ((np.sum(pow((self.y - out),2))) / 2)

#pre-processing the dataset based on URL , which includes replacing unknown values, categorical to numerical conversions, standardization and normalization
def preprocess_data(X):     
    
    X = X.replace('[?]', np.nan, regex=True) # Replace ? in dataSet with nan

    # NaN values of columns with dtype object are replaced with the most frequent value in that column. Columns of other types are imputed with mean of column.    
    fill = pd.Series([X[col].value_counts().index[0]
            if X[col].dtype == np.dtype('O') else X[col].mean() for col in X],
            index=X.columns)
    X = X.fillna(fill) 

    #convert categorical data to numeric value between 0 and n_classes-1
    enc = preprocessing.LabelEncoder() 
    for col in X.columns.values:
        # Encoding only categorical variables
        if X[col].dtypes=='object':
            enc.fit(X[col].astype(str))
            X[col]=enc.transform(X[col].astype(str))

    #scaling
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    #Normalizing the scaled data   
    X = Normalizer().fit_transform(X)      

    X = pd.DataFrame(X)
   
    return X

if __name__ == "__main__":

    if (len(sys.argv)) < 2 :   #Print error if number of arguments is less than 2
        print("Invalid number of arguments")
    else:
        activation = sys.argv[1] # Read activation fucntion from command line        
        responseData = input("1.Train.CSV and Test.CSV\n2.IRIS Dataset\n3.Car Evaluation Dataset\n4.Adult Census Income Dataset\nPlease enter your selection: ") #enable user to select dataset from the commandline 
        if (responseData == '1'):
            neural_network = NeuralNet("train.csv", int(responseData), activation)
            neural_network.train()
            testError = neural_network.predict("test.csv")
        elif (responseData == '2'):
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"        
        elif responseData == '3':
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
        elif responseData == '4':
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
        else:
            print("Invalid Input")
        if responseData == '2' or responseData == '3' or responseData == '4':       
            raw_input = pd.read_csv(url,header=None)     
            dataset = preprocess_data(raw_input)
            ncols = len(dataset.columns)
            nrows = len(dataset.index)
            X = dataset.iloc[:, 0:(ncols -1)].values.reshape(nrows, ncols-1)
            y = dataset.iloc[:, (ncols-1)].values.reshape(nrows, 1)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
            neural_network = NeuralNet(X_train, int(responseData), activation, y_train)
            neural_network.train()   
            testError = neural_network.predict_data(X_test,y_test)
        print("Test error is ", testError)
    
    
    

