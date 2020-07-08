import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn import model_selection
from sklearn import preprocessing

# define NeuralNet
class NeuralNet:
    def __init__(self, train, w12=0, w23=0, w34=0, trsf1=False, trsf2=False, trsf3=False):
        h1 = 10
        h2 = 5
        h3 = 5
        np.random.seed(1)
        train_dataset = train
        nrows, ncols = train_dataset.shape
        self.X = train_dataset[:, :ncols-1]
        self.y = train_dataset[:, ncols-1:ncols]
        #
        # Find number of input and output layers from the dataset
        #
        input_layer_size = len(self.X[0])
        if not isinstance(self.y, np.ndarray):
            output_layer_size = 1
        else:
            output_layer_size = len(np.unique(self.y))
        output_layer_size = 1

        # assign random weights to matrices in network
        # number of weights connecting layers = (no. of nodes in previous layer) x (no. of nodes in following layer)
        self.w01 = 2 * np.random.random((input_layer_size, h1)) - 1
        self.X01 = self.X
        self.w12 = 2 * np.random.random((h1, h2)) - 1
        self.X12 = np.zeros((len(self.X), h1))
        self.delta12 = np.zeros((nrows, h1))
        self.w23 = 2 * np.random.random((h2, h3)) - 1
        self.X23 = np.zeros((len(self.X), h2))
        self.delta23 = np.zeros((nrows, h2))
        self.w34 = 2 * np.random.random((h3, output_layer_size)) - 1
        self.X34 = np.zeros((len(self.X), h3))
        self.delta34 = np.zeros((nrows, h3))
        self.deltaOut = np.zeros((nrows, output_layer_size))
        if trsf1 != 0:
            self.w12 = w12
        if trsf2 != 0:
            self.w23 = w23
        if trsf3 != 0:
            self.w34 = w34


    def __activation(self, x, activation="sigmoid"):
        if activation == "sigmoid":
            x = np.clip(x, -500, 500)
            return 1 / (1 + np.exp(-x))
        if activation == "tanh":
            return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
            # return 2*self.__sigmoid(2*x) - 1
        if activation == "ReLu":
            return np.maximum(x, 0)

    def __activation_derivative(self, x, activation="sigmoid"):
        if activation == "sigmoid":
            return x * (1 - x)
        if activation == "tanh":
            return 1 - np.power(x, 2)
        if activation == "ReLu":
            return 1 * (x > 0)


    def train(self, max_iterations = 1000, learning_rate = 0.05, activation="sigmoid"):
        self.activation = activation
        for iteration in range(max_iterations):
            out = self.forward_pass(activation)
            error = 0.5 * np.power((out - self.y), 2)
            self.backward_pass(out, activation)
            update_layer3 = learning_rate * self.X34.T.dot(self.deltaOut)
            update_layer2 = learning_rate * self.X23.T.dot(self.delta34)
            update_layer1 = learning_rate * self.X12.T.dot(self.delta23)
            update_input = learning_rate * self.X01.T.dot(self.delta12)
            self.w34 += update_layer3
            self.w23 += update_layer2
            self.w12 += update_layer1
            self.w01 += update_input


    def forward_pass(self, activation="sigmoid"):
        # pass our inputs through our neural network
        in1 = np.dot(self.X01, self.w01)
        self.X12 = self.__activation(in1, activation)
        in2 = np.dot(self.X12, self.w12)
        self.X23 = self.__activation(in2, activation)
        in3 = np.dot(self.X23, self.w23)
        self.X34 = self.__activation(in3, activation)
        in4 = np.dot(self.X34, self.w34)
        out = self.__activation(in4, "sigmoid")
        return out

    def backward_pass(self, out, activation="sigmoid"):
        # pass our inputs through our neural network
        self.deltaOut = (self.y - out) * (self.__activation_derivative(out, "sigmoid"))
        self.delta34 = (self.deltaOut.dot(self.w34.T)) * (self.__activation_derivative(self.X34, activation))
        self.delta23 = (self.delta34.dot(self.w23.T)) * (self.__activation_derivative(self.X23, activation))
        self.delta12 = (self.delta23.dot(self.w12.T)) * (self.__activation_derivative(self.X12, activation))

    def predict(self, test):
        test_dataset = test
        nrows, ncols = test_dataset.shape
        self.X = test_dataset[:, :ncols-1]
        self.y = test_dataset[:, ncols-1:ncols]
        self.X01 = self.X
        out = self.forward_pass(self.activation)
        error = 0.5 * np.power((out - self.y), 2)
        rows, cols = out.shape
        for row in range(rows):
            for col in range(cols):
                if out[row][col] >= 0.5:
                    out[row][col] = 1.
                else:
                    out[row][col] = 0.
        # print(out)
        # print(self.y)
        right = int(np.sum(out==self.y))
        total = len(self.y)
        print("right classification: %f" % right)
        print("total data: %f" % total)
        return right / total


# train and test hepatitis with random weights
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/hepatitis/hepatitis.data',
names = ['Class','Age','SEX','STEROID', 'ANTIVIRALS', 'FATIGUE', 'Deg-MALAISE', 'ANOREXIA', 'LIVER BIG', 'LIVER FIRM',
'SPLEEN PALPABLE', 'SPIDERS', 'ASCITES', 'VARICES', 'BILIRUBIN', 'ALK PHOSPHATE', 'SGOT', 'ALBUMIN', 'PROTIME', 'HISTOLOGY'])

# fill ? to most frequent value
del df['LIVER BIG']
del df['LIVER FIRM']
del df['ALK PHOSPHATE']
del df['ALBUMIN']
del df['PROTIME']

imp = SimpleImputer(missing_values='?', strategy='most_frequent')
df = imp.fit_transform(df)
df = df.astype(np.float)
min_max_scaler = preprocessing.MinMaxScaler()
df = min_max_scaler.fit_transform(df)

trainDF, testDF = model_selection.train_test_split(df, test_size=0.2)
neural_network = NeuralNet(trainDF)

neural_network.train(max_iterations=100000, learning_rate=0.001, activation="sigmoid")

acc = neural_network.predict(testDF)
print("Accuracy: %f" % acc)





# train and test breast cancer
from keras.models import Sequential
from keras.layers import Flatten, Dense, Activation
import pandas as pd
import numpy as np
import sklearn as sk
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint

import warnings
warnings.filterwarnings("ignore")
X = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer/breast-cancer.data', names = ['Class','Age','Menopause','TumorSize', 'InvNodes', 'NodeCaps', 'Deg-Malig', 'Breast', 'BreastQuadrant', 'Irradiated'])

X = X.replace("?", np.nan)    
datacopy = X.dropna()
datacopy['Class'] = X.Class.map({'no-recurrence-events':0, 'recurrence-events':1})
datacopy['Age'] = X.Age.map({'10-19':1, '20-29':2,'30-39':3, '40-49':4,'50-59':5, '60-69':6,'70-79':7, '80-89':8,'90-99':9})
datacopy['Menopause'] = X.Menopause.map({'lt40':1, 'ge40':2, 'premeno':3})
datacopy['TumorSize'] = X.TumorSize.map({'0-4':1, '5-9':2,'10-14':3, '15-19':4,'20-24':5, '25-29':6,'30-34':7, '35-39':8,'40-44':9, '45-49':10, '50-54':11,'55-59':12})
datacopy['InvNodes'] = X.InvNodes.map({'0-2':1, '3-5':2,'6-8':3, '9-11':4,'12-14':5, '15-17':6,'18-20':7, '21-23':8,'24-26':9, '27-29':10, '30-32':11,'33-35':12, '36-39':13})
datacopy['NodeCaps'] = X.NodeCaps.map({'no':0, 'yes':1})
datacopy['Breast'] = X.Breast.map({'left':1, 'right':2})
datacopy['BreastQuadrant'] = X.BreastQuadrant.map({'left_up':1, 'left_low':2, 'right_up':3, 'right_low':4, 'central':5})
datacopy['Irradiated'] = X.Irradiated.map({'no':0, 'yes':1})
x = datacopy.values 
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
datacopy = pd.DataFrame(x_scaled)
train, test = train_test_split(datacopy, test_size=0.20, random_state=42)

ncols = len(train.columns)
nrows = len(train.index)
ncolstest = len(test.columns)
nrowstest = len(test.index)
xtrain = train.iloc[:, 0:(ncols -1)].values.reshape(nrows, ncols-1)
ytrain = train.iloc[:, (ncols-1)].values.reshape(nrows, 1)
xtest = test.iloc[:, 0:(ncolstest -1)].values.reshape(nrowstest, ncolstest-1)
ytest = test.iloc[:, (ncolstest-1)].values.reshape(nrowstest, 1)
xtrain = xtrain.reshape(221,9,1)
print(xtrain.shape)
print(ytrain.shape)


model = Sequential()
model.add(Flatten())
model.add(Dense(10, input_shape=(221,9,), activation='sigmoid'))
model.add(Dense(5, activation='sigmoid'))
model.add(Dense(5, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['mse', 'accuracy'])
earlystop = [EarlyStopping(monitor='val_loss', patience=2)]
model.fit(xtrain, ytrain, epochs=1000, verbose=1, validation_split = 0.20, callbacks = earlystop)


# transfer weights

for layer in model.layers:
    weights = layer.get_weights()
    print(len(weights))
    for i in range(len(weights)):
      print(weights[i].shape)


w12 = model.layers[2].get_weights()[0]
w23 = model.layers[3].get_weights()[0]
w34 = model.layers[4].get_weights()[0]


# train and test hepatitis with transfer weights

neural_network = NeuralNet(trainDF, w12=w12, trsf1=True)
neural_network.train(max_iterations=100000, learning_rate=0.001, activation="sigmoid")
acc = neural_network.predict(testDF)
print("Accuracy: %f" % acc)
neural_network = NeuralNet(trainDF, w23=w23, trsf2=True)
neural_network.train(max_iterations=100000, learning_rate=0.001, activation="sigmoid")
acc = neural_network.predict(testDF)
print("Accuracy: %f" % acc)
neural_network = NeuralNet(trainDF, w34=w34, trsf3=True)
neural_network.train(max_iterations=100000, learning_rate=0.001, activation="sigmoid")
acc = neural_network.predict(testDF)
print("Accuracy: %f" % acc)
neural_network = NeuralNet(trainDF, w12, w23, w34, trsf1=True, trsf2=True, trsf3=True)
neural_network.train(max_iterations=100000, learning_rate=0.001, activation="sigmoid")
acc = neural_network.predict(testDF)
print("Accuracy: %f" % acc)