import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn import model_selection
import warnings
warnings.filterwarnings("ignore")


class NeuralNet:
    def __init__(self, train, header = True, h1 = 10, h2 = 5, h3 = 5):
        np.random.seed(1)
        train_dataset = train
        nrows, ncols = train_dataset.shape
        self.X = train_dataset[:, :ncols-1]
        self.y = train_dataset[:, ncols-1:ncols]
        #
        # Find number of input and output layers from the dataset
        #
        ## the given output layer size is wrong! I modified it.
        input_layer_size = len(self.X[0])
        if not isinstance(self.y, np.ndarray):
            output_layer_size = 1
        else:
            output_layer_size = len(np.unique(self.y))

        # assign random weights to matrices in network
        # number of weights connecting layers = (no. of nodes in previous layer) x (no. of nodes in following layer)
        self.w01 = 2 * np.random.random((input_layer_size, h1)) - 1
        self.X01 = self.X
        self.delta01 = np.zeros((input_layer_size, h1))
        self.w12 = 2 * np.random.random((h1, h2)) - 1
        self.X12 = np.zeros((len(self.X), h1))
        self.delta12 = np.zeros((h1, h2))
        self.w23 = 2 * np.random.random((h2, h3)) - 1
        self.X23 = np.zeros((len(self.X), h2))
        self.delta23 = np.zeros((h2, h3))
        self.w34 = 2 * np.random.random((h3, output_layer_size)) - 1
        self.X34 = np.zeros((len(self.X), h3))
        self.delta34 = np.zeros((h3, output_layer_size))
        self.deltaOut = np.zeros((output_layer_size, 1))


    def __activation(self, x, activation="sigmoid"):
        if activation == "sigmoid":
            return self.__sigmoid(x)
        if activation == "tanh":
            return self.__tanh(x)
        if activation == "ReLu":
            return self.__ReLu(x)

    def __activation_derivative(self, x, activation="sigmoid"):
        if activation == "sigmoid":
            return self.__sigmoid_derivative(x)
        if activation == "tanh":
            return self.__tanh_derivative(x)
        if activation == "ReLu":
            return self.__ReLu_derivative(x)

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def __tanh(self, x):
        return 2*self.__sigmoid(2*x) - 1
        # return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    
    def __ReLu(self, x):
        return np.maximum(x, 0)

    # derivative of sigmoid function, indicates confidence about existing weight
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    def __tanh_derivative(self, x):
        return 1 - np.power(x, 2)

    def __ReLu_derivative(self, x):
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
            # somewhere also wrong i think, it should be minus

            self.w34 -= update_layer3
            self.w23 -= update_layer2
            self.w12 -= update_layer1
            self.w01 -= update_input

        print("After " + str(max_iterations) + " iterations, the total error is " + str(np.sum(error)))
        print("The final weight vectors are (starting from input to output layers)")
        print(self.w01)
        print(self.w12)
        print(self.w23)
        print(self.w34)

    def forward_pass(self, activation="sigmoid"):
        # pass our inputs through our neural network
        in1 = np.dot(self.X, self.w01)
        self.X12 = self.__activation(in1, activation)
        in2 = np.dot(self.X12, self.w12)
        self.X23 = self.__activation(in2, activation)
        in3 = np.dot(self.X23, self.w23)
        self.X34 = self.__activation(in3, activation)
        in4 = np.dot(self.X34, self.w34)
        out = self.__activation(in4, activation)
        return out

    def backward_pass(self, out, activation="sigmoid"):
        # pass our inputs through our neural network
        self.deltaOut = (self.y - out) * (self.__activation_derivative(out, activation))
        self.delta34 = (self.deltaOut.dot(self.w34.T)) * (self.__activation_derivative(self.X34, activation))
        self.delta23 = (self.delta34.dot(self.w23.T)) * (self.__activation_derivative(self.X23, activation))
        self.delta12 = (self.delta23.dot(self.w12.T)) * (self.__activation_derivative(self.X12, activation))

    def predict(self, test, header = True):
        # raw_input = pd.read_csv(test)
        test_dataset = test
        nrows, ncols = test_dataset.shape
        self.X = test_dataset[:, :ncols-1]
        self.y = test_dataset[:, ncols-1:ncols]
        out = self.forward_pass(self.activation)
        # error = 0.5 * np.power((out - self.y), 2)
        rows, cols = out.shape
        result = []
        for row in range(rows):
            MAX = out[row][0]
            pos = 0;
            for col in range(cols):
                if out[row][col] > MAX:
                    MAX = out[row][col]
                    pos = col
            result.append(pos)
        result = np.array([result]).T
        right = int(np.sum(result==self.y))
        total = len(self.y)
        print("right classification: %f" % right)
        print("total data: %f" % total)
        return right / total

if __name__ == "__main__":

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


    imp = SimpleImputer(strategy='most_frequent')
    datacopy = imp.fit_transform(datacopy)
    datacopy = datacopy.astype(np.float)
    trainDF, testDF = model_selection.train_test_split(datacopy, test_size=0.20, random_state=42)
    print(trainDF)
    neural_network = NeuralNet(trainDF)
    
    neural_network.train(max_iterations=50000, learning_rate=0.1, activation="sigmoid")
    # neural_network.train(max_iterations=10000, learning_rate=0.2, activation="tanh")
    # neural_network.train(max_iterations=50000, learning_rate=0.1, activation="ReLu")

    acc = neural_network.predict(testDF)
    print("Accuracy: %f" % acc)