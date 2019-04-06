# Mine
from numpy import exp, array, random, dot
import nltk
from TranningData import TrainingData
from TextPreprocessor import tweet_preprocessor
import spacy

tp = tweet_preprocessor()
t_data = TrainingData()
words = []
classes = []
documents = []


def simlarity2words():
    nlp = spacy.load('en_core_web_sm')
    doc1 = nlp("find")
    doc2 = nlp("here")
    a = doc1.similarity(doc2)
    print(a)

class Text2Vector():

    def tokenStr(self,str):
        global words
        global classes
        global documents

        # loop through each sentence in our training data
        for pattern in t_data.getTrainData():
            # tokenize each word in the sentence
            w = nltk.word_tokenize(pattern['sent'])
            # add to our words list
            words.extend(w)
            # add to documents in our corpus
            documents.append((w, pattern['class']))
            # add to our classes list
            if pattern['class'] not in classes:
                classes.append(pattern['class'])

        # stem and lower each word and remove duplicates
        words = tp.batchProcessingString(words)
        words = list(set(words))
        # remove duplicates
        classes = list(set(classes))

        print(len(documents), "documents")
        print(len(classes), "classes", classes)
        print(len(words), "unique stemmed words", words)

    def parse2Vector(self):
        # create our training data
        training = []
        output = []
        # create an empty array for our output
        output_empty = [0] * len(classes)
        print(documents)
        # training set, bag of words for each sentence
        for doc in documents:
            # initialize our bag of words
            bag = []
            # list of tokenized words for the pattern
            pattern_words = doc[0]
            # stem each word
            pattern_words = tp.batchProcessingString(pattern_words)
            # create our bag of words array
            for w in words:
                bag.append(1) if w in pattern_words else bag.append(0)

            training.append(bag)
            # output is a '0' for each tag and '1' for current tag
            output_row = list(output_empty)
            output_row[classes.index(doc[1])] = 1
            output.append(output_row)

        for t in training:
            print(t)


class NeuralNetwork():

    def __init__(self):
        random.seed(1)
        # 输入层三个神经元作为第一层
        # 第二层定义为5个神经元
        # 第三层定义为4个神经元
        layer2 = 5
        layer3 = 4

        # 随机初始化各层权重
        self.synaptic_weights1 = 2 * random.random((3, layer2)) - 1
        self.synaptic_weights2 = 2 * random.random((layer2, layer3)) - 1
        self.synaptic_weights3 = 2 * random.random((layer3, 1)) - 1

    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):

            # 正向传播过程，即神经网络“思考”的过程
            activation_values2 = self.__sigmoid(dot(training_set_inputs, self.synaptic_weights1))
            activation_values3 = self.__sigmoid(dot(activation_values2, self.synaptic_weights2))
            output = self.__sigmoid(dot(activation_values3, self.synaptic_weights3))

            # 计算各层损失值
            delta4 = (training_set_outputs - output) * self.__sigmoid_derivative(output)
            delta3 = dot(self.synaptic_weights3, delta4.T) * (self.__sigmoid_derivative(activation_values3).T)
            delta2 = dot(self.synaptic_weights2, delta3) * (self.__sigmoid_derivative(activation_values2).T)

            # 计算需要调制的值
            adjustment3 = dot(activation_values3.T, delta4)
            adjustment2 = dot(activation_values2.T, delta3.T)
            adjustment1 = dot(training_set_inputs.T, delta2.T)

            # 调制权值
            self.synaptic_weights1 += adjustment1
            self.synaptic_weights2 += adjustment2
            self.synaptic_weights3 += adjustment3


    def think(self, inputs):
        activation_values2 = self.__sigmoid(dot(inputs, self.synaptic_weights1))
        activation_values3 = self.__sigmoid(dot(activation_values2, self.synaptic_weights2))
        output = self.__sigmoid(dot(activation_values3, self.synaptic_weights3))
        return output

if __name__ == "__main__":
    simlarity2words()
    t = Text2Vector()
    t.tokenStr("hello world how are you!")
    t.parse2Vector()
    # neural_network = NeuralNetwork()
    # # 训练集不变
    # training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    # training_set_outputs = array([[0, 1, 1, 0]]).T
    # neural_network.train(training_set_inputs, training_set_outputs, 10000)
    # print("\nNew synaptic weights (layer 1) after training: ")
    # print(neural_network.synaptic_weights1)
    # print("\nNew synaptic weights (layer 2) after training: ")
    # print(neural_network.synaptic_weights2)
    # print("\nNew synaptic weights (layer 3) after training: ")
    # print(neural_network.synaptic_weights3)
    # print("Considering new situation [1, 0, 0] -> ?: ")
    # print(neural_network.think(array([0, 1, 1])))