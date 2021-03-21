import json
import os
import pickle
import random
import typing as T

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Prevent Tensorflow from outputting hardware warnings

import nltk
import numpy
import tflearn
import tensorflow
from nltk.stem.lancaster import LancasterStemmer

tensorflow.compat.v1.logging.set_verbosity(tensorflow.compat.v1.logging.ERROR)

stemmer = LancasterStemmer()

def create_training_dataset(data: T.Dict[str, T.Any]) -> T.Tuple[T.List, T.List, T.List, T.List]:
    """
    Convert a dictionary into a training dataset for a chatbot.
    """
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)
    
    return (words, labels, training, output)


def create_model(input_size, output_size) -> tflearn.DNN:
    """
    Declare a neural network architecture and return the compiled model.
    """
    net = tflearn.input_data(shape=[None, input_size])
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, output_size, activation="softmax")
    net = tflearn.regression(net)

    return tflearn.DNN(net)

def bag_of_words(input_sentence: T.List[str], words: T.List[str]) -> numpy.array:
    """
    Tokenize an input sentence based on a previously defined bag of words.
    """
    bag = [0 for _ in range(len(words))]

    S_words = nltk.word_tokenize(input_sentence)
    S_words = [stemmer.stem(word.lower()) for word in S_words]

    for se in S_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = (1)

    return numpy.array(bag)

def chat(model: tflearn.DNN, data: T.Dict[str, T.Any], words: T.List, labels: T.List) -> None:
    """
    Initialize a discussion with a user and respond to user input with trained model.
    """
    print("Start speaking with me! Enter Q to quit")
    
    while True:
        inp = input("You: ")
        if inp.lower() == "q" or inp.lower() == "Q":
            break

        results = model.predict([bag_of_words(inp, words)])
        results_index = numpy.argmax(results)
        tag = labels[results_index]
        
        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']
        
        print(random.choice(responses))


if __name__ == "__main__":
    # Build input, then create, train, save, and return model.

    # Note that this function will try and load a trained TensorFlow
    # model with components saved with the prefix `model.tflearn`. If
    # the model is not found, then the model will be retrained.
    with open('intents.json') as file:
        data = json.load(file)

    try:
        with open("data.pickle", "rb") as f:
            words, labels, training, output = pickle.load(f)
    except:
        words, labels, training, output = create_training_dataset(data)

    tensorflow.reset_default_graph()
    model = create_model(len(training[0]), len(output[0]))

    try:
        model.load("model.tflearn")
    except:
        model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
        model.save("model.tflearn")
    
    chat(model, data, words, labels)