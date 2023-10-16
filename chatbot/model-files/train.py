import random
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Dropout
from keras.models import load_model
from keras.models import Sequential
from keras import callbacks
import numpy as np
import pickle
import json
import nltk
from nltk.stem import WordNetLemmatizer


lemmatizer = WordNetLemmatizer()
nltk.download('omw-1.4')
nltk.download("punkt")
nltk.download("wordnet")


# init file
words = []
classes = []
documents = []
ignore_words = ["?", "!", ",", "\n"]
data_file = open("medical-dataset.json").read()
intents = json.loads(data_file)

# words
for intent in intents["intents"]:
  for question in intent['question']:
        w = nltk.word_tokenize(question)
        words.extend(w)
        documents.append((w, intent['tags']))
        if intent['tags'] not in classes:
            classes.append(intent['tags'])
  # # take each word and tokenize it
  # w = nltk.word_tokenize(intent["question"])
  # words.extend(w)
  # # adding documents
  # documents.append((w, intent["tags"]))

  # # adding classes to our class list
  # if intent["tags"] not in classes:
  #   classes.append(str(intent["tags"]))
# lemmatizer
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(classes)
#print(len(documents), "documents")

#print(len(classes), "classes", classes)

#print(len(words), "unique lemmatized words", words)


pickle.dump(words, open("words.pkl", "wb"))
pickle.dump(classes, open("classes.pkl", "wb"))

# training initializer
# initializing training data
training = []
output_empty = [0] * len(classes)
for doc in documents:
    # initializing bag of words
    bag = []
    # list of tokenized words for the question
    question_words = doc[0]
    # lemmatize each word - create base word, in attempt to represent related words
    question_words = [lemmatizer.lemmatize(word.lower()) for word in question_words]
    # create our bag of words array with 1, if word match found in current pattern
    for w in words:
      bag.append(1) if w in question_words else bag.append(0)

    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])
# shuffle our features and turn into np.array
random.shuffle(training)
training = np.asarray(training, dtype="object")
# create train and test lists. X - patterns, Y - intents
train_x = list(training[:, 0])
train_y = list(training[:, 1])
#print("Training data created")

# actual training
# Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation="softmax"))
model.summary()



# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = tf.keras.optimizers.legacy.SGD(learning_rate=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

# for choosing an optimal number of training epochs to avoid underfitting or overfitting use an early stopping callback to keras
# based on either accuracy or loos monitoring. If the loss is being monitored, training comes to halt when there is an
# increment observed in loss values. Or, If accuracy is being monitored, training comes to halt when there is decrement observed in accuracy values.

# from keras import callbacks
earlystopping = callbacks.EarlyStopping(monitor ="loss", mode ="min", patience = 5, restore_best_weights = True)
callbacks =[earlystopping]

# fitting and saving the model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=1000, batch_size=5)
model.save("chatbot_model.h5", hist)
print("model created")