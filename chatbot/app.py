# libraries
import random
import numpy as np
import pickle
import json
from flask import Flask, render_template, request 
import nltk
from keras.models import load_model
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

# chat initialization
model = load_model("model-files/chatbot_model.h5")
intents = json.loads(open("model-files/medical-dataset.json").read())
words = pickle.load(open("model-files/words.pkl", "rb"))
classes = pickle.load(open("model-files/classes.pkl", "rb"))


#===========FLASK============
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")


@app.route('/response', methods=['GET', 'POST'])
def response():
    msg = request.form["msg"]
    return chatbot_response(msg)


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def get_response(intents_list, intents_json):
  result = []
  try:
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tags'] == tag:
            result = random.choice(i['answer'])
            break
  except:
    result = "I cannot understand this statement. Perhaps rephrase it differently?"
  return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = get_response(ints, intents)
    return res
#test the chatbot with CLI 
# while True:
#   message = input("You: ")
#   try:
#     ints = predict_class(message, model)
#     print(ints)
#     response = get_response(ints, intents)
#     print("Bot: " + response)

#   except IndexError:
#     print("What?")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=3000)