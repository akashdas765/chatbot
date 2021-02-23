import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
import joblib
import json
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from tqdm import tqdm
import os
import bert
from bert import BertModelLayer
from bert.loader import StockBertConfig, map_stock_config_to_params, load_stock_weights
from bert.tokenization.bert_tokenization import FullTokenizer



from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Flatten
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau



main_path=""
from tensorflow.keras.models import load_model
model = load_model(main_path+'saved_models/model.h5', custom_objects={"BertModelLayer": bert.BertModelLayer})



bert_model_name="uncased_L-12_H-768_A-12"

#bert_ckpt_dir = os.path.join("model/", bert_model_name)
bert_ckpt_file = os.path.join(main_path,"bert_model.ckpt.index")
bert_config_file = os.path.join(main_path,"bert_config.json")


train = pd.read_csv(main_path+"train.csv")
#valid = pd.read_csv("valid.csv")
test = pd.read_csv(main_path+"train.csv")



class IntentDetectionData:
  DATA_COLUMN = "questions"
  LABEL_COLUMN = "labels"

  def __init__(self, train, test, tokenizer: FullTokenizer, classes, max_seq_len=192):
    self.tokenizer = tokenizer
    self.max_seq_len = 0
    self.classes = classes

    ((self.train_x, self.train_y), (self.test_x, self.test_y)) = map(self._prepare, [train, test])

    print("max seq_len", self.max_seq_len)
    self.max_seq_len = min(self.max_seq_len, max_seq_len)
    self.train_x, self.test_x = map(self._pad, [self.train_x, self.test_x])

  def _prepare(self, df):
    x, y = [], []

    for _, row in tqdm(df.iterrows()):
      text, label = row[IntentDetectionData.DATA_COLUMN], row[IntentDetectionData.LABEL_COLUMN]
      tokens = self.tokenizer.tokenize(text)
      tokens = ["[CLS]"] + tokens + ["[SEP]"]
      token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
      self.max_seq_len = max(self.max_seq_len, len(token_ids))
      x.append(token_ids)
      y.append(self.classes.index(label))

    return np.array(x), np.array(y)

  def _pad(self, ids):
    x = []
    for input_ids in ids:
      input_ids = input_ids[:min(len(input_ids), self.max_seq_len - 2)]
      input_ids = input_ids + [0] * (self.max_seq_len - len(input_ids))
      x.append(np.array(input_ids))
    return np.array(x)



tokenizer = FullTokenizer(vocab_file=os.path.join(main_path+"uncased_L-12_H-768_A-12/vocab.txt"))


tokenizer.tokenize("I can't wait to visit Bulgaria again!")



classes = train.labels.unique().tolist()

data = IntentDetectionData(train, test, tokenizer, classes, max_seq_len=128)



responses=pd.read_csv(main_path+'response.csv')


# import our chat-bot intents file
import json
with open(main_path+'intents.json') as json_data:
    intents = json.load(json_data)


import random
# create a data structure to hold user context
context = {}


ERROR_THRESHOLD = 0.25
def classify(sentence):
    sentence=[sentence]
    pred_tokens = map(tokenizer.tokenize, sentence)
    pred_tokens = map(lambda tok: ["[CLS]"] + tok + ["[SEP]"], pred_tokens)
    pred_token_ids = list(map(tokenizer.convert_tokens_to_ids, pred_tokens))

    pred_token_ids = map(lambda tids: tids +[0]*(data.max_seq_len-len(tids)),pred_token_ids)
    pred_token_ids = np.array(list(pred_token_ids))

    predictions = model.predict(pred_token_ids)
    predictions1=np.argmax(model.predict(pred_token_ids))

    return_list =[]
    return_list.append((classes[predictions1],np.amax(predictions[0])))

    return return_list


def response(sentence, userID='123', show_details=False):
    results = classify(sentence)
    # if we have a classification then find the matching intent tag
    if results:
        # loop as long as there are matches to process
        while results:
            for i in intents['intents']:
                # find a tag matching the first result
                if i['tag'] == results[0][0]:
                    x=i['id']
                    # set context for this intent if necessary
                    if 'context_set' in i:
                        if show_details:
                            return (('context:', i['context_set']),x)
                        context[userID] = i['context_set']

                    # check if this intent is contextual and applies to this user's conversation
                    if not 'context_filter' in i or (userID in context and 'context_filter' in i and i['context_filter'] == context[userID]):
                        if show_details: print ('tag:', i['tag'])
                        # a random response from the intent
                        return (random.choice(i['responses']),x)

            results.pop(0)



print(response('hi'))



import random
sentences = [
  "heya",
  "Intake Capacity"
]

pred_tokens = map(tokenizer.tokenize, sentences)
pred_tokens = map(lambda tok: ["[CLS]"] + tok + ["[SEP]"], pred_tokens)
pred_token_ids = list(map(tokenizer.convert_tokens_to_ids, pred_tokens))

pred_token_ids = map(lambda tids: tids +[0]*(data.max_seq_len-len(tids)),pred_token_ids)
pred_token_ids = np.array(list(pred_token_ids))

predictions = model.predict(pred_token_ids).argmax(axis=-1)
#print(predictions)
#print(classes)
for text, label in zip(sentences, predictions):
  print("text:", text, "\nintent:", classes[label])
  print()
  c=0
  for i in range(len(classes)):
    if responses['labels'][i]==classes[label]:
      #x=responses.groupby('labels').get_group(i).responses
      # r = np.random.randint(0,x)
      resp = responses['responses'][c]
      print(resp)
    c=c+1
  print()

sentence=['hi']
def give_response(sentence):
    pred_tokens = map(tokenizer.tokenize, sentence)
    pred_tokens = map(lambda tok: ["[CLS]"] + tok + ["[SEP]"], pred_tokens)
    pred_token_ids = list(map(tokenizer.convert_tokens_to_ids, pred_tokens))

    pred_token_ids = map(lambda tids: tids +[0]*(data.max_seq_len-len(tids)),pred_token_ids)
    pred_token_ids = np.array(list(pred_token_ids))

    predictions = model.predict(pred_token_ids).argmax(axis=-1)
    print(predictions)
    print()
    print("\nintent:", classes[predictions[0]])
    print('response:',responses['responses'][predictions[0]])

give_response(sentence)





from flask import Flask, render_template, request, jsonify
#from flask_ngrok import run_with_ngrok
app = Flask(__name__, template_folder=main_path+'templates/UI')
#run_with_ngrok(app)

@app.route('/')
def index():
	return render_template('index2.html')

@app.route('/get')
def get_bot_response():
	message = request.args.get('msg')
	if message:
		message = message.lower()
		res,x=response(message)
        #x=str(x)
		return (str(x)+" "+str(res))
	return "Missing Data!"



if __name__ == "__main__":
	app.run(host='0.0.0.0',port=8080)
