import discord
from discord.ext import commands
from discord.ext.commands import Bot
import asyncio
import json
import markovify
from rake_nltk import Rake
from fastai.text import *
from fastai.data_block import ItemBase, ItemList
from torch.utils.data import Dataset
import csv
from itertools import chain
import os
import pickle
import numpy as np
from sklearn import model_selection
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from keras.models import Sequential
from keras import layers


startup = True

h = []

with open('config.json') as f:
    config = json.load(f)

bot = commands.Bot(command_prefix=config['prefix'], self_bot=False)

@bot.event
async def on_ready():
    global startup
    global h
    if startup:
        startup = False
        print('Logged in')
        for chan_id in config['training channels']:
            chan = bot.get_channel(chan_id)
            hist_itr = chan.history(limit=config['nn backlog'])
            print('adding words from channel #%d' % chan_id)
            async for m in hist_itr:
                h.append(m.content)
                # print(m.content)
        h = list(reversed(h))
        all_pairs = []
        for x, item1 in enumerate(h):
            for y, item2 in enumerate(h):
                if x != y:
                    all_pairs.append({'relevant':1 if x==y+1 else 0, 'sentence 1':item1, 'sentence 2':item2})

        sentences = [i['sentence 1']+'            '+i['sentence 2'] for i in all_pairs]
        y = [i['relevant'] for i in all_pairs]

        sentences_train, sentences_test, y_train, y_test = model_selection.train_test_split(sentences, y, test_size = 0.25, random_state = 1000)
        # print(all_pairs)
        # print('\n')
        # print(sentences_train)
        # print('\n')
        # print(sentences_test)
        # print('\n')
        # print(y_train)

        vectorizer = CountVectorizer()

        vectorizer.fit(sentences_train)
        X_train = vectorizer.transform(sentences_train)
        X_test = vectorizer.transform(sentences_test)
        print(X_train)
        print(X_test)

        input_dim = X_train.shape[1]  # Number of features

        model = Sequential()
        model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
        model.summary()

        history = model.fit(X_train, y_train, epochs = 100, verbose = False, validation_data = (X_test, y_test),batch_size = 10)

        loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
        print("Training Accuracy: {:.4f}".format(accuracy))
        loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
        print("Testing Accuracy:  {:.4f}".format(accuracy))

        print('done')


loop = asyncio.get_event_loop()
loop.create_task(bot.start(config['token']))

try:
    loop.run_forever()
finally:
    loop.stop()