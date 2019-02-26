import discord
from discord.ext import commands
from discord.ext.commands import Bot
import asyncio
import json
import markovify
from rake_nltk import Rake

startup = True

history = []

with open('config.json') as f:
    config = json.load(f)

bot = commands.Bot(command_prefix=config['prefix'], self_bot=False)

def get_relevant_convos(message, context):
    r = Rake()
    r.extract_keywords_from_text(message)
    keywords = r.get_ranked_phrases()
    keywords = ' '.join(keywords).split(' ')
    keywords = [x.lower() for x in keywords]
    if keywords == ['']:
        print('no keywords found, using words longer that 3 chars')
        keywords = [x.lower() for x in message.split() if len(x) > 3]
    if not keywords:
        keywords = [x.lower() for x in message.split()]
        print('no keywords found, using all words')
    print(keywords)
    global history
    msg_since_relevant = 20
    relevant_convos = ''
    for m in history:
        # print(m)
        for word in m.split():
            if word.lower() in keywords:
                msg_since_relevant = 0
        if msg_since_relevant < context:
            relevant_convos = m + '\n' + relevant_convos
        msg_since_relevant += 1
    return relevant_convos

def get_response(message):
    context = 20
    response = ''
    while not response:
        relevant_convos = get_relevant_convos(message, context)
        if not relevant_convos:
            l = reversed(history[:100])
            relevant_convos = '\n'.join([x.lower() for x in l])
        print(relevant_convos)
        text_model = markovify.Text(relevant_convos)
        response = text_model.make_sentence()
        context += 10
        if context > 1000:
            break
    return response

@bot.command()
async def chat(ctx, *args):
    message = ' '.join(args)
    response = get_response(message)
    await ctx.send(response)


@bot.event
async def on_ready():
    global startup
    if startup:
        startup = False
        # print('Logged in')
        for chan_id in config['training channels']:
            chan = bot.get_channel(chan_id)
            hist_itr = chan.history(limit=config['training backlog'])
            print('adding words from channel #%d' % chan_id)
            async for m in hist_itr:
                history.append(m.content)
                print(m.content)

loop = asyncio.get_event_loop()
loop.create_task(bot.start(config['token']))

try:
    loop.run_forever()
finally:
    loop.stop()