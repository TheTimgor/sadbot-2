import discord
from discord.ext import commands
from discord.ext.commands import Bot
import asyncio
import json
import markovify
from rake_nltk import Rake
import nltk

startup = True

with open('config.json') as f:
    config = json.load(f)

bot = commands.Bot(command_prefix=config['prefix'], self_bot=False)

async def get_hist(chan):
    history = []
    hist_itr = chan.history(limit=config['history backlog'])
    print('adding words from channel #%d' % chan_id)
    async for m in hist_itr:
        history.insert(0,m.content)
        print(m.content)
    return history


async def get_response(message,chan):
    text_model = await markovify.Text(get_hist(chan))
    response = text_model.make_sentence()
    return response

@bot.command()
async def chat(ctx, *args):
    message = ' '.join(args)
    response = await get_response(message)
    await ctx.send(response)


@bot.event
async def on_ready():
    global startup
    if startup:
        startup = False
        print('Logged in')

loop = asyncio.get_event_loop()
loop.create_task(bot.start(config['token']))

try:
    loop.run_forever()
finally:
    loop.stop()