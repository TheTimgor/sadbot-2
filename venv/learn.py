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

@bot.event
async def on_ready():
    global startup
    global history
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
        history = reversed(history)


loop = asyncio.get_event_loop()
loop.create_task(bot.start(config['token']))

try:
    loop.run_forever()
finally:
    loop.stop()