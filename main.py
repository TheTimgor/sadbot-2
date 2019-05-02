from discord.ext import commands
import discord
import asyncio
import json
import Chat
import traceback

startup = True

with open('config.json') as f:
    config = json.load(f)

bot = commands.Bot(command_prefix=config['prefix'], self_bot=False)


async def get_hist(chan, lim):
    history = []
    hist_itr = chan.history(limit=lim)
    async for m in hist_itr:
        history.insert(0, m.content)
        print(m.content)
    print(history)
    return history


@bot.command()
async def train(ctx, n):
    if "sudoers file" in [y.name for y in ctx.author.roles]:
        await ctx.send('This operation takes a while, please do not run it too frequently')
        if not n:
            n = config['history backlog']
        print('getting history')
        h = await get_hist(ctx.channel, int(n))
        print('generating model')
        Chat.generate_model(h)
        await ctx.send('Done!')
    else:
        await ctx.send(f'`{ctx.author.display_name} is not in the sudoers file. This incident will be reported.`')


@bot.command()
async def chat(ctx, *, message):
    async with ctx.channel.typing():
        print('getting response')
        response = Chat.get_response(message)
    await ctx.send(response)


@bot.command()
async def whoops(ctxs):
    raise Exception('you fool. you absolute buffoon. you think you can challenge me in my own realm? you think you can rebel against my authority? you dare come into my house and upturn my dining chairs and spill coffee grounds in my Keurig? you thought you were safe in your chain mail armor behind that screen of yours. I will take these laminate wood floor boards and destroy you. I didn’t want war. but i didn’t start it.')


@bot.event
async def on_command_error(ctx, error):
    await ctx.send(f'oopsie woopsie! i made a fucky wucky!\n{error}')
    raise error


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
