#This is a basic setup for a discord bot to start/stop/reboot a specfic EC2 instance or instances.
#With some slight changes it is also possible to specify the instances in the discord message

import discord
import boto3



def StartServer(message):
    ec2 = boto3.client("ec2")
    instance = ec2.describe_instances(InstanceIds = ['Specify Instances Here'])
    for pythonins in instance['Reservations']:
        for printout in pythonins['Instances']:
            state = printout['State']['Name']

    if state == 'running':
        message.channel.send("This instance is alreay running")

    elif state == 'stopped':
        try:
            ec2.start_instances(InstanceIds = ['Specify Instances Here'])
            message.channel.send("Server starting")
        except:
            message.channel.send("Error occured. Please try again in a few minutes")
            


def RebootServer(message):
    ec2 = boto3.client("ec2")
    instance = ec2.describe_instances(InstanceIds = ['Specify Instances Here'])
    for pythonins in instance['Reservations']:
        for printout in pythonins['Instances']:
            state = printout['State']['Name']

    if state == 'running':
        ec2.reboot_instances(InstanceIds = ['Specify Instances Here'])
        message.channel.send('Rebooting Sever')
    
def StopServer(message):
    ec2 = boto3.client("ec2")
    instance = ec2.describe_instances(InstanceIds = ['Specify Instances Here'])
    for pythonins in instance['Reservations']:
        for printout in pythonins['Instances']:
            state = printout['State']['Name']

    if state == 'running':
        ec2.stop_instances(InstanceIds = ['Specify Instances Here'])
        message.channel.send("Stopping Server")


intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)


@client.event
async def on_ready():
    print(f'We have logged in as {client.user}')

@client.event
async def on_message(message):
    messageuse = str(message.content).lower()

    if message.author == client.user:
        return
    
    if '!help' in messageuse:
        await message.channel.send("Avalible commands are !start_server_please : !stop_server_please : !reboot_server_please")
        return
    
    if '!start_server' in messageuse:
        await message.channel.send("Starting server")
        StartServer(message)
        return

    if '!stop_server' in messageuse:
        await message.channel.send("Stopping Server")
        StopServer(message)
        return

    if '!reboot_server' in messageuse:
        await message.channel.send("Rebooting Server")
        RebootServer(message)
        return


client.run('Enter your discord bot token here')
