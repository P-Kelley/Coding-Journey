#This is a showcase of how to implement basic api calls to an AWS EC2 instance.
#Note: You need to have config and credential files setup for your user to make AWS API calls



import boto3

def StartServer():
    ec2 = boto3.client("ec2")
    instance = ec2.describe_instances(InstanceIds = ['Specify Instances Here'])
    for pythonins in instance['Reservations']:
        for printout in pythonins['Instances']:
            state = printout['State']['Name']

    if state == 'running':
        print("This is alreay running")
        
    elif state == 'stopped':
        try:
            ec2.start_instances(InstanceIds = ['Specify Instances Here'])
        except:
            print("Error occured. Wait for a bit and try again. ")
        else:
            print("Server starting")
 

def RebootServer():
    ec2 = boto3.client("ec2")
    instance = ec2.describe_instances(InstanceIds = ['Specify Instances Here'])
    for pythonins in instance['Reservations']:
        for printout in pythonins['Instances']:
            state = printout['State']['Name']

    if state == 'running':
        ec2.reboot_instances(InstanceIds = ['Specify Instances Here'])
        print('Rebooting Sever')

    if state == 'stopped':
        print("This instance is stopped. Please start it before rebooting")
    
def StopServer():
    ec2 = boto3.client("ec2")
    instance = ec2.describe_instances(InstanceIds = ['Specify Instances Here'])
    for pythonins in instance['Reservations']:
        for printout in pythonins['Instances']:
            state = printout['State']['Name']

    if state == 'running':
        ec2.stop_instances(InstanceIds = ['Specify Instances Here'])
        print("Stopping Server")

    if state == 'stopped':
        print("This is alreay stopped")

