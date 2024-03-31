import socket
import ast

# Define server address and port
SERVER_HOST = '127.0.0.1'  # Replace with the actual IP address of the server
SERVER_PORT = 12345                # Same port as the server

# Create a socket object
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Connect to the server
client_socket.connect((SERVER_HOST, SERVER_PORT))
message = ""

AI = False
Manual = False
EBrake = False
Training = False

# Receive data from the server
while True:
    data = ast.literal_eval(client_socket.recv(1024).decode())
    if not data:
        break
    if data[0] == "Autonomous Mode Sent":
        AI = not AI
        Manual = False
        EBrake = False
        message = f"['Autonomous Mode Triggered : {AI}',{AI},{Manual},{EBrake},{Training}]"
        print(f"lat : {data[1][0]} lon : {data[1][1]}")
    elif data[0] == "Manual Mode Sent":
        Manual = not Manual
        AI = False
        EBrake = False
        message = f"['Manual Mode Triggered : {Manual}',{AI},{Manual},{EBrake},{Training}]"
        print(data)
    elif data[0] == "E-Brake Sent":
        EBrake = not EBrake
        AI = False
        Manual = False
        Training = False
        message = f"['E-Brake Triggered : {EBrake}',{AI},{Manual},{EBrake},{Training}]"
        print(data)
    elif data[0] == "Training Mode Sent":
        Training = not Training
        AI = False
        Manual = True
        EBrake = False
        message = f"['Training Mode Triggered : {Training}',{AI},{Manual},{EBrake},{Training}]"
    print(message)
    client_socket.send(message.encode())


# Close the connection with the server
client_socket.close()
