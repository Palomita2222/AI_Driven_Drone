import tkinter as tk
from tkinter import ttk
import socket
import threading
import ast

# Define host and port
HOST = '0.0.0.0'
PORT = 12345

# Create a socket object
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind the socket to the address and port
server_socket.bind((HOST, PORT))

# Listen for incoming connections
server_socket.listen(5)

print('Server is listening...')

# List to hold connected client sockets
clients = []

log = ""
def fix_log_length(log):
    llist = log.split("\n")
    if len(llist) > 10:
        llist = llist[-10:]
        lastlog = """"""
        for element in llist:
            lastlog += f"\n{element}"
        print(lastlog)
        return lastlog
    else:
        return log

def color_update():
    if "True" in str(AutVal.get()):
        AutStyle.configure("Aut.TLabel", foreground="green")
    else:
        AutStyle.configure("Aut.TLabel", foreground="red")
    if "True" in str(ManVal.get()):
        ManStyle.configure("Man.TLabel", foreground="green")
    else:
        ManStyle.configure("Man.TLabel", foreground="red")
    if "True" in str(EBVal.get()):
        EBStyle.configure("EB.TLabel", foreground="green")
    else:
        EBStyle.configure("EB.TLabel", foreground="red")
    if "True" in str(TrVal.get()):
        TrStyle.configure("Tr.TLabel", foreground="green")
    else:
        TrStyle.configure("Tr.TLabel", foreground="red")

def handle_client(client_socket, address):
    global log
    # Add client socket to the list
    clients.append(client_socket)
    LVariable.set("Server Running : Drone Connected")
    labelstyle.configure("Custom.TLabel", foreground="green")
    log += f"\n* Drone Connected *"
    logs.set(fix_log_length(log))

    while True:
        try:
            data = ast.literal_eval(client_socket.recv(1024).decode())
            print(data)
            if not data:
                break

            log += f"\n{data[0]}"
            logs.set(fix_log_length(log))
            AI, MANUAL, EBRAKE, TRAINING = data[1], data[2], data[3], data[4]
            AutVal.set(f"Autonomous : {AI}")
            ManVal.set(f"Manual : {MANUAL}")
            EBVal.set(f"EBrake : {EBRAKE}")
            TrVal.set(f"Training : {TRAINING}")
            color_update()
        except ConnectionResetError:
            print(f'Connection with {address} has been closed.')
            clients.remove(client_socket)
            LVariable.set("Server Running : Drone Disconnected")
            labelstyle.configure("Custom.TLabel", foreground="orange")
            log += "\n* Drone Disconnected *"
            logs.set(fix_log_length(log))
            break

def accept_connections(server_socket):
    while True:
        # Accept connection from client
        client_socket, address = server_socket.accept()
        print(f'Connection from {address} has been established!')
        client_thread = threading.Thread(target=handle_client, args=(client_socket, address))
        client_thread.start()

accept_thread = threading.Thread(target=accept_connections, args=(server_socket,))
accept_thread.start()

def get_entry_data():
    lat = entry_lat.get()
    lon = entry_lon.get()
    return lat,lon

# Function to send data to all clients
def send_data(data):
    for client in clients:
        client.send(data.encode())

# Tkinter GUI
def send_message(message):
    global log
    if message == 1:
        data = ["Autonomous Mode Sent", get_entry_data()]
    elif message == 2:
        data = ["Manual Mode Sent"]
    elif message == 3:
        data = ["E-Brake Sent"]
    elif message == 4:
        data = ["Training Mode Sent"]
    log += f"\n{data[0]}"
    logs.set(fix_log_length(log))
    send_data(str(data))

root = tk.Tk()
root.title("Drone Server")
LVariable = tk.StringVar()
LVariable.set("Server Running : No Drone Connected")
labelstyle = ttk.Style()
labelstyle.configure("Custom.TLabel", foreground="red")
label = ttk.Label(root, textvariable=LVariable, font=("Arial", 20), style="Custom.TLabel")
label.pack(padx=10, pady=10)

#STYLES

AutStyle = ttk.Style()
AutStyle.configure("Aut.TLabel", foreground="red")

ManStyle = ttk.Style()
ManStyle.configure("Man.TLabel", foreground="red")

EBStyle = ttk.Style()
EBStyle.configure("EB.TLabel", foreground="red")

TrStyle = ttk.Style()
TrStyle.configure("Tr.TLabel", foreground="red")


# Buttons to send data to clients
btn_frame = ttk.Frame(root)
btn_frame.pack(pady=10)

button_style = ttk.Style()
button_style.configure("Big.TButton", font=("Arial", 16))

takeoff_btn = ttk.Button(btn_frame, text="AUTONOMOUS", command=lambda: send_message(1), width=15, style="Big.TButton")
takeoff_btn.grid(row=0, column=0, padx=5)

manual_btn = ttk.Button(btn_frame, text="MANUAL", command=lambda: send_message(2), width=15, style="Big.TButton")
manual_btn.grid(row=0, column=1, padx=5)

ebrake_btn = ttk.Button(btn_frame, text="E-BRAKE", command=lambda: send_message(3), width=15, style="Big.TButton")
ebrake_btn.grid(row=0, column=2, padx=5)

train_btn = ttk.Button(btn_frame, text="TRAIN", command=lambda: send_message(4), width=15, style="Big.TButton")
train_btn.grid(row=0, column=3, padx=5)

label_lat = tk.Label(btn_frame, text="Lat:", font=("Arial", 16))
label_lat.grid(row=1, column=0, padx=5, pady=3)

entry_lat = tk.Entry(btn_frame, font=("Arial", 16))
entry_lat.grid(row=1, column=1, padx=5, pady=5)

label_lon = tk.Label(btn_frame, text="Lon:", font=("Arial", 16))
label_lon.grid(row=2, column=0, padx=5, pady=3)

entry_lon = tk.Entry(btn_frame, font=("Arial", 16))
entry_lon.grid(row=2, column=1, padx=5, pady=5)

AutVal = tk.StringVar()
AutVal.set("Autonomous : False")
AutLabel = ttk.Label(btn_frame, textvariable=AutVal, font=("Arial", 16), style="Aut.TLabel")
AutLabel.grid(row=2, column=2, padx=5)

ManVal = tk.StringVar()
ManVal.set("Manual : False")
ManLabel = ttk.Label(btn_frame, textvariable=ManVal, font=("Arial", 16), style="Man.TLabel")
ManLabel.grid(row=2, column=3, padx=5)

EBVal = tk.StringVar()
EBVal.set("EBrake : False")
EBLabel = ttk.Label(btn_frame, textvariable=EBVal, font=("Arial", 16), style="EB.TLabel")
EBLabel.grid(row=3, column=2, padx=5)

TrVal = tk.StringVar()
TrVal.set("Training : False")
TrLabel = ttk.Label(btn_frame, textvariable=TrVal, font=("Arial", 16), style="Tr.TLabel")
TrLabel.grid(row=3, column=3, padx=5)


logs = tk.StringVar()
Llogs = ttk.Label(root, textvariable=logs, font=("Courier New", 16))
Llogs.pack()

root.mainloop()

server_socket.close()
