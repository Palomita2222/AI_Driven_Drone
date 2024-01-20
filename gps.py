import serial
import pynmea2
from geopy.distance import geodesic
from math import atan2, degrees, sin, cos

def parse_gps_data(gps_data): #Parsing the GPS recieved sigal
    try:
        msg = pynmea2.parse(gps_data)
        if msg.sentence_type == 'GGA':
            return msg.latitude, msg.longitude 
    except pynmea2.ParseError as e:
        print(f"Error parsing GPS data: {e}")
    return None, None

def read_gps_data(serial_port): #Recieving the GPS signal, then sending to parse
    with serial.Serial(serial_port, baudrate=9600, timeout=1) as ser: #The port is /dev/ttyUSB0 ???
        while True:
            gps_data = ser.readline().decode('utf-8')
            if gps_data.startswith('$'):
                return parse_gps_data(gps_data)

def calculate_bearing(current_coords, destination_coords): #2D bearing (height is not taken into account)
    #Calculate the direction between two points (bearing)
    lat1, lon1 = current_coords
    lat2, lon2 = destination_coords

    delta_lon = lon2 - lon1 #The change in x

    x = atan2(
        sin(delta_lon) * cos(lat2),
        cos(lat1) * sin(lat2) - (sin(lat1) * cos(lat2) * cos(delta_lon))
    )

    # Convert radians to degrees
    bearing = (degrees(x) + 360) % 360
    return bearing

def calculate_distance(current_coords, destination_coords):
    return geodesic(current_coords, destination_coords).meters

if __name__ == "__main__": #So i can import the file from another without executing anything
    serial_port = "/dev/ttyUSB0"  #Replace with GPS port

    #Read current GPS coordinates
    current_lat, current_lon = read_gps_data(serial_port)

    if current_lat is not None and current_lon is not None:
        print(f"Current Location: Latitude {current_lat}, Longitude {current_lon}")

        #For test (for now)
        dest_lat = float(input("Enter destination latitude: "))
        dest_lon = float(input("Enter destination longitude: "))

        # Calculate direction (bearing) to the destination
        destination_coords = (dest_lat, dest_lon)
        #bearing = calculate_bearing((current_lat, current_lon), destination_coords)
        distance = calculate_distance((current_lat, current_lon), (dest_lat, dest_lon))
        print(f"Bearing to destination: {bearing} degrees")
        print(f"Distance to destination: {distance} meters")
    else:
        print("Unable to obtain GPS coordinates.")