import serial
import pynmea2
from geopy.distance import geodesic
from math import sin, cos, atan2, degrees

# Serial port settings
port = "/dev/ttyACM0"
baudrate = 9600
ser = serial.Serial(port, baudrate, timeout=1)

def get_data():
    try:
        # Open serial port

        # Read a line from the serial port
        nmea_sentence = ser.readline().decode('utf-8')

        # Check if it's a GPRMC sentence
        if nmea_sentence.startswith('$GPRMC'):
            try:
                    # Parse the sentence
                msg = pynmea2.parse(nmea_sentence)

                    # Check if the data is valid
                if msg.status == 'V':
                    pass
                    #print("Data is void")
                else:
                    # Extract latitude and longitude
                    latitude = msg.latitude
                    longitude = msg.longitude
                    #print("Latitude:", latitude)
                    #print("Longitude:", longitude)
                    if latitude is not None:
                        print(latitude,longitude)
                        return (latitude,longitude)
                    else:
                        pass
            except pynmea2.ParseError as e:
                pass
                #print(f'Parse error: {e}')
        else:
            pass

    except serial.SerialException as e:
        pass
        #print("Serial port error: ", e)


def calculate_bearing(current_coords, destination_coords): #2D bearing (height is not taken into account)
    #Calculate the direction between two points (bearing)
    lat1, lon1 = current_coords[0], current_coords[1]
    lat2, lon2 = destination_coords[0], destination_coords[1]

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


if __name__=="__main__":
    objective = (float(input("Enter Objective Lat : ")), float(input("Enter Objective Lon : ")))
    while True:

        pos = get_data()
        print(pos)
        if type(pos) is tuple:
            print(pos)
            bearing = calculate_bearing(pos, objective)
            distance = calculate_distance(pos, objective)
            print(f"Bearing to Objective : {bearing}, \n Distance to Objective : {distance}")
            break
