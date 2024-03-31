import serial
import pynmea2
import time

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
                        return [latitude,longitude]
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





if __name__=="__main__":
    positions = []
    for i in range(100):
        data = get_data()
        if type(data) == list:
            positions.append(data)
        else:
            pass
    print(f"Positions (lat, lon): \n {positions}")
