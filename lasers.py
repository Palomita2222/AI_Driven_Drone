import smbus
import time

#I2C addresses for each sensor (Need to correctly define them)
sensor_addresses = [0x29, 0x30, 0x31, 0x32, 0x33]

bus = smbus.SMBus(1)  #1 indicates the I2C bus number on Jetson Nano

# Function to read distance from a sensor
def read_distance(sensor_address):
    # Write to register to initiate measurement
    bus.write_byte_data(sensor_address, 0x00, 0x01)

    # Wait for measurement to complete
    while not (bus.read_byte_data(sensor_address, 0x4D) & 0x01):
        time.sleep(0.01)

    # Read distance
    distance = bus.read_word_data(sensor_address, 0x14)
    return distance

def read_distances():
  readings = []
  for sensor in sensor_adresses:
    readings.append(read_distance(sensor))
  return readings

if __name__ == "__main__":
  # Read distances from all sensors
  for address in sensor_addresses:
      distance = read_distance(address)
      print(f"Sensor at address {hex(address)} - Distance: {distance} mm")
  
  # Close the I2C bus
  bus.close()
