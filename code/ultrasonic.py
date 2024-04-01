import RPi.GPIO as GPIO
import time

# GPIO pins
TRIG = [6,17, 22, 24, 9]
ECHO = [5,27,4, 23, 10]
distances = [i for i in range(len(TRIG))]

def setup():
    GPIO.setmode(GPIO.BCM)
    for snumber in range(len(TRIG)):
        GPIO.setup(TRIG[snumber], GPIO.OUT)
        GPIO.setup(ECHO[snumber], GPIO.IN)

def distance(snumber):
        GPIO.output(TRIG[snumber], True)
        time.sleep(0.002)
        GPIO.output(TRIG[snumber], False)
        pulse_start = time.time()
        pulse_end = time.time()
        i = 0
        while GPIO.input(ECHO[snumber]) == 0:
            pulse_start = time.time()
            i+=1
            if i >= 250:
                return lastdistances[snumber]

        while GPIO.input(ECHO[snumber]) == 1:
            pulse_end = time.time()

        pulse_duration = pulse_end - pulse_start

        # Speed of sound in air at 20°C is approximately 343 meters per second
        distance = pulse_duration * 17150

        distance = round(distance, 2)
        return distance


if __name__ == '__main__':
    try:
        setup()
        while True:
            time.sleep(0.2)
            lastdistances = distances.copy()
            distances = []
            for snumber in range(len(TRIG)):
                dist = distance(snumber)
                if str(dist) != "" and dist > 0:
                    distances.append(dist)
                else:

                    if str(dist) == "":
                        print(r"str(dist) == ''")
                    if dist < 0:
                        print(f"dist < 0 {dist}")

                    distances.append(lastdistances[snumber])

            print(f"Distance: {distances}")
            time.sleep(0.1)
    except KeyboardInterrupt:
        GPIO.cleanup()
        print("Ctrl+C Pressed")
