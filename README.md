# AI Driven Drone
Hey, This is one of my biggest projects yet. I am currently building a drone and I want to add an AI algorithm to it, so It can go to any GPS location, avoiding obstacles on the way.
This is one of my biggest projects and its for school too.

For this project i'm using:
<ol>
  <li>
    Hawk's Work F450 Drone Kit + Pixhawk Flight Computer
    <ul>
      <li>
        <a href="https://www.amazon.es/dp/B09SZ74YFK?ref=emc_s_m_5_i_n">Link 1</a>
      </li>
    </ul>
  </li>
  <li>
    Time Of Flight VL53L0X Laser distance sensors x 5
    <ul>
      <li>
        <a href="https://www.amazon.es/TECNOIOT-VL53L0X-Flight-Distance-GY-VL53L0XV2/dp/B084BTP479/ref=sr_1_16">Link 2</a>
      </li>
    </ul>
  </li>
  <li>
    Ultrasonic Distance Sensor HC-SR04
    <ul>
      <li>
        <a href="https://www.amazon.es/AZDelivery-Distancia-Ultrasónico-Raspberry-incluido/dp/B07TKVPPHF/ref=sr_1_7">Link 3</a>
      </li>
    </ul>
  <li>
    Jetson Nano Microcomputer (for quick and reliable AI predictions and communication to the Flight Computer)
    <ul>
      <li>
        <a href="https://www.amazon.es/Waveshare-Jetson-Nano-Developer-Kit/dp/B07QWLMR24/ref=sr_1_2_sspa">Link 4</a>
      </li>
    </ul>
  </li>
  <li>
    Ender-3 V2 NEO 3D printer (For 3d printing the supports for the lasers and height sensor ++)
    <ul>
      <li>
        <a href="https://www.amazon.es/Creality-V2-Neo-Preinstalado-Principiantes/dp/B0BQJD1QFT/ref=sr_1_1_sspa">Link 5</a>
      </li>
    </ul>
  </li>
  <li>
    Python 3.11.4 + Jupyter Notebook 6.5.4
    <ul>
      <li>
        <a href="https://www.python.org">Link 6</a>
      </li>
    </ul>
  </li>
  </ol>
    <h5 color="red">COST : 920€~</h5>


The AI consists of <b>three supervised models</b>, A 2D CNN for images ( For the camera on board ), a 1D CNN ( For the distance sensors onboard ) and a FC NN that recieves both NN's action and decides the best one for the job.

I will get lots of data by flying myself and the program I am developing will capture the data and save it into different folders for classification.
There will be 5 folders, for 5 actions the Drone can take. Forward, Right, Left, Rotate Right and Rotate Left. Whenever I fly the drone in a direction, the drone will take a picture, as well as save the distances to the direction's directory.


Later, I will train the models and when everything is working, I will save the model to be able to import it to a new code file.
This new code file will take in a GPS location, and its own, and will calculate the line of shortest path between the drone and the objective (in 2D). It will then (by communicating through from the Jetson Nano to the Pixhawk Flight Computer) move towards it. Only when the distance sensors detect an obstacle close by, will the drone start making predictions on where to go, and hopefully, it will avoid obstacles.

<img src="/process/IMG_0337.JPG">
<img src="/models/2DCNN.png", width=500, height=3000>

