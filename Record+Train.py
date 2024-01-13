import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, concatenate, Input
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.utils import to_categorical
import cv2
import os
from time import sleep


path = os.getcwd()
forward = "fd"
right = "rt"
left = "lt"
right_rotate = "rtr"
left_rotate = "ltr"

forward_folder = f"{os.path.join(path, forward)}\\"
right_folder = f"{os.path.join(path, right)}\\"
left_folder = f"{os.path.join(path, left)}\\"
rotate_right_folder = f"{os.path.join(path, right_rotate)}\\"
rotate_left_folder = f"{os.path.join(path, left_rotate)}\\"

folders = [forward_folder, right_folder, left_folder, rotate_right_folder, rotate_left_folder]

recording_number = 25

try:
    os.mkdir(forward_folder)
    os.mkdir(right_folder)
    os.mkdir(left_folder)
    os.mkdir(rotate_right_folder)
    os.mkdir(rotate_left_folder)
except:
    pass
#Instantiate cv2 cam (TAKES LESS NOW)
cam = cv2.VideoCapture(0)#Put 1 if better webcam availiable
width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

Images = np.empty((0, height, width))#set the image array to an empty one, with the shape of height width (To make the image shape)
labels = np.empty(0)
Lasers = np.empty((0, 5)) #Put 8 if 8 lasers in the end

def preprocess_image(image):
    # Grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Blur the image
    blurred_image = cv2.GaussianBlur(image, (5, 5), 4)

    # Sharpen the image
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    sharpened_image = cv2.filter2D(blurred_image, -1, kernel)

    # Enhance contrast (optional)
    clahe = cv2.createCLAHE(clipLimit=16.0, tileGridSize=(16, 16))
    contrast_enhanced = clahe.apply(sharpened_image)

    return contrast_enhanced

Images = np.empty((0, height, width))#set the image array to an empty one, with the shape of height width (To make the image shape)
labels = np.empty(0)
Lasers = np.empty((0, 5)) #Put 8 if 8 lasers in the end

def image_sort(path):
    images = []
    for element in os.listdir(path):
        if ".jpg" in element:
            images.append(element)
    return images

def text_sort(path):
    texts = []
    for element in os.listdir(path):
        if ".txt" in element:
            texts.append(element)
    return texts

def folder_value(folder):
    if folder == forward_folder:
        return 0
    elif folder == right_folder:
        return 1
    elif folder == left_folder:
        return 2
    elif folder == rotate_right_folder:
        return 3
    elif folder == rotate_left_folder:
        return 4
    else:
        raise "Problem with folder, the value given is not equal to a valid folder adress stated above!"

def rec_image(folder, times=1):
    for i in range(times):
        sleep(0.01)
        ret, image = cam.read()
        image = preprocess_image(image)
        cv2.imwrite(f"{folder}{len(image_sort(folder))}.jpg",image)

def rec_lasers(folder, times=1):
    #recieve lengths from 8 or 5 lasers
    for i in range(times):
        data = [1,2,3,4,5]
        file = open(f"{folder}{len(text_sort(folder))}.txt", "w")
        file.write(str(data))
        file.close()

def read_data(folder):
    global Images, labels, Lasers
    for i in range(len(image_sort(folder))):
        if i < len(image_sort(folder))-1:
            image = cv2.imread(f"{folder}{i+1}.jpg", cv2.IMREAD_GRAYSCALE)
            Images = np.append(Images, [image], axis=0)
            labels = np.append(labels, folder_value(folder))
    for i in range(len(text_sort(folder))):
        if i < len(text_sort(folder))-1:
            data = open(folder+text_sort(folder)[i],"r")
            data = data.read().strip('][').split(', ')
            Lasers = np.append(Lasers, [data], axis=0) # Load the LASER DATA
            
def process_predictions(tensor): #Has to be a numpy tensor, because tensorflow const cannot append
    final_tensor = np.array([])
    for prediction in tensor:
        i = 1
        for choice in prediction:
            #print(choice)
            if round(float(choice)) == 1:
                final_tensor = np.concatenate((final_tensor, [i]))
                #print("IT SHOULD CONCAT 1!")
            else:
                i+=1
    return tf.constant(np.round(final_tensor))

def process_predictions_combined(predictions_image, predictions_laser):
    max_indices_image = np.argmax(predictions_image, axis=1)
    max_indices_laser = np.argmax(predictions_laser, axis=1)
    
    # Combine max indices for image and laser predictions
    combined_indices = np.column_stack((max_indices_image, max_indices_laser))
    
    return tf.constant(np.round(combined_indices))
                
def train():
    global Images, labels, Lasers
    
    for folder in folders:
        read_data(folder)
        
    Lasers = Lasers.reshape((Lasers.shape[0], Lasers.shape[1], 1))

    print(Images.shape)
    print(labels.shape)
    print(Lasers.shape)

    # Split the data
    train_images, test_images, train_labels, test_labels = train_test_split(Images, labels, test_size=0.1, random_state=21, stratify=labels)
    train_lasers, test_lasers, _, _ = train_test_split(Lasers, labels, test_size=0.1, random_state=21, stratify=labels)
    #print(f"LASERS : {train_lasers},{train_lasers.shape}")
    #Preprocess the data
    train_images = tf.expand_dims(train_images, axis=-1)
    test_images = tf.expand_dims(test_images, axis=-1)
    
    train_images = tf.cast(train_images, dtype=tf.int32)
    test_images = tf.cast(test_images, dtype=tf.int32)
    """    
    train_lasers = tf.expand_dims(train_lasers, axis=-1)
    train_lasers = tf.expand_dims(train_lasers, axis=-1)"""
    
    train_lasers = tf.strings.to_number(train_lasers, out_type=tf.float64)
    test_lasers = tf.strings.to_number(test_lasers, out_type=tf.float64)
    
    train_labels = train_labels.astype(int)
    test_labels = test_labels.astype(int)
    laser_count = len(Lasers[0])
    
    print(f"LASERS : {train_lasers},{train_lasers.shape}")
    # CONV2D
    model1 = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(height, width, 1), name="Image_Input"),
        tf.keras.layers.Conv2D(16, (3, 3), activation=tf.keras.activations.relu, name="Image_1_Conv2D"),
        tf.keras.layers.MaxPooling2D((2, 2), name="Image_2_Pooling2D"),
        tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.activations.relu, name="Image_3_Conv2D"),
        tf.keras.layers.MaxPooling2D((2, 2), name="Image_4_Pooling2D"),
        tf.keras.layers.Conv2D(64, (3, 3), activation=tf.keras.activations.relu, name="Image_5_Conv2D"),
        tf.keras.layers.MaxPooling2D((2, 2),name="Image_6_Pooling2D"),
        tf.keras.layers.Flatten(name="Image_7_Flatten"),
        tf.keras.layers.Dropout(0.5,name="Image_8_Dropout"),
        tf.keras.layers.Dense(5, activation=tf.keras.activations.softmax)
        
    ])
    #CONV1D
    model2 = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(laser_count, 1), name="Laser_Input"),
        tf.keras.layers.Conv1D(32, kernel_size=3, activation=tf.keras.activations.relu, name="Laser_Conv1D"),
        tf.keras.layers.Flatten(name="Laser_flatten"),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(5, activation=tf.keras.activations.softmax)
    ])
    #FC MODEL
    model3 = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=2, name="Combined_Input"),
        tf.keras.layers.Dense(32, activation=tf.keras.activations.relu, name="Combined_1_Dense"),
        tf.keras.layers.Dense(64, activation=tf.keras.activations.relu, name="Combined_2_Dense"),
        tf.keras.layers.Dropout(0.3, name="Combined_3_Dropout"),
        tf.keras.layers.Dense(5, activation=tf.keras.activations.softmax, name="Combined_OUTPUT")
    ])
    #Compile and train the models
    model1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                          loss=tf.keras.losses.sparse_categorical_crossentropy, #Because it is category, mse is for numbers
                          metrics=["accuracy"])

    model2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.003),
                          loss=tf.keras.losses.sparse_categorical_crossentropy, #Because it is category, mse is for numbers
                          metrics=["accuracy"])

    model3.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
                          loss=tf.keras.losses.sparse_categorical_crossentropy, #Because it is category, mse is for numbers
                          metrics=["accuracy"])
    #print(train_labels)
    model1.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))
    """print("Train lasers shape:", train_lasers.shape)
    print("Test lasers shape:", test_lasers.shape)
    """
    
    model2.fit(train_lasers, train_labels, epochs=10, validation_data=(test_lasers, test_labels))
    
    predictions_model1 = model1.predict(train_images)
    predictions_model2 = model2.predict(train_lasers)

    # Concatenate predictions along the axis corresponding to models (axis=1)
    """print(len(train_images)+len(train_lasers))
    print("LENGTH of predictions",len(predictions))
    print("LENGTH of predictions[0]", len(predictions[0]))
    print(np.array(predictions).shape)
    print(predictions)"""
    predictions = process_predictions_combined(predictions_model1, predictions_model2)
    print(predictions)
    print(len(predictions))
    #predictions = process_predictions(predictions)
    #predictions = predictions #- 1
    print(f"Predictions : {predictions.shape}")
    print(f"Labels : {train_labels.shape}")
    
    model3.fit(predictions-1, train_labels, epochs=25)
    return model1, model2, model3
    
    
def test():
    sleep(1)
    cam = cv2.VideoCapture(0)
    ret, image = cam.read()
    images = np.empty((0, height, width))
    images = np.append(images, [preprocess_image(image)], axis=0)
    images = images.reshape((1, 480, 640, 1))
    print(images, images.shape)
    lasers = np.array([[1, 2, 3, 4, 5]])
    lasers = lasers.reshape((1,5,1))
    
    predictions_model1 = model1.predict(images)
    predictions_model2 = model2.predict(lasers)
    combined = process_predictions_combined(predictions_model1, predictions_model2)
    combined = combined - 1
    print(combined, combined.shape)

    pred3 = model3.predict(combined)
    print(pred3)
    print(pred3.argmax())
        
while True:
    main = input("1(fd), 2(rt), 3(lt), 4(rtr), 5(ltr), 6(Train), 7(Test)")
    #Instead of input, get values from recv
    #Channel 1 to fwd
    #Channel 2 to rt and lt (depends on the number)
    #Channel 3 constant (set drone to hover / constant throttle)
    #Channel 4 to rtr and ltr (depends on number)

    if main == "1":
        rec_image(forward_folder, recording_number)
        rec_lasers(forward_folder, recording_number)
    elif main == "2":
        rec_image(right_folder, recording_number)
        rec_lasers(right_folder, recording_number)
    elif main == "3":
        rec_image(left_folder, recording_number)
        rec_lasers(left_folder, recording_number)
    elif main == "4":
        rec_image(rotate_right_folder, recording_number)
        rec_lasers(rotate_right_folder, recording_number)
    elif main == "5":
        rec_image(rotate_left_folder, recording_number)
        rec_lasers(rotate_left_folder, recording_number)
    elif main == "6":
        model1, model2, model3 = train()
    elif main == "7":
        test()
    else:
    
        print("Not an option")

