"""
Face Recognizer.py  is to perform Face detection and recognition continuously on the input image using
dlib library.
HOG detection is used for Face Detection.

"""
import pandas as pd
from sklearn.externals import joblib
import cv2
import dlib
import pickle
import numpy as np
import time
import distutils
import sklearn.neighbors.typedefs
import configparser


# Load the Knn model
knn = joblib.load('Dependencies/KNN_classifier_model.sav')

# Load the encodings for train images
data = pickle.loads(open('Dependencies/Encodings', 'rb').read())
print("\nNumber of encodings loaded:  ", len(data))

# Store face detector model, shape predictor model and face recognition model as individual variables.
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('Dependencies/shape_predictor_5_face_landmarks.dat')
face_recognition_model = dlib.face_recognition_model_v1('Dependencies/dlib_face_recognition_resnet_model_v1.dat')

# Tweak the min_distance parameter to get minimised false recognition
config = configparser.ConfigParser()
config.read('Dependencies/config.ini')
min_distance = config.getfloat( "Threshold_initialization" , 'min_distance')
print( "Initialized threshold: " , min_distance)

# Specify colours for display bounding box and text.
frame_rect_color = (0, 255, 0)
frame_text_color = (0, 0, 0)


def read_image(filename):
    """
    This function reads the image specified by its filename
    :param filename: It is a string containing the filename along with the extension of image
    :return: It returns image stored in the specified filename
    """
    image = cv2.imread(filename, 1)
    return image


def draw_rectangle(image, coordinates_1, coordinates_2, rect_color, filled_color):
    """
    This function draws rectangle on the image passed and according to the coordinates specified.
    :param image: Image on which we need to draw a rectangle
    :param coordinates_1: It is a tuple containing value of left, top points of rectangle
    :param coordinates_2: It is a tuple containing value of right, bottom points of rectangle
    :param rect_color: It is a tuple specifying colour of the rectangle
    :param filled_color: It is a boolean value used to either fill the rectangle with color or not.
    :return: It returns image with drawn rectangle
    """
    if filled_color:
        cv2.rectangle(image, coordinates_1, coordinates_2, rect_color, cv2.FILLED)
    else:
        cv2.rectangle(image, coordinates_1, coordinates_2, rect_color, 1)
    return image


def write_text(image, message, coord, text_color):
    """

    :param image: Image on which we need to write a text
    :param message: String containing text to be written on the image
    :param coord: It is tuple containing coordinates from where the text has to be written
    :param text_color: It is tuple specifying color of the text
    :return: It returns image with text written on it.
    """
    cv2.putText(image, message, coord, cv2.FONT_HERSHEY_DUPLEX, 0.5, text_color, 1)
    return image


def L2_distance(face_encoding, index_list):
    """
    This function calculates the L2 distance between each of the encodings of N Neighbours with the encodings of the
    face detected.

    :param face_encoding: It gets a 128-dimensional list of facial encoding of face detected in the webcam or video feed
    :param index_list: It contains the index values of n nearest neighbours returned by the knn classifier.
    :return: A string indicating the name of the person recognised based on the threshold(min_distance) we set.
    """
    database_list = [tuple([data[index][0], np.linalg.norm(
        face_encoding - data[index][1])]) for index in index_list]
    database_list.sort(key=lambda x: x[1])
    if database_list[0][1] > min_distance:
        duplicate = list(database_list[0])
        duplicate[0] = 'unknown'
        database_list.insert(0, tuple(duplicate))
    return database_list

def sharpen(image):
    """
    This function is to apply image processing technique - sharpening to the image.
    :param image: Pass the input image.
    :return: Returns the sharpened image.
    """
    # Creating sharpening kernel, it must equal to one eventually
    kernel_sharpening = np.array([[-1, -1, -1],
                                  [-1, 9, -1],
                                  [-1, -1, -1]])
    # applying the sharpening kernel to the input image.
    sharpened = cv2.filter2D(image, -1, kernel_sharpening)
    return sharpened


def resize_image(image, image_dimension):
    """
    This function resizes the image according to the dimension we provide
    :param image: It is the image which needs to be resized
    :param image_dimension: It is tuple containing image dimension to which the original image to be resized
    :return: It returns the resized image
    """
    image = cv2.resize(image, image_dimension)
    return image


def video_write(frame_array):
    """

    :param frame_array: It is a list containing frames from the camera feed
    :return: It returns none
    """
    print(len(frame_array))
    fourcc = cv2.VideoWriter_fourcc(*"DIVX")
    writer = cv2.VideoWriter('final_video5.avi', fourcc, 13, (frame_array[0].shape[1], frame_array[0].shape[0]), True)
    for frame in frame_array:
        writer.write(frame)
    writer.release()


def face_recognizer():
    """
    This function detects and recognises face fetched through web cam feed.
    :return:It displays the image obtained from camera with bounding and text if faces are detected.
    """
    # Initializes a variable for Video Capture
    cap = cv2.VideoCapture(0)
    frame_array = []
    while True:
        ret, frame = cap.read()
        (frame_height, frame_width, channels) = frame.shape
        if ret:
            # frame = sharpen(image)
            img = resize_image(frame, (224, 224))
            # Get the image dimensions
            (img_height, img_width, img_channels) = img.shape
            faces = detector(img, 1)
            if len(faces) != 0:
                for face, d in enumerate(faces):

                    # Get the locations of facial features like eyes, nose for using it to create encodings
                    shape = sp(img, d)


                    # Get the coordinates of bounding box enclosing the face.
                    left = d.left()
                    top = d.top()
                    right = d.right()
                    bottom = d.bottom()

                    # Converting the coordinates to match 480x640 resolution since we are resizing img to 224x224
                    cal_left = int(left * frame_width / img_width)
                    cal_top = int(top * frame_height / img_height)
                    cal_right = int(right * frame_width / img_width)
                    cal_bottom = int(bottom * frame_height / img_height)
                    cv2.rectangle(frame, (cal_left, cal_top), (cal_right, cal_bottom), (0, 255, 0), 2)

                    # Calculate encodings of the face detected
                    #start_encode = time.time()
                    face_descriptor = list(face_recognition_model.compute_face_descriptor(img, shape))
                    #print("Time taken to encode each face = "+str((time.time()-start_encode) * 1000)+" ms")

                    face_encoding = pd.DataFrame([face_descriptor])
                    face_encoding_list = [np.array(face_descriptor)]

                    # Get indices the N Neighbours of the facial encoding
                    list_neighbors = knn.kneighbors(face_encoding, return_distance=False)

                    # Calculate the L2 distance between the encodings of N neighbours and the detected face.
                    database_list = L2_distance(face_encoding_list, list_neighbors[0])
                    person_name = database_list[0][0]

                    # Draw the bounding box and write name on top of the face.
                    frame = draw_rectangle(frame, (cal_left, cal_top), (cal_right, cal_bottom), frame_rect_color, False)
                    frame = draw_rectangle(frame, (cal_left, cal_top - 30), (cal_right, cal_top), frame_rect_color, True)
                    frame = write_text(frame, person_name, (cal_left + 6, cal_top - 6), frame_text_color)

            frame_array.append(frame)
            cv2.imshow('Frame: Press q to quit the frame', frame)
            # Break the loop when 'q' is pressed
            if cv2.waitKey(25) & 0xff == ord('q'):
                video_write(frame_arraywq
                            )
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    face_recognizer()
