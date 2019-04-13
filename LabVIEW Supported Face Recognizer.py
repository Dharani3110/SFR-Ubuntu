"""
LabVIEW Supported Face Recognizer.py  is to perform Face detection and recognition on the given input image using
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
from imutils import paths
import requests
import configparser
from flask import Flask,request, jsonify


#========================================= Initial setup ===============================================#

# Load the Knn model
knn = joblib.load('Dependencies\KNN_classifier_model.sav')

# Load the encodings created for train images
data = pickle.loads(open('Dependencies\Encodings', 'rb').read())
print("\nNumber of encodings loaded:  ", len(data))

# Store face detector model, shape predictor model and face recognition model in individual variables.
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('Dependencies\shape_predictor_5_face_landmarks.dat')
face_recognition_model = dlib.face_recognition_model_v1('Dependencies\dlib_face_recognition_resnet_model_v1.dat')

# Tweak the min_distance parameter to get minimised false recognition
config = configparser.ConfigParser()
config.read('dependencies\config.ini')
min_distance = config.getfloat( "Threshold_initialization" , 'min_distance')
print( "Initialized threshold: " , min_distance)


#====================================== Function definitions ==========================================#

def read_image(filename):
    """
    This function reads the image specified by its filename
    :param filename: It is string containing the filename along with the extension of image
    :return: It returns image stored in the specified filename
    """
    image = cv2.imread(filename, 1)
    return image


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


def L2_distance(face_encoding, index_list):
    """
    This function calculates the L2 distance between each of the encodings of N Neighbours with the encodings of the
    face detected.
    :param face_encoding: It gets a 128-dimensional list of facial encoding of face detected in the test_image.
    :param index_list: It contains the index values of n nearest neighbours returned by the knn classifier.
    :return: A string indicating the name of the person recognised based on the threshold(min_distance) we set.
    """
    database_list = [tuple([data[index][0], np.linalg.norm( face_encoding - data[index][1])]) for index in index_list]
    database_list.sort(key = lambda x: x[1])
    print('\n')
    if database_list[0][1] > min_distance:
        duplicate = list(database_list[0])
        # If no distance is less than the min distance, name is considered as 'Unknown".
        duplicate[0] = 'Unknown'
        database_list.insert(0, tuple(duplicate))
    return database_list


def face_recognizer(image):
    """
    This function detects and recognises faces in the given image.
    :return: It returns a json with names and the locations of the persons detected.
    """
    frame =  sharpen(image)
    faces = detector(frame, 1)
    json_data = {}
    json_data['Faces_detected'] = str(len(faces))
    if len(faces) != 0:
        for face, d in enumerate(faces):

            # Get the locations of facial features like eyes, nose for using it to create encodings
            shape = sp(frame, d)

            # Get the coordinates of bounding box enclosing the face.
            left = d.left()
            top = d.top()
            right = d.right()
            bottom = d.bottom()

            # Calculate encodings of the face detected
            #start_encode = time.time()
            face_descriptor = list( face_recognition_model.compute_face_descriptor( frame, shape))
            #print("Time taken to encode face "+ str(face+1) + " :::  " +str((time.time()-start_encode) * 1000)+"  ms")

            face_encoding = pd.DataFrame([face_descriptor])
            face_encoding_list = [np.array(face_descriptor)]

            # Get indices the N Neighbours of the facial encoding
            list_neighbors = knn.kneighbors( face_encoding, return_distance=False)

            # Calculate the L2 distance between the encodings of N neighbours and the detected face.
            database_list = L2_distance( face_encoding_list, list_neighbors[0])
            person_name = database_list[0][0]

            #Write as json
            key = 'face '+str(face + 1)
            json_data[key] =  (
                                  {'name':person_name,
                                   'locations': { 'left'   : left,
                                                  'right'  : right,
                                                  'top'    : top,
                                                  'bottom' : bottom
                                                }
                                  }
                              )

    return json_data


def ReadAndRecognize():
    """
    This function reads and calls recognizer function.
    :return: Returns data with detected faces and their locations in the form of json.
    """
    image_paths = list(paths.list_images("Test image"))
    if len(image_paths)== 0:
        json_data = {}
        json_data['Warning'] = 'No image file exists!'
        return json_data
    else:
        image = read_image(image_paths[0])
        json_data = face_recognizer(image)
        return json_data


#=========================================== HTTP request =============================================#


app = Flask(__name__)

@app.route('/predict', methods = ['GET'])
def send_json():
    """
    This function is to return the json data to client with method GET,when connected.
    :return: Returns the json data to client
    """
    json_data = ReadAndRecognize()
    print(json_data)
    return jsonify(json_data)


if __name__ == '__main__':
    app.run()
