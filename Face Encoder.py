"""
Face Encoder.py  is to create encodings for all the train_images using dlib library.

"""

import os
import dlib
import glob
import cv2
import numpy as np
import pickle
import csv
from imutils import paths
import pandas as pd
import sklearn.neighbors.typedefs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib

predictor_path = 'Dependencies/shape_predictor_5_face_landmarks.dat'
face_rec_model_path = 'Dependencies/dlib_face_recognition_resnet_model_v1.dat'
train_images_path = "Train images"

# Load all the models - a detector to find the faces, a shape predictor
#to find face landmarks so we can precisely localize the face, and finally the face recognition model.
detector = dlib.get_frontal_face_detector()
shape = dlib.shape_predictor(predictor_path)
face_recognition_model = dlib.face_recognition_model_v1(face_rec_model_path)
data = []

def list_to_csv(data):
    """
    This function converts encodings to CSV file.
    :param data: It contains list of names and encodings
    :return: Returns Encodings_csv.csv file with names and encodings in it.
    """
    # sorting the encoding according to alphabetical order.
    data.sort(key=lambda x: x[0])
    names_list = []
    with open('Dependencies/Encodings_csv.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, dialect='excel')
        for encoding in data:
            # converting the 128-d encoding to string
            encoding_list = [str(value) for value in encoding[1]]
            # converting list of string to list of float value
            encoding_list_value = [float(value) for value in encoding_list]
            # converting the name to a list.
            names_list.append(encoding[0])
            row = names_list+encoding_list_value
            writer.writerow(row)
            names_list.pop()

def store_KNN_model():
    """
    It is to store the KNN model as .sav file.
    :return: Returns knn_classifier_model.sav file with model loaded in it.
    """
    # read the csv file containing the encodings
    df = pd.read_csv('Dependencies/Encodings_csv.csv', header=None)
    # separate the encodings from the csv file
    encodings = df.drop(df.columns[0], axis=1)
    # separate the class name i.e name of person from the csv file
    names = df[0]
    # specify number of neighbours for the model
    knn = KNeighborsClassifier(n_neighbors=5)
    # Train the model
    knn.fit(encodings, names)
    filename = 'Dependencies/KNN_classifier_model.sav'
    # Store the model for later use
    joblib.dump(knn, filename)
    print("\nKNN Model trained and stored....\n")

def read_image(filename):
    """
    This function reads the image specified by its filename
    :param filename: It is string containing the filename along with the extension of image
    :return: It returns image stored in the specified filename
    """
    image = cv2.imread(filename)
    return image


def create_encodings():
    """
     It is to create encodings for all the images in the dataset.
    :return: It returns Encodings.dat, a binary data file.
    """
    print("Preparing database............")
    image_paths = list(paths.list_images(train_images_path))
    for (i, image)  in enumerate(image_paths):
        print("Processing image: {0}/{1}".format(i+1, len(image_paths)))
        img = read_image(image)
        #split the path of the image
        (img_folder, ext) = os.path.splitext(image)
        (main_fldr , fldr_name, img_name )= img_folder.split('/')

        # Ask the detector to find the bounding boxes of each face. The 1 in the
        # second argument indicates that we should up-sample the image 1 time. This
        # will make everything bigger and allow us to detect more faces.
        detections = detector(img, 1)
        if len(detections) != 0:
            # Now process each face found.
            for face in detections:
                # Get the landmarks/parts for the face in box d.
                sp = shape(img, face)
                # Compute the 128D vector that describes the face in img identified by
                face_descriptor = list(face_recognition_model.compute_face_descriptor(img, sp))
                # Append the encoding along with its corresponding image name to a list
                data.append([fldr_name, np.array(face_descriptor)])

    # convert the encodings into csv file
    list_to_csv(data)
    print("\nCompleted encoding...")
    with open('Dependencies/Encodings', 'wb') as fp:
        fp.write(pickle.dumps(data))
    store_KNN_model()


if __name__ == '__main__':
    create_encodings()
