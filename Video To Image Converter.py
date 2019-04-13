"""
It is convert the videos in the specified path to images and store them with the name of the corresponding video name.
"""
import cv2
import os
import configparser

config = configparser.ConfigParser()
config.read('Dependencies/FramesCount_config.ini')
required_frames = config.getint("Frames",'count')

# "dist_folder' is the destination path for the folder creation.
dist_folder = "Train images"
videos_path= "Train videos"
videos = os.listdir(videos_path)


def folder_check(sub_folder):
    """
    It is to check and return True or False depending on whether that folder already exists.
    :param sub_folder: It is the name of sub folder to be created under the main folder.
    :return: It returns True - when the sub folder is already present;
                        False - when the sub folder doesn't exist.
    """
    # Create the dist_folder if it doesn't exist.
    if not os.path.exists(dist_folder):
        os.mkdir(dist_folder)
    sub_folder_path = dist_folder + "/" + sub_folder

    # Creating a new folder inside dist_folder
    if not os.path.exists(sub_folder_path):
        folder_status = False
        os.mkdir(sub_folder_path)

    else:
        folder_status = True
        print (sub_folder + " already exists!!!")
    return folder_status


def acquire_frames(video_filename, folder_name):
    """
    It is to acquire the required number of frames form the video and store those frames as .jpg files
    under a folder with the name of the video.
    :param video_filename: It is the name of the video file.
    :param folder_name: It is the name of the folder to be created within which the acquired frames are stored.
    :return: It returns a folder with the obtained frames
    """
    count = 0
    frame_number = 0
    first_digit = 0
    second_digit = 0

    cap = cv2.VideoCapture(videos_path+'/'+video_filename)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while (cap.isOpened()):

        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            # Acquire required number of frames from the video file
            if (count % int(frameCount / required_frames) == 0) and (frame_number < required_frames):

                # save frame as JPG file
                cv2.imwrite(os.path.join(dist_folder + "/" + folder_name,folder_name + "{0}{1}.jpg".format(first_digit, second_digit)), frame)
                #print(first_digit,second_digit)
                frame_number += 1
                num = frame_number
                second_digit += 1
                if (num % 10 == 0):
                    first_digit += 1
                    second_digit = 0
            count += 1
        else:
            break



        # When everything done, release the capture
    cap.release()
    print("=========> Frames grabbed from ",video_filename)

if __name__ == '__main__':
    """
   
    :return: returns n frames as .jpg files
    """
    print("\nOutput sent to Train images.\n")
    for video in videos:
        new_folder_name, ext = os.path.splitext(video)
        if not folder_check(new_folder_name):
            acquire_frames(video, new_folder_name)


