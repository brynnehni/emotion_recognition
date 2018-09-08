########## this sorts the images by emotion ############
#### result: sorted_set directory

import glob
from shutil import copyfile
emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"] #Define emotion order
participants = glob.glob("source_emotion//*") #Returns a list of all folders with participant numbers
for x in participants:
    part = "%s" %x[-4:] #store current participant number
    for sessions in glob.glob("%s//*" %x): #Store list of sessions for current participant
        for files in glob.glob("%s//*" %sessions):
            current_session = files[20:-30]
            file = open(files, 'r')
            emotion = int(float(file.readline())) #emotions are encoded as a float, readline as float, then convert to integer.
            sourcefile_emotion = glob.glob("source_images//%s//%s//*" %(part, current_session))[-1] #get path for last image in sequence, which contains the emotion
            sourcefile_neutral = glob.glob("source_images//%s//%s//*" %(part, current_session))[0] #do same for neutral image
            dest_neut = "sorted_set//neutral//%s" %sourcefile_neutral[25:] #Generate path to put neutral image
            dest_emot = "sorted_set//%s//%s" %(emotions[emotion], sourcefile_emotion[25:]) #Do same for emotion containing image
            copyfile(sourcefile_neutral, dest_neut) #Copy file
            copyfile(sourcefile_emotion, dest_emot) #Copy file


#### step 2
import cv2
import glob

faceDet = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
faceDet_two = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
faceDet_three = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
faceDet_four = cv2.CascadeClassifier("haarcascade_frontalface_alt_tree.xml")
emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"] #Define emotions
def detect_faces(emotion):
    files = glob.glob("sorted_set//%s//*" %emotion) #Get list of all images with emotion
    filenumber = 0
    for f in files:
        frame = cv2.imread(f) #Open image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Convert image to grayscale
        #Detect face using 4 different classifiers
        face = faceDet.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        face_two = faceDet_two.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        face_three = faceDet_three.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        face_four = faceDet_four.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        #Go over detected faces, stop at first detected face, return empty if no face.
        if len(face) == 1:
            facefeatures = face
        elif len(face_two) == 1:
            facefeatures = face_two
        elif len(face_three) == 1:
            facefeatures = face_three
        elif len(face_four) == 1:
            facefeatures = face_four
        else:
            facefeatures = ""
        #Cut and save face
        for (x, y, w, h) in facefeatures: #get coordinates and size of rectangle containing face
            print("face found in file: %s" %f)
            gray = gray[y:y+h, x:x+w] #Cut the frame to size
            try:
                out = cv2.resize(gray, (350, 350)) #Resize face so all images have same size
                cv2.imwrite("dataset//%s//%s.jpg" %(emotion, filenumber), out) #Write image
            except:
               pass #If error, pass file
        filenumber += 1 #Increment image number
for emotion in emotions:
    detect_faces(emotion) #Call functiona

#### step 3: train the classifier
import cv2
import glob
import random
import numpy as np
emotions = ["anger", "neutral", "happy"] #Emotion list
### may need to run `pip install opencv-contrib-python`
fishface = cv2.face.FisherFaceRecognizer_create() #Initialize fisher face classifier
data = {}
def get_files(emotion): #Define function to get file list, randomly shuffle it and split 80/20
    files = glob.glob("dataset2//%s//*" %emotion)
    random.shuffle(files)
    training = files #get first 100% of file list
    return training
def make_sets():
    training_data = []
    training_labels = []
    for emotion in emotions:
        training = get_files(emotion)
        #Append data to training and prediction list, and generate labels 0-7
        for item in training:
            image = cv2.imread(item) #open image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayscale
            training_data.append(gray) #append image array to training data list
            training_labels.append(emotions.index(emotion))
    return training_data, training_labels
def run_recognizer(image_path):
    training_data, training_labels = make_sets()
    print("training fisher face classifier")
    print("size of training set is:", len(training_labels), "images")
    fishface.train(training_data, np.asarray(training_labels))
    print("predicting classification set")
    #gray scale prediction image
    prediction = []
    confidence = []
    photo = []
    pred_image = glob.glob(image_path)
    for item in pred_image:
        image = cv2.imread(item)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (350, 350))
        pred, conf = fishface.predict(gray)
        prediction.append(emotions[pred])
        confidence.append(conf)
        photo.append(item)
    return prediction, confidence, photo
#Now run it
labels, confidence, photo_file = run_recognizer(image_path = "pred_images//*")
for idx, val in enumerate(photo_file):
    print("PHOTO FILE: ", val)
    print("PREDICTED EMOTION: ", labels[idx])
    print("CONFIDENCE: ", confidence[idx])
