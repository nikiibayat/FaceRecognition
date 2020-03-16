import matplotlib.pyplot as plt
from builtins import len
import numpy as np
import joblib
import argparse
import sys
sys.path.insert(1, '/tmp/pycharm_project_243/face_recognition')
import preprocessing
from PIL import Image
import os
import pickle



face_recogniser = joblib.load('training/model/face_recogniser.pkl')
preprocess = preprocessing.ExifOrientationNormalize()

def predict(img):
    img = preprocess(img)
    # convert image to RGB (stripping alpha channel if exists)
    img = img.convert('RGB')
    faces = face_recogniser(img)
    return \
        {
            'faces': [
                {
                    'top_prediction': face.top_prediction._asdict()
                }
                for face in faces
            ]
        }

def create_hist_dict():
    rootdir = '/home/nbayat5/Desktop/celebA/face_recognition_HR_train'
    hist_data = {}
    old_file_label = None
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            file_label = file.split('.')[0].split('_')[0]
            print(file_label)
            if old_file_label is not None:
                if old_file_label == file_label:
                    count += 1
                else:
                    if count in hist_data.keys():
                        hist_data[count].append(file_label)
                    else:
                        hist_data[count] = [file_label]
                    print("Identity {} has {} training sample.".format(file_label,count))
                    count = 0
                    old_file_label = None
            else:
                old_file_label = file_label
                count = 1

    for key in hist_data.keys():
        print("{} identities have {} training samples.".format(len(hist_data[key]), key))
    pickle.dump(hist_data, open("hist_data.pkl", "wb"))


def create_detected_labels():
    rootdir = '/home/nbayat5/Desktop/celebA/face_recognition_HR_test'
    detected_labels = []
    score = 0
    length = 0
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            file_label = file.split('.')[0].split('_')[0]
            print(file_label)

            path = os.path.join(rootdir, file)
            img1 = Image.open(path)  # HR
            top_pred1 = predict(img1)
            if len(top_pred1['faces']) > 0:
                label1 = top_pred1['faces'][0]["top_prediction"]['label']
                if label1 == file_label:
                    score += 1
                    print("Label {} added to detected labels.".format(label1))
                    detected_labels.append(label1)
                length += 1
    print("Accuracy", score/length*100)
    pickle.dump(detected_labels, open("detected_labels.pkl", "wb"))


def plot_histogram():
    hist_data = pickle.load(open("hist_data.pkl","rb"))
    detected_labels = pickle.load(open("detected_labels.pkl","rb"))
    x = []
    y = []
    y_prime = []
    for key, value in sorted(hist_data.items()):
        x.append(key)
        y.append(len(value))
        count = 0
        for label in value:
            if label in detected_labels:
                count += 1
        print("Percentage of detection for key {} is {}".format(key, count/len(value)))
        y_prime.append(count)
    plt.figure(figsize=[10, 8])
    p1 = plt.bar(x, y_prime, color='#0504aa')
    p2 = plt.bar(x, y, bottom=y_prime, color='#00ff00')
    plt.xlim(min(x)-1, max(x)+1)
    plt.xlabel('# training samples', fontsize=15)
    plt.ylabel('# identites', fontsize=15)
    plt.title('# Train Sample per Identity Histogram', fontsize=15)
    plt.legend((p1[0], p2[0]), ('Detected', 'Undetected'))
    plt.savefig("histogram.png")





def main():
    # create_hist_dict()
    # create_detected_labels()
    plot_histogram()

if __name__ == '__main__':
    main()
