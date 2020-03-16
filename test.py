from builtins import len
import numpy as np
import joblib
import argparse
import sys
sys.path.insert(1, '/tmp/pycharm_project_243/face_recognition')
import preprocessing
from face_features_extractor import FaceFeaturesExtractor
from PIL import Image
import os
import torch
import pickle

face_recogniser = joblib.load('training/model/srgan_face_recogniser.pkl')
preprocess = preprocessing.ExifOrientationNormalize()

def parse_args():
    IMAGE_KEY = 'image'
    parser = argparse.ArgumentParser(
        description='Script for testing Face Recognition model.')
    # parser.add_argument(IMAGE_KEY, type=FileStorage, location='files', required=True,
    #                 help='Image on which face recognition will be run.')
    parser.add_argument('-i', '--img_path', help='Path to image for face recognition.')

    return parser.parse_args()

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



def main():

    # img = Image.open('Hisham.jpg')
    # top_pred = predict(img)
    # if len(top_pred['faces']) > 0:
    #     label = top_pred['faces'][0]["top_prediction"]['label']
    #     print("hey hisham you look just like {}".format(label))
    rootdir = '/home/nbayat5/Desktop/celebA/face_recognition_test_srgan'
    HR_path = '/home/nbayat5/Desktop/celebA/face_recognition_HR_test'
    # save_path = '/home/nbayat5/Desktop/celebA/finetune_train_data'
    # label_embeddings = {}
    scores = 0
    length = 0
    scores1 = 0
    length1 = 0
    scores2 = 0
    length2 = 0
    loss_fn = torch.nn.MSELoss()
    different_srgan_embedding = None
    different_HR_embedding = None

    # Loss Evaluation
    HR_HR_different = []
    SR_SR_different = []
    HR_SR_same = []
    HR_SR_different = []

    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            file_label = file.split('.')[0].split('_')[0]
            # file_path = file_label + '/' + file
            path = os.path.join(rootdir, file)
            img1 = Image.open(path) #srgan
            img2 = Image.open(os.path.join(HR_path, file.split('_')[0]+'.'+file.split('.')[1])) #HR
            filename = file.split('.')[0]
            top_pred1 = predict(img1) #srgan
            top_pred2 = predict(img2) #HR
            if len(top_pred2['faces']) > 0:
                label2 = top_pred2['faces'][0]["top_prediction"]['label']
                bbs2, embeddings2 = FaceFeaturesExtractor().extract_features(img2)
                if label2 == file_label:
                    scores2 += 1
                length2 += 1
                if len(top_pred1['faces']) > 0:
                    label1 = top_pred1['faces'][0]["top_prediction"]['label']
                    if label1 == file_label:
                        scores1 += 1
                    length1 += 1
                    bbs1, embeddings1 = FaceFeaturesExtractor().extract_features(img1)
                    HR_SR_same.append(loss_fn(torch.from_numpy(embeddings2), torch.from_numpy(embeddings1)))
                    if different_srgan_embedding is not None:
                        HR_HR_different.append(loss_fn(torch.from_numpy(embeddings2), torch.from_numpy(different_HR_embedding)))
                        SR_SR_different.append(loss_fn(torch.from_numpy(embeddings1), torch.from_numpy(different_srgan_embedding)))
                        HR_SR_different.append(loss_fn(torch.from_numpy(embeddings2), torch.from_numpy(different_srgan_embedding)))


                different_srgan_embedding = embeddings1
                different_HR_embedding = embeddings2

    print("Range of loss between SRGAN Loss for different identities: {}-{} and Average: {}".format(max(SR_SR_different),min(SR_SR_different),sum(SR_SR_different)/len(SR_SR_different)))
    print("Range of loss between HR Loss for different identities: {}-{} and Average: {}".format(max(HR_HR_different),min(HR_HR_different),sum(HR_HR_different)/len(HR_HR_different)))
    print("Range of loss between SRGAN and HR Loss for same identities: {}-{} and Average: {}".format(max(HR_SR_same),min(HR_SR_same),sum(HR_SR_same)/len(HR_SR_same)))
    print("Range of loss between SRGAN and HR Loss for different identities: {}-{} and Average: {}".format(max(HR_SR_different),min(HR_SR_different),sum(HR_SR_different)/len(HR_SR_different)))
    if length != 0:
        print("average accuracy for SRGAN test is: ", np.divide(scores1, length1))
        print("average accuracy for HR test is: ", np.divide(scores2, length2))

if __name__ == '__main__':
    main()
