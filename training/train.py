import os
import argparse
import joblib
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import pickle

import sys
sys.path.insert(1, '/tmp/pycharm_project_243/face_recognition')
import preprocessing
import face_features_extractor
import face_recogniser

MODEL_DIR_PATH = 'model'

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, 2048)
        self.fc3 = nn.Linear(2048, 10177)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x))
        return x



def parse_args():
    parser = argparse.ArgumentParser(
        description='Script for training Face Recognition model. You can either give path to dataset or provide path '
                    'to pre-generated embeddings, labels and class_to_idx. You can pre-generate this with '
                    'util/generate_embeddings.py script.')
    parser.add_argument('-d', '--dataset-path', help='Path to folder with images.')
    parser.add_argument('-e', '--embeddings-path', help='Path to file with embeddings.')
    parser.add_argument('-l', '--labels-path', help='Path to file with labels.')
    parser.add_argument('-c', '--class-to-idx-path', help='Path to pickled class_to_idx dict.')
    parser.add_argument('--grid-search', action='store_true',
                        help='If this option is enabled, grid search will be performed to estimate C parameter of '
                             'Logistic Regression classifier. In order to use this option you have to have at least '
                             '3 examples of every class in your dataset. It is recommended to enable this option.')
    return parser.parse_args()


def dataset_to_embeddings(dataset, features_extractor):
    transform = transforms.Compose([
        preprocessing.ExifOrientationNormalize(),
        transforms.Resize(1024)
    ])

    embeddings = []
    labels = []
    for img_path, label in dataset.samples:
        print(img_path)
        _, embedding = features_extractor(transform(Image.open(img_path).convert('RGB')))
        if embedding is None:
            print("Could not find face on {}".format(img_path))
            continue
        if embedding.shape[0] > 1:
            print("Multiple faces detected for {}, taking one with highest probability".format(img_path))
            embedding = embedding[0, :]
        embeddings.append(embedding.flatten())
        labels.append(label)

    return np.stack(embeddings), labels


def load_data(args, features_extractor):
    if args.embeddings_path:
        return np.loadtxt(args.embeddings_path), \
               np.loadtxt(args.labels_path, dtype='str').tolist(), \
               joblib.load(args.class_to_idx_path)

    dataset = datasets.ImageFolder(args.dataset_path)
    embeddings, labels = dataset_to_embeddings(dataset, features_extractor)
    return embeddings, labels, dataset.class_to_idx


def train(args, embeddings, labels):
    softmax = LogisticRegression(solver='lbfgs', multi_class='multinomial', C=10, max_iter=10000)
    if args.grid_search:
        clf = GridSearchCV(
            estimator=softmax,
            param_grid={'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]},
            cv=3
        )
    else:
        clf = softmax
    clf.fit(embeddings, labels)

    return clf.best_estimator_ if args.grid_search else clf

def train_NN(embeddings, labels, device, batch_size=32):
    net = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    embeddings = np.array_split(np.asarray(embeddings), np.ceil(len(embeddings)/batch_size))
    labels = np.array_split(np.asarray(labels), np.ceil(len(labels)/batch_size))
    for epoch in range(30):
        running_loss = 0.0
        for i in range(len(embeddings)):
            print("epoch {} batch #: {}".format(epoch+1, i+1))
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(torch.tensor(embeddings[i]).to(device))
            loss = criterion(outputs, torch.tensor(labels[i]).to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
        if epoch % 2 == 0:
            print('epoch: %d ended! loss: %.3f' % (epoch + 1, running_loss / 2))

    print('Finished Training')
    return net

def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    features_extractor = face_features_extractor.FaceFeaturesExtractor()
    # embeddings, labels, class_to_idx = load_data(args, features_extractor)

    embeddings = pickle.load(open('model/srgan_embeddings.pkl', 'rb'))
    labels = pickle.load(open('model/srgan_labels.pkl', 'rb'))
    class_to_idx = pickle.load(open('model/srgan_class_to_index.pkl', 'rb'))

    clf = train_NN(embeddings, labels, device)
    pickle.dump(clf, open('/home/nbayat5/face-recognition-master/training/model/NN_classifier.pkl', 'wb'))
    outputs = clf(torch.tensor(embeddings).to(device))
    _, predicted = torch.max(outputs.data, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    print('Accuracy of the network on the 180000 train images: %d %%' % (
            100 * correct / total))

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    # clf = train(args, embeddings, labels)
    # target_names = map(lambda i: i[1], sorted(idx_to_class.items(), key=lambda i: i[0]))
    # print(metrics.classification_report(labels, clf.predict(embeddings), target_names=list(target_names)))

    if not os.path.isdir(MODEL_DIR_PATH):
        os.mkdir(MODEL_DIR_PATH)
    model_path = os.path.join('/home/nbayat5/face-recognition-master/training/model', 'srgan_nn_face_recogniser.pkl')
    FaceRecogniser = face_recogniser.FaceRecogniser(features_extractor, clf, idx_to_class)
    joblib.dump(FaceRecogniser, model_path)


if __name__ == '__main__':
    main()
