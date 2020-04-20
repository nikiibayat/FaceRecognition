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
from torchsummary import summary
import pickle
import sys
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset

sys.path.insert(1, '/tmp/pycharm_project_243/face_recognition')
import preprocessing
import face_features_extractor
import face_recogniser
from torch.optim.lr_scheduler import StepLR

MODEL_DIR_PATH = 'model'



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
        transforms.Resize(1024),
        ToTensor()
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

    print("# labels: ", len(labels))
    print("# embeddings: ", len(embeddings))
    return np.stack(embeddings), labels


def load_data(args, features_extractor, SR_method="HR", type="train"):
    if args.embeddings_path:
        return np.loadtxt(args.embeddings_path), \
               np.loadtxt(args.labels_path, dtype='str').tolist(), \
               joblib.load(args.class_to_idx_path)

    dataset = datasets.ImageFolder(args.dataset_path)
    try:
        embeddings, labels = dataset_to_embeddings(dataset, features_extractor)
        pickle.dump(embeddings, open('/home/nbayat5/Desktop/face-recognition-master/training/model/bbx/{}_{}_embeddings.pkl'.format(SR_method,type), 'wb'))
        pickle.dump(labels, open('/home/nbayat5/Desktop/face-recognition-master/training/model/bbx/{}_{}_labels.pkl'.format(SR_method,type), 'wb'))
        pickle.dump(dataset.class_to_idx, open('/home/nbayat5/Desktop/face-recognition-master/training/model/bbx/{}_{}_class_to_index.pkl'.format(SR_method,type), 'wb'))
    except:
        print("dataset to embedding failed")
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

def load_embedding_label(SR_method="HR", type = "train"):
    embeddings = pickle.load(open('/home/nbayat5/Desktop/face-recognition-master/training/model/bbx/{}_{}_embeddings.pkl'.format(SR_method,type), 'rb'))
    labels = pickle.load(open('/home/nbayat5/Desktop/face-recognition-master/training/model/bbx/{}_{}_labels.pkl'.format(SR_method,type), 'rb'))
    class_to_idx = pickle.load(open('/home/nbayat5/Desktop/face-recognition-master/training/model/bbx/{}_{}_class_to_index.pkl'.format(SR_method,type), 'rb'))
    return embeddings, labels, class_to_idx


class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, 2048)
        self.fc3 = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x))
        return x


def train(model, device, train_loader, optimizer, epoch,train_data, test_data):
    model.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        pred = output.argmax(dim=1, keepdim=True)
        correct += compare_prediction(train_data, test_data, pred, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset), loss.item()))

    print('Train set: Accuracy: {}/{} ({:.0f}%)'.format(correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
    return loss.item(), correct

def test(model, device, test_loader, train_data, test_data):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad(): # temporarily set all the requires_grad flag to false
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += compare_prediction(train_data, test_data, pred, target)

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss, correct


def compare_prediction(train_data, test_data, pred, target):
    count = 0
    for idx in range(list(pred.size())[0]):
        target1 = test_data.index_to_class(target.data[idx].item())
        target2 = train_data.index_to_class(pred.data[idx][0].item())
        if target1 == target2:
            count += 1
    # count = pred.eq(target.view_as(pred)).sum().item()
    return count


def train_NN(type, unique_label, train_data, test_data, device, batch_size=32):
    test_loss = []
    train_loss = []
    test_acc = []
    train_acc = []
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
    model = Net(unique_label).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = StepLR(optimizer, step_size=7, gamma=0.7)
    for epoch in range(1, 20):
        print("Epoch: ", epoch)
        loss1, correct1 = train(model, device, train_loader, optimizer, epoch, train_data, test_data)
        scheduler.step()
        loss2, correct2 = test(model, device, test_loader, train_data, test_data)
        train_loss.append(loss1)
        train_acc.append(correct1)
        test_loss.append(loss2)
        test_acc.append(correct2)
        if epoch % 10 == 0:
            torch.save(model.state_dict(), "/home/nbayat5/Desktop/face-recognition-master/training/model/bbx_HR_NN_model.pt")
    print('Finished Training')
    #plot_figures(train_loss, train_acc, test_loss, test_acc, type)
    return model

def plot_figures(train_loss, train_acc, test_loss, test_acc, type):
    epoch = np.arange(len(train_loss))
    plt.plot(epoch, train_loss, label='train loss', c='blue')
    plt.plot(epoch, test_loss, label='test loss', c='red')
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.title("{} train vs {} test loss based on epoch".format(type,type))
    plt.legend()
    plt.savefig("/home/nbayat5/face-recognition-master/training/{}_loss.png".format(type))
    plt.close()
    plt.plot(epoch, train_acc, label='# train', c='blue')
    plt.plot(epoch, test_acc, label='# test', c='red')
    plt.ylabel("# correct predictions")
    plt.xlabel("Epoch")
    plt.title("{} train vs {} test: # of correct prediction".format(type,type))
    plt.legend()
    plt.savefig("/home/nbayat5/face-recognition-master/training/HR_accuracy.png".format(type))
    plt.close()


class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels, class_to_idx):
        self.samples = []
        batch_size = 32
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}
        embeddings = np.array_split(embeddings, np.ceil(len(embeddings) / batch_size))
        labels = np.array_split(labels, np.ceil(len(labels) / batch_size))
        for i in range(len(embeddings)):
            for idx in range(len(embeddings[i])):
                self.samples.append((embeddings[i][idx], np.int(labels[i][idx])))

        embeddings = None
        labels = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def index_to_class(self, idx):
         return self.idx_to_class[idx]

def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    features_extractor = face_features_extractor.FaceFeaturesExtractor()

    # extract embeddings and labels
    #load_data(args, features_extractor, SR_method="SRGAN", type="test")

    # """
    # load embedding and labels
    train_embeddings, train_labels, train_class_to_idx = load_embedding_label(SR_method="HR", type="train")
    test_embeddings, test_labels, test_class_to_idx = load_embedding_label(SR_method="HR", type="test")

    print("# num of labels in directory: ", len(train_class_to_idx.keys()))

    unique = []
    for value in train_labels:
        if value not in unique:
            unique.append(value)
    print("# unique labels detected by MTCNN",len(unique))

    not_detected = []
    for value in train_class_to_idx.values():
        if value not in unique:
            not_detected.append(value)
    print("Train identities that are not detected: ", not_detected)

    not_detected = []
    for value in test_class_to_idx.values():
        if value not in test_labels:
            not_detected.append(value)
    print("# Test identities that are not detected: ", len(not_detected))

    train_data = EmbeddingDataset(train_embeddings, train_labels, train_class_to_idx)
    test_data = EmbeddingDataset(test_embeddings, test_labels, test_class_to_idx)


    model = train_NN("HR", 9809, train_data, test_data, device)
    # """
    """
    
    #logistic regression classifier
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    target_names = map(lambda i: i[1], sorted(idx_to_class.items(), key=lambda i: i[0]))
    clf = train(args, embeddings, labels)
    print(metrics.classification_report(labels, clf.predict(embeddings), target_names=list(target_names)))

    if not os.path.isdir(MODEL_DIR_PATH):
        os.mkdir(MODEL_DIR_PATH)
    model_path = os.path.join('/home/nbayat5/face-recognition-master/training/model', 'HR_nn_face_recogniser.pkl')
    FaceRecogniser = face_recogniser.FaceRecogniser(features_extractor, clf, idx_to_class)
    joblib.dump(FaceRecogniser, model_path)
    """


if __name__ == '__main__':
    main()
