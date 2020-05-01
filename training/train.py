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
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset

sys.path.insert(1, '../face_recognition')
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
        transforms.Resize((64, 64))
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
        pickle.dump(embeddings, open(
            '/home/nbayat5/Desktop/face-recognition-master/training/model/celebA/{}_{}_embeddings.pkl'.format(SR_method,
                                                                                                              type),
            'wb'))
        pickle.dump(labels, open(
            '/home/nbayat5/Desktop/face-recognition-master/training/model/celebA/{}_{}_labels.pkl'.format(SR_method,
                                                                                                          type),
            'wb'))
        pickle.dump(dataset.class_to_idx, open(
            '/home/nbayat5/Desktop/face-recognition-master/training/model/celebA/{}_{}_class_to_index.pkl'.format(
                SR_method, type), 'wb'))
    except:
        print("dataset to embedding failed")
    return embeddings, labels, dataset.class_to_idx


def bounding_box_comparison(features_extractor):
    transform = transforms.Compose([
        preprocessing.ExifOrientationNormalize(),
        transforms.Resize((64, 64))
    ])
    not_detected_HR = []
    not_detected_srgan = []
    not_detected_srcnn = []
    hr_srgan_mse_total = []
    hr_srcnn_mse_total = []

    HR_dataset = datasets.ImageFolder("/home/nbayat5/Desktop/celebA/face_recognition_without_bbx/HR")
    SRGAN_dataset = datasets.ImageFolder("/home/nbayat5/Desktop/celebA/face_recognition_without_bbx/SRGAN")
    SRCNN_dataset = datasets.ImageFolder("/home/nbayat5/Desktop/celebA/face_recognition_without_bbx/SRCNN")

    for idx, (hr_path, label) in enumerate(HR_dataset.samples):
        parts= hr_path.split('/')
        parts[6] = 'SRGAN'
        srgan_path = '/'.join(parts)
        parts[6] = 'SRCNN'
        srcnn_path = '/'.join(parts)

        hr_bbs, _ = features_extractor(transform(Image.open(hr_path).convert('RGB')))
        srgan_bbs, _ = features_extractor(transform(Image.open(srgan_path).convert('RGB')))
        srcnn_bbs, _ = features_extractor(transform(Image.open(srcnn_path).convert('RGB')))

        if hr_bbs is None:
            # print("Could not find face on {} HR".format(parts[8]))
            not_detected_HR.append(parts[8])
        if srgan_bbs is None:
            # print("Could not find face on {} SRGAN".format(parts[8]))
            not_detected_srgan.append(parts[8])
        if srcnn_bbs is None:
            # print("Could not find face on {} SRCNN".format(parts[8]))
            not_detected_srcnn.append(parts[8])
        if hr_bbs is not None and srgan_bbs is not None and srcnn_bbs is not None:
            if hr_bbs.shape[0] > 1:
                # print("Multiple faces detected for {} HR, taking one with highest probability".format(parts[8]))
                hr_bbs = hr_bbs[0, :]
            if srgan_bbs.shape[0] > 1:
                # print("Multiple faces detected for {} SRGAN, taking one with highest probability".format(parts[8]))
                srgan_bbs = srgan_bbs[0, :]
            if srcnn_bbs.shape[0] > 1:
                # print("Multiple faces detected for {} SRCNN, taking one with highest probability".format(parts[8]))
                srcnn_bbs = srcnn_bbs[0, :]
            # find MSE loss between bounding boxes
            try:
                hr_srgan_mse = (np.square(hr_bbs - srgan_bbs)).mean(axis=1)[0]
                print(idx, ": MSE distance between HR and SRGAN bbx: ", hr_srgan_mse)
                hr_srgan_mse_total.append(hr_srgan_mse)
                hr_srcnn_mse = (np.square(hr_bbs - srcnn_bbs)).mean(axis=1)[0]
                print(idx, ": MSE distance between HR and SRCNN bbx: ", hr_srcnn_mse)
                hr_srcnn_mse_total.append(hr_srcnn_mse)
            except Exception as e:
                print("Exception {} happend!".format(e))
                print("HR: ",hr_bbs)
                print("SRGAN: ",srgan_bbs)
                print("SRCNN: ",srcnn_bbs)
        else:
            continue

    print("-----------------------------------------------------------")
    print("Number of not detected HR samples: ", len(not_detected_HR))
    print("Number of not detected SRGAN samples: ", len(not_detected_srgan))
    print("Number of not detected SRCNN samples: ", len(not_detected_srcnn))
    print("{} not detected identities are shared between srgan and srcnn".format(
        len([value for value in not_detected_srgan if value in not_detected_srcnn])))
    print("******FINAL RESULT********")
    print("Average MSE distance between HR and SRGAN bbx: ", np.mean(hr_srgan_mse_total))
    print("Average MSE distance between HR and SRCNN bbx: ", np.mean(hr_srcnn_mse_total))


def train_softmax(args, embeddings, labels):
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


def load_embedding_label(SR_method="HR", type="train"):
    embeddings = pickle.load(open(
        '/home/nbayat5/Desktop/face-recognition-master/training/model/celebA/{}_{}_embeddings.pkl'.format(SR_method,
                                                                                                          type),
        'rb'))
    labels = pickle.load(open(
        '/home/nbayat5/Desktop/face-recognition-master/training/model/celebA/{}_{}_labels.pkl'.format(SR_method, type),
        'rb'))
    class_to_idx = pickle.load(open(
        '/home/nbayat5/Desktop/face-recognition-master/training/model/celebA/{}_{}_class_to_index.pkl'.format(SR_method,
                                                                                                              type),
        'rb'))
    return embeddings, labels, class_to_idx


class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(512, 1024)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(1024, 2048)
        self.fc3 = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = F.relu(self.dropout(self.fc1(x)))
        x = F.relu(self.fc2(x))
        # x = F.relu(self.dropout(self.fc2(x)))
        x = F.log_softmax(self.fc3(x))
        return x


def train(model, device, train_loader, optimizer, epoch, train_data, test_data):
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
        if batch_idx % 1000 == 0:
            print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset), loss.item()))

    print('Train set: Accuracy: {}/{} ({:.0f}%)'.format(correct, len(train_loader.dataset),
                                                        100. * correct / len(train_loader.dataset)))
    return loss.item(), correct


def test(model, device, test_loader, train_data, test_data):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():  # temporarily set all the requires_grad flag to false
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += compare_prediction(train_data, test_data, pred, target)

    test_loss /= len(test_loader.dataset)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
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
    # test_loss = []
    # train_loss = []
    # test_acc = []
    # train_acc = []
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
    model = Net(unique_label).to(device)
    # bounding box
    # , weight_decay = 1e-6   # l2 penalty
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # MTCNN - Adam => lr=1e-3 - 10 epochs - step size =7 gamma = 0.7
    scheduler = StepLR(optimizer, step_size=7, gamma=0.7)
    for epoch in range(1, 60):
        print("Epoch: ", epoch)
        loss1, correct1 = train(model, device, train_loader, optimizer, epoch, train_data, test_data)
        # scheduler.step()
        loss2, correct2 = test(model, device, test_loader, train_data, test_data)
        # train_loss.append(loss1)
        # train_acc.append(correct1)
        # test_loss.append(loss2)
        # test_acc.append(correct2)
        if epoch % 10 == 0:
            torch.save(model.state_dict(),
                       "/home/nbayat5/Desktop/face-recognition-master/training/model/bbx_HR_NN_model.pt")
    print('Finished Training')
    # plot_figures(train_loss, train_acc, test_loss, test_acc, type)
    return model


def plot_figures(train_loss, train_acc, test_loss, test_acc, type):
    epoch = np.arange(len(train_loss))
    plt.plot(epoch, train_loss, label='train loss', c='blue')
    plt.plot(epoch, test_loss, label='test loss', c='red')
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.title("{} train vs {} test loss based on epoch".format(type, type))
    plt.legend()
    plt.savefig("/home/nbayat5/face-recognition-master/training/{}_loss.png".format(type))
    plt.close()
    plt.plot(epoch, train_acc, label='# train', c='blue')
    plt.plot(epoch, test_acc, label='# test', c='red')
    plt.ylabel("# correct predictions")
    plt.xlabel("Epoch")
    plt.title("{} train vs {} test: # of correct prediction".format(type, type))
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


def remove_identities(train_embeddings, train_labels, test_labels, test_embeddings, train_class_to_idx,
                      test_class_to_idx):
    train_idx_to_class = {v: k for k, v in train_class_to_idx.items()}
    test_idx_to_class = {v: k for k, v in test_class_to_idx.items()}

    (unique, counts) = np.unique(train_labels, return_counts=True)
    frequencies = np.asarray((unique, counts)).T
    discarded = []
    for item in frequencies:
        if item[1] < 3:
            discarded.append(item[0])

    indexes = []
    for idx in range(len(train_labels)):
        if train_labels[idx] in discarded:
            indexes.append(idx)

    train_labels = np.delete(train_labels, indexes)
    train_embeddings = np.delete(train_embeddings, indexes, axis=0)

    indexes = []
    for idx in range(len(test_labels)):
        if test_labels[idx] in discarded:
            indexes.append(idx)

    test_labels = np.delete(test_labels, indexes)
    test_embeddings = np.delete(test_embeddings, indexes, axis=0)
    for key in indexes:
        if key in test_idx_to_class:
            del test_idx_to_class[key]
            del train_idx_to_class[key]

    print(len(test_embeddings), len(test_labels), len(test_idx_to_class.items()))
    print("{} unique identities were discarded due to lack of sufficient training examples (<3)".format(len(discarded)))
    print(len(train_idx_to_class.items()))
    print(len(test_idx_to_class.items()))

    return train_embeddings, train_labels, test_labels, test_embeddings, train_idx_to_class, test_idx_to_class


def neural_network_classifier(device, sr_method="HR"):
    ### load embedding and labels
    train_embeddings, train_labels, train_class_to_idx = load_embedding_label(SR_method=sr_method, type="train")
    test_embeddings, test_labels, test_class_to_idx = load_embedding_label(SR_method=sr_method, type="test")
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


    model = train_NN("HR", len(train_class_to_idx.keys()), train_data, test_data, device)



def logistic_regression_classifier(args, sr_method="HR"):
    train_embeddings, train_labels, train_class_to_idx = load_embedding_label(SR_method=sr_method, type="train")
    test_embeddings, test_labels, test_class_to_idx = load_embedding_label(SR_method=sr_method, type="test")
    # train_idx_to_class = {v: k for k, v in train_class_to_idx.items()}
    # test_idx_to_class = {v: k for k, v in test_class_to_idx.items()}
    # train_target_names = map(lambda i: i[1], sorted(train_idx_to_class.items(), key=lambda i: i[0]))
    # test_target_names = map(lambda i: i[1], sorted(test_idx_to_class.items(), key=lambda i: i[0]))

    print("Started training")
    clf = train_softmax(args, train_embeddings, train_labels)
    train_score = clf.score(train_embeddings, train_labels)
    test_score = clf.score(test_embeddings, test_labels)
    print("Train Accuracy: ", train_score * 100)
    print("HR Test Accuracy: ", test_score * 100)

    # if not os.path.isdir(MODEL_DIR_PATH):
    #  os.mkdir(MODEL_DIR_PATH)
    # model_path = os.path.join('./model','{}_face_recogniser.pkl'.format(SR_method))
    # FaceRecogniser = face_recogniser.FaceRecogniser(features_extractor, clf, train_idx_to_class)
    # joblib.dump(FaceRecogniser, model_path)


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    features_extractor = face_features_extractor.FaceFeaturesExtractor()

    # Comparing bbs detected for SRGAN vs SRCNN
    # bounding_box_comparison(features_extractor)

    ### extract embeddings and labels for an ImageFolder format dataset
    #load_data(args, features_extractor, SR_method="srcnn", type="train")

    # Neural Network Classifier
    #neural_network_classifier(device, sr_method="HR")

    # logistic regression classifier
    logistic_regression_classifier(args, sr_method="HR")


if __name__ == '__main__':
    main()
