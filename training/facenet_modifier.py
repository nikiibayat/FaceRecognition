import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from facenet_pytorch.models.utils.detect_face import extract_face
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset
from PIL import Image
import pickle
import sys
import os.path as osp
import numpy as np
import lmdb
import matplotlib.pyplot as plt
from scipy import spatial
import multiprocessing as mp
from multiprocessing import Process
sys.path.insert(1, '../face_recognition')
import preprocessing


def load_data(dataset, transform, batch_size, batch_num):
    aligner = MTCNN(prewhiten=False, keep_all=True, thresholds=[0.6, 0.7, 0.9])
    facenet_preprocess = transforms.Compose([preprocessing.Whitening()])
    samples = []
    begin = batch_num * batch_size
    end = begin + batch_size
    if len(dataset.samples) < end:
        end = len(dataset.samples)
    for idx, (img_path, label) in enumerate(dataset.samples[begin:end]):
        # print("image {} - {}".format(idx + 1, img_path))
        img = transform(Image.open(img_path).convert('RGB'))
        print("type and shape ", type(img), img.size)
        bbs, _ = aligner.detect(img)
        if bbs is None:
            continue
        faces = torch.stack([extract_face(img, bb) for bb in bbs])
        preprocessed_faces = facenet_preprocess(faces)
        samples.append((preprocessed_faces[0], label))

    return samples


class ImageFolderLMDB(Dataset):
    def __init__(self, db_path):
        self.db_path = db_path
        self.env = lmdb.open(db_path, subdir=osp.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = pickle.loads(txn.get(b'__len__')) - 1
            self.keys = pickle.loads(txn.get(b'__keys__'))
            self.keys = self.keys[:-1]

    def __getitem__(self, index):
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        return pickle.loads(byteflow)

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'


def train(model, device, train_loader, epoch):
    f = open("log.txt", "a")
    correct = 0
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.7)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        correct += compare_prediction(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            # f.write('Train Epoch: {} [{}/{}]\tLoss: {:.6f}\n'.format(epoch, batch_idx * len(data), len(dataset),
            #                                                          loss.item()))
            print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset), loss.item()))
    # scheduler.step()
    print('Train set: Accuracy: {}/{} ({:.0f}%)'.format(correct, len(train_loader.dataset),
                                                        100. * correct / len(train_loader.dataset)))

    f.close()
    return 100. * correct / len(train_loader.dataset)


def compare_prediction(output, target):
    count = 0
    pred = output.argmax(dim=1, keepdim=True)
    for idx in range(list(pred.size())[0]):
        if pred[idx].item() == target[idx].item():
            count += 1
    return count


def test(model, device, test_loader):
    train_idx_to_class = pickle.load(open('./model/srgan_train_idx_to_class.pkl', 'rb'))
    test_idx_to_class = pickle.load(open('./model/srgan_test_idx_to_class.pkl', 'rb'))
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    with torch.no_grad():  # temporarily set all the requires_grad flag to false
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            for idx in range(list(pred.size())[0]):
                if train_idx_to_class[pred[idx].item()] == test_idx_to_class[target[idx].item()]:
                    correct += 1

    test_loss /= len(test_loader.dataset)
    print('Test set: Accuracy: {}/{} ({:.0f}%)'.format(correct, len(test_loader.dataset),
                                                       100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)


def train_NN():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = InceptionResnetV1(pretrained='vggface2', classify=True)

    model.logits = nn.Linear(in_features=512, out_features=9809, bias=True)
    model.softmax = nn.Sequential()

    for name, child in model.named_children():
        if name in ['avgpool_1a', 'last_linear', 'last_bn', 'logits']:
            print(name + ' is unfrozen')
            for param in child.parameters():
                param.requires_grad = True
        else:
            print(name + ' is frozen')
            for param in child.parameters():
                param.requires_grad = False

    model.to(device)
    batch_size = 32

    train_dataset = ImageFolderLMDB('./model/Facenet_vgg_srgan_train_removeless25_noresize.lmdb')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print("Length of train dataset: ", len(train_dataset))

    test_dataset = ImageFolderLMDB('./model/Facenet_vgg_srgan_test_removeless25_noresize.lmdb')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    print("Length of test dataset: ", len(test_dataset))
    train_acc = []
    test_acc = []
    for epoch in range(1, 20):
        print("Epoch: ", epoch)
        train_acc.append(train(model, device, train_loader, epoch))
        test_acc.append(test(model, device, test_loader))
        pickle.dump(model, open("./model/facenet_HR_train_celebA.pkl", "wb"))

    x = np.arange(1, 20)
    plt.plot(x, train_acc, marker='o', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4,
             label='Train')
    plt.plot(x, test_acc, marker='', color='red', linewidth=2, label='Test')
    plt.legend()
    plt.xlabel('Accuracy')
    plt.ylabel('Epoch')
    plt.savefig('train_test_on_HR.jpg')


def cleanse_data(gallery_embeddings, gallery_labels, probe_embeddings, probe_labels, gallery_index_to_class,
                 probe_index_to_class):
    # remove all probe embeddings that have less than 25 examples or there is no corresponding gallery image for them
    unique_probe_labels, unique_probe_counts = np.unique(probe_labels, return_counts=True)
    unique_probe_targets = []
    for idx in range(len(unique_probe_labels)):
        unique_probe_targets.append(probe_index_to_class[unique_probe_labels[idx]])
    gallery_targets = [gallery_index_to_class[label] for label in gallery_labels]
    common_targets = []
    probe_samples = []
    for idx in range(probe_embeddings.shape[0]):
        target = probe_index_to_class[probe_labels[idx]]
        index = unique_probe_targets.index(target)
        # remove all test embeddings that there is less than 25 probe images for them
        if unique_probe_counts[index] >= 1:
            if target in gallery_targets:
                probe_samples.append((probe_embeddings[idx], target))
                common_targets.append(target)
    gallery_samples = []
    for idx in range(gallery_embeddings.shape[0]):
        target = gallery_index_to_class[gallery_labels[idx]]
        if target in common_targets:
            gallery_samples.append((gallery_embeddings[idx], target))

    return probe_samples, gallery_samples


def compute_cosine(distances, gallery_embedding, probe_embedding):
    cosine = spatial.distance.cosine(gallery_embedding, probe_embedding)
    distances.append(cosine)


def cosine_evaluation(dataset="celebA"):
    root = '/home/nbayat5/Desktop/face-recognition-master/training/model/{}/'.format(dataset)
    # root = '/home/nbayat5/scratch/{}/embeddings/'.format(dataset)
    gallery_embeddings = pickle.load(
        open(root + 'HR_test_embeddings.pkl', 'rb'))
    gallery_labels = pickle.load(
        open(root + 'HR_test_labels.pkl', 'rb'))
    gallery_class_to_index = pickle.load(
        open(root + 'HR_test_class_to_index.pkl', 'rb'))

    probe_embeddings = pickle.load(
        open(root + 'SRGAN_train_embeddings.pkl', 'rb'))
    probe_labels = pickle.load(
        open(root + 'SRGAN_train_labels.pkl', 'rb'))
    probe_class_to_index = pickle.load(
        open(root + 'SRGAN_train_class_to_index.pkl', 'rb'))
    gallery_index_to_class = {v: k for k, v in gallery_class_to_index.items()}
    probe_index_to_class = {v: k for k, v in probe_class_to_index.items()}

    probe_samples, gallery_samples = cleanse_data(gallery_embeddings, gallery_labels, probe_embeddings, probe_labels,
                                                  gallery_index_to_class,
                                                  probe_index_to_class)
    print("number of srgan probe samples: ", len(probe_samples))
    print("number of HR gallery samples: ", len(gallery_samples))
    f = open("parallel_log.txt", "a")
    correct = 0
    total = len(probe_samples)
    for idx1, (probe_embedding, probe_target) in enumerate(probe_samples):
        # jobs = []
        # num_cores = 32
        # manager = mp.Manager()
        # distances = manager.list()
        num_cores = 0
        distances = []
        for idx2, (gallery_embedding, gallery_target) in enumerate(gallery_samples):
            if num_cores > 1:
                p = Process(target=compute_cosine,
                            args=(distances, gallery_embedding, probe_embedding))
                p.start()
                jobs.append(p)
                if len(jobs) == num_cores:
                    for job in jobs:
                        job.join()
                    jobs = []
                    parallel_report = "{} gallery faces for probe {}".format(num_cores, idx1) + " are explored!\n"
                    print(parallel_report)
                    f.write(parallel_report)
            else:
                compute_cosine(distances, gallery_embedding, probe_embedding)
        # if len(jobs) != 0:
        #     for job in jobs:
        #         job.join()

        index = distances.index(min(distances))
        gallery_target = gallery_samples[index][1]
        log = "probe {} with target {} has min distance with gallery {} with target {} -distance is {}\n".format(idx1,
                                                                                                               probe_target,
                                                                                                               idx2,
                                                                                                               gallery_target,
                                                                                                               min(
                                                                                                                   distances))
        print(log)
        f.write(log)
        if gallery_target == probe_target:
            correct += 1

    result = "Accuracy: "+ str(100 * np.divide(correct, total))+ "\n"
    print(result)
    f.write(result)
    f.close()


def cosine_embedding_loss(device, probe_embeddings, labels, gallery_loader, train_idx_to_class):
    criterion = nn.CosineEmbeddingLoss()
    y1 = torch.ones(1).to(device)
    y2 = -torch.ones(1).to(device)
    loss1 = 0
    loss2 = []
    flag = False
    for idx, probe_embd in enumerate(probe_embeddings):
        probe_target = train_idx_to_class[labels[idx].item()]
        for gallery_embedding, gallery_target in gallery_loader:
            if probe_target == gallery_target[0]:
                loss1 = criterion(probe_embd.reshape([1, 512]), gallery_embedding.to(device), y1)
                flag = True
            else:
                if len(loss2) >= 10 and flag == True:
                    break
                elif len(loss2) >= 10 and flag == False:
                    continue
                else:
                    tmp_loss = criterion(probe_embd.reshape([1, 512]), gallery_embedding.to(device), y2)
                    loss2.append(tmp_loss)

    return loss1 + (sum(loss2)/len(loss2))



def cosine_train(model, device, probe_label_loader, gallery_loader, train_idx_to_class, epoch):
    f = open("log.txt", "a")
    model.train()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.7)
    total_loss = []
    for batch_idx, (faces, labels) in enumerate(probe_label_loader):
        faces, labels = faces.to(device), labels.to(device)
        optimizer.zero_grad()
        probe_embeddings = model(faces)
        loss = cosine_embedding_loss(device, probe_embeddings, labels, gallery_loader, train_idx_to_class)
        loss.backward()
        optimizer.step()
        total_loss.append(loss.item())
        if batch_idx % 10 == 0:
            f.write('Train Epoch: {} [{}/{}]\tLoss: {:.6f}\n'.format(
                epoch, batch_idx * len(faces), len(probe_label_loader.dataset), loss.item()))
            print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(faces), len(probe_label_loader.dataset), loss.item()))
    # scheduler.step()
    print("Average Loss in Layer {} is {}.".format(epoch, sum(total_loss)/len(total_loss)))
    f.close()
    return sum(total_loss)/len(total_loss)

def cosine_test(model, device, probe_label_dataset, gallery_dataset, train_idx_to_class):
    f = open("accuracy.txt", "a")
    model.eval()

    gallery_loader = torch.utils.data.DataLoader(gallery_dataset, batch_size=1)
    probe__label_loader = torch.utils.data.DataLoader(probe_label_dataset, batch_size=1, shuffle=True)

    correct = 0
    total = 0
    for idx, (face, label) in enumerate(probe__label_loader):
        #print("probe number {} started.".format(idx))
        face, label = face.to(device), label.to(device)
        probe_embedding = model(face)
        distances = []
        probe_target = train_idx_to_class[label.cpu().item()]
        for gallery_embedding, target in gallery_loader:
            compute_cosine(distances, gallery_embedding[0], probe_embedding.cpu().detach().numpy())

        for i in range(5):
            index = distances.index(min(distances))
            gallery_target = gallery_dataset.__getitem__(index)[1]
            if gallery_target == probe_target:
                correct += 1
                break
            distances.pop(index)
        total += 1
        if total == 500:
            break

    print("Train Accuracy: ", 100 * (correct/500))
    f.write("Train Accuracy: {}\n".format(100 * (correct/500)))
    f.close()

def plot_loss(total_loss):
    x = np.arange(1, len(total_loss)+1)
    plt.plot(x, total_loss, markersize=12, color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('Cosine_Finetune_loss.png')


def cosine_finetune():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model = InceptionResnetV1(pretrained='vggface2')
    model = pickle.load(open("./model/facenet_finetuned_cosine_celebA_srgan.pkl", "rb"))
    model.to(device)
    batch_size = 32

    #probe_dataset = ImageFolderLMDB('./model/HR_Train_remove25_probe_face_gallery_embedding.lmdb')
    #probe_loader = torch.utils.data.DataLoader(probe_dataset, batch_size=batch_size)
    #print("Length of probe dataset: ", len(probe_dataset))


    gallery_dataset = ImageFolderLMDB('./model/HR_gallery_embedding_target.lmdb')
    gallery_loader = torch.utils.data.DataLoader(gallery_dataset, batch_size=1, shuffle=True)
    print("Number of gallery identities: ", len(gallery_dataset))
    # probe_label_dataset = ImageFolderLMDB('./model/Facenet_HR_Train_removeless25_noresize.lmdb')
    probe_label_dataset = ImageFolderLMDB('./model/Facenet_Vgg_Srgan_Train_removeless25_noresize.lmdb')
    probe_label_loader = torch.utils.data.DataLoader(probe_label_dataset, batch_size=batch_size)

    train_idx_to_class = pickle.load(open('./model/HR_train_idx_to_class.pkl', 'rb'))


    cosine_test(model, device, probe_label_dataset, gallery_dataset, train_idx_to_class)

    total_loss = []
    for epoch in range(21, 51):
        print("Epoch: ", epoch)
        total_loss.append(cosine_train(model, device, probe_label_loader, gallery_loader, train_idx_to_class, epoch))
        if epoch % 5 == 0:
            cosine_test(model, device, probe_label_dataset, gallery_dataset, train_idx_to_class)
            pickle.dump(model, open("./model/facenet_finetuned_cosine_celebA_srgan.pkl", "wb"))
            plot_loss(total_loss)


def main():
    # train_NN()
    # cosine_evaluation(dataset="LFW")
    cosine_finetune()


if __name__ == "__main__":
    main()
