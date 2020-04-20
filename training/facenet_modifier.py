import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from facenet_pytorch.models.utils.detect_face import extract_face
from torchvision import transforms, datasets
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset
from PIL import Image
import torch.nn.functional as F
import pickle
import sys
import numpy as np
sys.path.insert(1, '/tmp/pycharm_project_243/face_recognition')
import preprocessing


def load_data(dataset_path):
    aligner = MTCNN(prewhiten=False, keep_all=True, thresholds=[0.6, 0.7, 0.9])
    facenet_preprocess = transforms.Compose([preprocessing.Whitening()])
    samples = []
    transform = transforms.Compose([
        preprocessing.ExifOrientationNormalize(),
        transforms.Resize(1024),
    ])
    dataset = datasets.ImageFolder(dataset_path, transform=transform)
    for img_path, label in dataset.samples:
        print(img_path)
        img = transform(Image.open(img_path).convert('RGB'))
        bbs, _ = aligner.detect(img)
        if bbs is None:
            continue
        faces = torch.stack([extract_face(img, bb) for bb in bbs])
        preprocessed_faces = facenet_preprocess(faces)
        samples.append((faces[0], label))
    return samples


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = InceptionResnetV1(pretrained='vggface2')
    # print(model)
    for param in model.parameters():
        param.requires_grad = False

    model.last_linear = nn.Linear(in_features=1792, out_features=1024, bias=False)
    model.last_bn = nn.BatchNorm1d(1024, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    model.logits = nn.Linear(in_features=1024, out_features=8631, bias=True)

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=7, gamma=0.7)

    # load train vggfaces2

    dataset_path = "/home/nbayat5/Desktop/VggFaces/train"
    train_data = load_data(dataset_path)
    pickle.dump(train_data, open('/home/nbayat5/face-recognition-master/training/model/vgg_data.pkl', 'wb'))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    print("Data Loaded!")


    for epoch in range(1, 20):
        print("Epoch: ", epoch)
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset), loss.item()))
        scheduler.step()

    pickle.dump(model, open("/home/nbayat5/face-recognition-master/training/model/facenet_1024.pkl", "wb"))


if __name__ == "__main__":
    main()