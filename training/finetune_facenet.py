import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data.dataloader import default_collate
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import numpy as np
import os
import joblib
import sys
import inception_resnet_v1
import mtcnn

sys.path.insert(2, '/tmp/pycharm_project_243/training/utils')
import training
import pickle

data_dir = '/home/nbayat5/Desktop/celebA/face_recognition_train_srgan'  # srgan train data
model_path = os.path.join('model', 'finetuned_resnet.pkl')

batch_size = 32
epochs = 10
workers = 0 if os.name == 'nt' else 8


def fixed_image_standardization(image_tensor):
    processed_tensor = (image_tensor - 127.5) / 128.0
    return processed_tensor


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

"""
mtcnn = mtcnn.MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)


dataset = datasets.ImageFolder(data_dir)  # loads a data set in folders format
dataset.samples = [
    (p, p.replace(data_dir, data_dir + '_cropped'))
    for p, _ in dataset.samples
]

loader = DataLoader(
    dataset,
    num_workers=workers,
    batch_size=batch_size,
    collate_fn=training.collate_pil
)

for i, (x, y) in enumerate(loader):
    mtcnn(x, save_path=y)
    print('\rBatch {} of {}'.format(i + 1, len(loader)), end='')

# Remove mtcnn to reduce GPU memory usage
del mtcnn
"""
def my_collate(batch):
    batch = list(filter (lambda x:x is not None, batch))
    return default_collate(batch)

resnet = inception_resnet_v1.InceptionResnetV1(
    # classify=True,
    pretrained='casia-webface',
    num_classes=10575
).to(device)

trans = transforms.Compose([
    np.float32,
    transforms.ToTensor(),
    fixed_image_standardization
])

class ImageFolderWithPaths(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super(ImageFolderWithPaths, self).__init__(root, transform=transform)
        with open('../finetune_train_datalabel_embedding_dictionary.pkl', 'rb') as f:
            self.emb = pickle.load(f)

    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        filename = path.split('/')[len(path.split('/')) - 1].split('.')[0]
        try:
            return original_tuple + (self.emb[filename],)
        except Exception as e:
            print(e)


dataset = ImageFolderWithPaths(data_dir + '_cropped', transform=trans)
dataset = torch.utils.data.DataLoader(dataset, num_workers=workers, batch_size=batch_size,
                                      sampler=SubsetRandomSampler(np.arange(len(dataset))), collate_fn=my_collate)



loss_fn = torch.nn.MSELoss()
optimizer = optim.Adam(resnet.parameters(), lr=0.001)
scheduler = MultiStepLR(optimizer, [5, 10])
writer = SummaryWriter()
resnet.train()

for epoch in range(epochs):
    dataloader_iterator = iter(dataset)
    for step, (images, labels, embeddings) in enumerate(dataloader_iterator):
        np.squeeze(images)
        print('\nEpoch {}/{}'.format(epoch + 1, epochs))
        print("Iteration: ", step)
        optimizer.zero_grad()
        y_hat = resnet(images.to(device))
        loss = loss_fn(y_hat, embeddings.to(device))
        print("loss: ", loss)
        loss.backward()
        optimizer.step()
        writer.add_scalar('loss', loss.mean(), epoch * step)


writer.close()
joblib.dump(resnet, model_path)