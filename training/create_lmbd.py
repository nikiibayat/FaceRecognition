import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from facenet_pytorch.models.utils.detect_face import extract_face
import os
import os.path as osp
import pickle
import lmdb
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import sys
sys.path.insert(1, '../face_recognition')
import preprocessing
from torch.utils.data import Dataset
from PIL import Image

class CelebA_Dataset(Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, samples):
        'Initialization'
        self.samples = samples

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.samples)

  def __getitem__(self, index):
        return self.samples[index][0], self.samples[index][1]


def folder2lmdb(samples, name="train", write_frequency=5000, num_workers=16):
    transform = transforms.Compose([
        preprocessing.ExifOrientationNormalize(),
        # transforms.Resize(1024),
        # transforms.ToTensor(),
    ])
    batch_size = 128
    dataset = CelebA_Dataset(samples)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    print("Number of training samples in total: {}".format(len(samples)))
    print("Number of batches: {}".format(len(data_loader)))
    lmdb_path = osp.join("./model", "%s.lmdb" % name)
    isdir = os.path.isdir(lmdb_path)

    print("Generate LMDB to %s" % lmdb_path)
    db = lmdb.open(lmdb_path, subdir=isdir,
                   map_size=10e+11, readonly=False,
                   meminit=False, map_async=True)

    aligner = MTCNN(keep_all=True, thresholds=[0.6, 0.7, 0.9])
    facenet_preprocess = transforms.Compose([preprocessing.Whitening()])
    facenet = InceptionResnetV1(pretrained='vggface2').eval()

    ii = 0
    txn = db.begin(write=True)
    for idx, (images, labels) in enumerate(data_loader):
        for j in range(len(images)):
            image = transform(Image.open(images[j]).convert('RGB'))
            bbs, _ = aligner.detect(image)
            if bbs is None:
                continue
            faces = torch.stack([extract_face(image, bb) for bb in bbs])
            preprocessed_faces = facenet_preprocess(faces)
            temp = facenet(preprocessed_faces)
            embeddings = temp.detach().numpy()
            print("putting image {} with label {}".format(ii, labels[j].shape))
            # txn.put(u'{}'.format(ii).encode('ascii'), dumps_pyarrow((embeddings[0], labels[j])))
            txn.put(u'{}'.format(ii).encode('ascii'), dumps_pyarrow((preprocessed_faces[0], labels[j])))
            ii += 1
            if ii % write_frequency == 0:
                print("[%d/%d]" % (ii, len(data_loader)*batch_size))
                txn.commit()
                txn = db.begin(write=True)

    # finish iterating through dataset
    txn.commit()
    keys = [u'{}'.format(k).encode('ascii') for k in range(ii+1)]
    with db.begin(write=True) as txn:
        txn.put(b'__keys__', dumps_pyarrow(keys))
        txn.put(b'__len__', dumps_pyarrow(len(keys)))

    print("Flushing database ...")
    db.sync()
    db.close()


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

def clean_celebA(Train_path, Test_path, mode='train'):
    gallery_dataset = ImageFolderLMDB('./model/HR_gallery_embedding_target.lmdb')
    gallery_loader = torch.utils.data.DataLoader(gallery_dataset, batch_size=1)
    gallery_targets = []
    gallery_target_embedding = {}
    for (embedding, target) in gallery_loader:
        print("gallery target {} appended.".format(target[0]))
        gallery_targets.append(target[0])
        gallery_target_embedding[target[0]] = embedding[0]

    print("-------------------------------------")
    print("number of unique targets in gallery: ", len(gallery_target_embedding.keys()))


    print("Loading dataset from %s" % Train_path)
    train_dataset = ImageFolder(Train_path)
    test_dataset = ImageFolder(Test_path)
    print("Datasets loaded!")
    samples = []
    remove = []
    count = 0
    current_class = train_dataset.samples[0][1]
    train_idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}
    test_idx_to_class = {v: k for k, v in test_dataset.class_to_idx.items()}

    for img_path, label in train_dataset.samples:
        if label == current_class:
            count += 1
        else:
            if count < 25:
                remove.append(train_idx_to_class[current_class])
            current_class = label
            count = 1

    if mode == 'train':
        dataset = train_dataset
        idx_to_class = train_idx_to_class
    else:
        dataset = test_dataset
        idx_to_class = test_idx_to_class

    unique_labels = []
    for img_path, label in dataset.samples:
        if (idx_to_class[label] not in remove) and (idx_to_class[label] in gallery_targets):
            samples.append((img_path, label))
            # samples.append((img_path, gallery_target_embedding[idx_to_class[label]]))
            # samples.append((img_path, idx_to_class[label]))
            if label not in unique_labels:
                unique_labels.append(label)

    print("{} identities in total were removed!".format(len(remove)))

    # pickle.dump(train_idx_to_class, open('./model/HR_train_idx_to_class.pkl', 'wb'))
    # pickle.dump(test_idx_to_class, open('./model/HR_test_idx_to_class.pkl', 'wb'))


    print("Number of samples: ", len(samples))
    print("Number of unique labels: ", len(unique_labels))
    return samples


def dumps_pyarrow(obj):
    return pickle.dumps(obj)


root = '/home/nbayat5/Desktop/celebA/face_recognition_without_bbx/'
Train_path = root + 'VGG_SRGAN_Train'
Test_path = root + 'face_recognition_HR_test'

samples = clean_celebA(Train_path, Test_path, mode='train')
folder2lmdb(samples, 'Facenet_Vgg_Srgan_Train_removeless25_noresize')
