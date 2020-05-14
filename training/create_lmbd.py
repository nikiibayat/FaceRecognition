import torch
from facenet_pytorch import MTCNN
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


def folder2lmdb(dpath, name="train", write_frequency=5000, num_workers=16):
    batch_size = 128
    print("Loading dataset from %s" % dpath)
    transform = transforms.Compose([
        preprocessing.ExifOrientationNormalize(),
        transforms.Resize(1024),
        transforms.ToTensor(),
    ])
    dataset = ImageFolder(dpath, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    print(len(dataset.samples))
    print(len(data_loader))
    lmdb_path = osp.join("./model", "%s.lmdb" % name)
    isdir = os.path.isdir(lmdb_path)

    print("Generate LMDB to %s" % lmdb_path)
    db = lmdb.open(lmdb_path, subdir=isdir,
                   map_size=10e+11, readonly=False,
                   meminit=False, map_async=True)

    aligner = MTCNN(keep_all=True, thresholds=[0.6, 0.7, 0.9])
    facenet_preprocess = transforms.Compose([preprocessing.Whitening()])

    ii = 0
    txn = db.begin(write=True)
    for idx, (images, labels) in enumerate(data_loader):
        for j in range(images.shape[0]):
            image = transforms.ToPILImage(mode="RGB")(images[j])
            bbs, _ = aligner.detect(image)
            if bbs is None:
                continue
            faces = torch.stack([extract_face(image, bb) for bb in bbs])
            preprocessed_faces = facenet_preprocess(faces)
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


def dumps_pyarrow(obj):
    """
    Serialize an object.
    Returns:
        Implementation-dependent bytes-like object
    """
    return pickle.dumps(obj)


folder2lmdb('/home/nbayat5/Desktop/celebA/face_recognition_without_bbx/face_recognition_HR_train', 'HR_Train_MTCNN_cropped')
