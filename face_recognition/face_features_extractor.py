import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import transforms
import preprocessing
from facenet_pytorch.models.utils.detect_face import extract_face
import joblib
from torchvision.transforms import ToTensor
import sys
sys.path.insert(1, '/tmp/pycharm_project_243/training')
import inception_resnet_v1

class FaceFeaturesExtractor:
    def __init__(self):
        self.aligner = MTCNN(prewhiten=False, keep_all=True, thresholds=[0.6, 0.7, 0.9])
        self.facenet_preprocess = transforms.Compose([preprocessing.Whitening()])
        self.facenet = InceptionResnetV1(pretrained='casia-webface').eval()
        # self.facenet = joblib.load('/tmp/pycharm_project_243/training/model/finetuned_resnet.pkl').eval()

    def extract_features(self, img):
        # if using MTCNN
        try:
            bbs, _ = self.aligner.detect(img)
        except Exception as e:
            print(e)
        if bbs is None:
            # if no face is detected
            return None, None

        faces = torch.stack([extract_face(img, bb) for bb in bbs])

        #if using already cropped faces
        # bbs = None
        # faces = torch.stack([img])
        preprocessed_faces = self.facenet_preprocess(faces)
        temp = self.facenet(preprocessed_faces)
        embeddings = temp.detach().numpy()
        return bbs, embeddings

    def __call__(self, img):
        return self.extract_features(img)
