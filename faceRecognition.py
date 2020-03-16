import os
from skimage.transform import resize
import imageio
import numpy as np
import glob
import scipy

def main():
    rootdir = "/home/nbayat5/Desktop/celebA/identities"
    #os.mkdir("/home/nbayat5/Desktop/celebA/face_recognition_srgan")
    for subdir, dirs, files in os.walk(rootdir):
        for dir in dirs:
            path = os.path.join(rootdir, subdir)
            parts = path.split("/")
            if len(parts) == 6:
                continue
            os.mkdir("/home/nbayat5/Desktop/celebA/face_recognition_srgan_test/%s" % (parts[6].rstrip()))
            imgs_hr, imgs_lr = load_dataforIdentities(path)
            counter = 1
            for img in imgs_hr:
                # fake_hr = gan.generator.predict(img_lr) #fix for loop to lr
                img = 0.5 * img + 0.5
                img = np.asarray(img)
                path_hr = "/home/nbayat5/Desktop/celebA/face_recognition_srgan_test/%s/%s_%d.png" % (
                parts[6].rstrip(), parts[6].rstrip(), counter)
                imageio.imwrite(path_hr, img)
                print("img %s_%d.png saved." % (parts[6].rstrip(), counter))
                counter += 1
            break


def load_dataforIdentities(path):
        imgs_hr = []
        imgs_lr = []
        os.chdir(path)
        # train_images = glob.glob("./train/*.jpg")
        # val_images = glob.glob("./validation/*.jpg")
        test_images = glob.glob("./test/*.jpg")
        # batch_images = train_images + val_images
        # batch_images = np.random.choice(path2, size=1)
        for img_path in test_images:
            img = scipy.misc.imread(img_path, mode='RGB').astype(np.float)

            img_hr = scipy.misc.imresize(img, (64, 64))
            img_lr = scipy.misc.imresize(img, (16, 16))


            imgs_hr.append(img_hr)
            imgs_lr.append(img_lr)

        imgs_hr = np.array(imgs_hr) / 127.5 - 1.
        imgs_lr = np.array(imgs_lr) / 127.5 - 1.

        return imgs_hr, imgs_lr


if __name__ == "__main__":
	main()
