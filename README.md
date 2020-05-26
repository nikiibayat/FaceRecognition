# Low Resolution Face Recognition
In this project we are aiming to recognize the identity of very low-resolution (16x16) face images using SRGAN super-resolution method. </br>
I used CelebA dataset to train SRGAN and SRCNN models to super resolve LR (16x16) faces into HR (64x64) counterparts. This dataset can be downloaded from: [CelebA Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)</br>
I used this implementation to train srgan using Keras API. [SRGAN Code](https://github.com/eriklindernoren/Keras-GAN#srgan)</br>
For sake of comparison, I also super resolved celebA dataset using another common SR technique called SRCNN. [SRCNN Code](https://github.com/titu1994/Image-Super-Resolution)</br>
The I use this repository to first crop the face using MTCNN, then extract facial embeddings using Inception-Resnet-V3 network and lastly, train a custom neural network to find mapping between embeddings and corresponding labels in celebA.</br>
This code is downloaded from [Face Recognition Code](https://github.com/arsfutura/face-recognition), however so many files or lines of code have been altered to match my purpose and dataset.</br>
In order to run the code you need to:</br>
1. Download CelebA cropped dataset and the "identity_CelebA.txt" file.
2. The convert dataset into ImageFolder format using the text file.
3. For each identity, pick one image for test, one for validation and the rest for train. If an identity has less than 3 samples discard them. You should be left with 9809 identities.
4. resize all faces to 64x64 to create grand truth dataset and to 16x16 to create LR dataset (two different copies of dataset).
5. Super resolve LR dataset with SRGAN and SRCNN separately.
6. Run ./training/train.py to crop faces and extract embeddings for each dataset. (using load_data method commented in main)
7. Afterwards, comment load data and use either neural_network_classifier or logistic_regression_classifier to train mapping between embedding and label for all 3 (HR, SRGAN, SRCNN) datasets.
8. If you are willing to compare SRGAN and SRCNN face detection performance, you can uncomment bounding_box_comparison method.

feel free to email me if you had any further questions.</br>
![Sample super resolution of an identity vs grand truth](https://github.com/nikiibayat/FaceRecognition/blob/master/fonts/MTCNN.PNG?raw=true)
</br>
---
</br>
![SR_Comparison](https://github.com/nikiibayat/FaceRecognition/blob/master/SR_Comparison.PNG?raw=true)
