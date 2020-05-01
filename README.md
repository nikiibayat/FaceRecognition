# Low Resolution Face Recognition
In this project we are aiming to recognize the identity of very low-resolution (16x16) face images using SRGAN super-resolution method. </br>
I used CelebA dataset to train SRGAN and SRCNN models to super resolve LR (16x16) faces into HR (64x64) counterparts. This dataset can be downloaded from: [CelebA Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)</br>
I used this implementation to train srgan using Keras API. [SRGAN Code](https://github.com/eriklindernoren/Keras-GAN#srgan)</br>
For sake of comparison, I also super resolved celebA dataset using another common SR technique called SRCNN. [SRCNN Code](https://github.com/titu1994/Image-Super-Resolution)</br>
