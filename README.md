# Domain Adaptation and Semantic Segmentation using UVCGAN and nnUNet
Training and inference code for [Crossmoda Challenge](https://crossmoda-challenge.ml/) MICCAI 2022 for segmentation task. The code includes the image to image translation from T1 to T2 images through UVCGAN and then utilizes the nnUNet to create segmentation. 
More comprehensively, we divide the project into two phases. First, we used one of the cycleGAN variants to translate T1 images to synthetic T2. Then, in the second phase, we used different variants of nnunet for the segmenta-tion of the synthetic T2 images. Furthermore, we used the trained model for self-training to produce pseudo-labels for unlabeled T2 images. In the end, we trained a new model with synthetic images and T2 images with pseudo-labels.  


# Table Of Contents
-  [Installation](#installation)
-  [Image-to-Image Translation](#image-to-image-translation)
-  [Segmentation](#segmentation)
-  [Code Structure](#code-structure)
-  [Future work](#future-work)
-  [Acknowledgments](#acknowledgments)
-  [Contributors](#contributors)
-  [Contributing](#contributing)

# Installation   
You need first and foremost to run 
```console
foo@bar:~$ cd crossmoda-challenge
foo@bar:~/crossmoda-challenge$ pip install -r requirements.txt
```


# Image-to-Image Translation   
For training the images for image translation
```console
foo@bar:~/crossmoda-challenge$ cd uvcgan
foo@bar:~/crossmoda-challenge/uvcgan$ python scripts/train/selfie2anime/cyclegan_selfie2anime-256.py
```
This will save trained model in `outdir` folder inside uvcgan folder.
 
Next, in `predict_img.py`  to create synthesized T2 images in 3D used the function ins.predict(), you may need to define the output path. Image output would have size 224x224, therefore to created labels abiding the same dimension funnel those label nifti files through ins.transform_labels(). After that your data is ready to be used for segmentation task.

```python
ins = Instructor(opt)
# get us sythetic T2 given ceT1
# ins.predict()
# get us synthetic T2 in graycale img form (not viable)
# ins.predict_syn_imgs()
# nifti to nifti labels (compatible to images e.g cropped and resized) transformation
ins.transform_labels()
```
# Segmentation
For segmentation, you must follow the nnUNet folder structure for inputting the input, run the `rename_for_single_modality` function in helper.py,  `generate_dataset_json` function (example present in playground.ipynb) from nnUNet's utils. Then run the following commands (don't forget to specify your input and output folder for inference command):
```console
foo@bar:~/crossmoda-challenge/uvcgan$ cd ..
foo@bar:~/crossmoda-challenge$ nnUNet_plan_and_preprocess -t 001 --verify_dataset_integrity
foo@bar:~/crossmoda-challenge$ nnUNet_train 3d_fullres nnUNetTrainerV2_DA5 Task001_Final 5
foo@bar:~/crossmoda-challenge$ nnUNet_predict -chk model_best -i [input_folder] -o [output_folder] -t Task001_Final
```

# Code Structure
The code includes some helpful files for other usecases then this one. Feel free to use them at your disposal. But the files crucial for this project are shown in a structure below:
```
├──  datasets
│    └── cyclegan.py  - Main dataset file class.
│
├──  nnUNet  - Customized version of the main nnUNet (link in the acknowledgement). 
│ 
│
├──  uvcgan  - Customized version of the uvcgan (link in the acknowledgement). 
│    └── scripts/train/selfie2anime/cyclegan_selfie2anime-256.py  - for training the T1 images for image translation to T2 (requires source folder `T1` and target     |                                                                     folder `T2`).
│
├──  utils
│   ├── helper.py     - this file contains helper functions.
│   └── normalization.py   - this file contains the normalization of the dataset.
│
│
├── predict_img.py - contain predict() and transform_labels() which makes the synthetic images and compatible labels (cropped and resized to have the same size as     |                     synthesized output).
```

# Docker Image for inference
We used Docker to containerize the code. The shipped code contains just an inference part, to keep the docker image as light as possible. 

```console
foo@bar:~$ docker run --rm --gpus all  -v [input directory]:/input/:ro -v [output directo-ry]:/output -it [image name]
```

The code requires GPU which can be utilized into the container through --gpus all parameter. Dockerhub image name is uzairnoman/crossmoda:latest.



# Future work
We really think that normalizing the dataset before working would have improved the results and training routine. Extensive hyperparamter tuning would have helped as well.

# Acknowledgments
- [nnUNet](https://github.com/MIC-DKFZ/nnUNet) (For segmentation in 3D)
- [UVCGAN](https://github.com/LS4GAN/uvcgan) (For image to image translation)
- [SMP](https://github.com/qubvel/segmentation_models.pytorch) (Open source library for segmentation models)
- [MONAI](https://github.com/Project-MONAI/MONAIh) (Open source library for 3D image processing for medical datasets)


# Contributors
- [Mina Rezaei](https://github.com/MinaRe)
- [Amirhossein Vahidi](https://github.com/amirvhd)

# Contributing
Any kind of enhancement or contribution is welcomed.

