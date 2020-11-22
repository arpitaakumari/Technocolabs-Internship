# Technocolabs-Internship
1 month Computer Vision internship at Technocolabs.

Before we walk through the steps of building this project, let us see a demo of this project.

![GithubScreen](https://user-images.githubusercontent.com/60249904/99908207-4de7f400-2d07-11eb-8734-0c0976106520.gif)

## Understanding the Project
Thanks to Corona Virus, face masks have now become a integral part of our lives XD
This project is made with the aim of enforcing masks on people.
This Face Mask detector Model will detect in two steps. First it will detect the face, and only if face gets detected it will move towards detecting whether the person wears a mask or not. The reason I did this is if no face detector model is used then it will detect the masks placed on a table, which is not the aim of the project !
Hence, both face and mask detector model will be used. The project will also save the persons face who are not wearing the mask. It gets saved in the nomaskss folder.

The model is then deployed on Web App, using Flask.


## Step 1 : Getting the dataset for our Face Mask Detector Model
I gathered the images from various online sources. I trained the model on 120000+ images.
The links of the various online resources are : 

* [Link 1](http://github.com)
* [Link 2](https://www.kaggle.com/shreyashwaghe/face-mask-dataset)
* [Link 3](https://www.kaggle.com/omkar1008/covid19-mask-detection)
* [Link 4](https://www.kaggle.com/kiranbeethoju/face-mask-and-kerchief)
* [Link 5](https://www.kaggle.com/ashishjangra27/face-mask-12k-images-dataset)

I organised the dataset into 2 seperate folders named 0(without mask) and 1(with mask) inside my dataset folder.

![data](https://user-images.githubusercontent.com/60249904/99908552-56412e80-2d09-11eb-8bef-06296b664fce.png)
Just like this !

## Step 2 : Creating the Mask Detector Model
For this project I preapred a Mask Detector Model from scratch, but for the Face Detection I will use a pretrained model. 
Follow facemaskmodel.py for the python codes of this project. All code blocks have all the necesaary comments. Do change your dataset folder location as per your local machine. The model prepared with the help of these python code is saved in model.h5 file. 
If you wish, you can download my model.h5 directly, no need to give hours in the training procedure XD (The file is present in the github repository)

## Step 3 : Writing the codes for detection
Here, for face detection I have used a pretrained model. And for mask, we have already prepared the model.
Follow detection_mask.py for python codes. here I have returned the frames so that it can run on my localhost web app.

## Step 4 : Deployment using Flask
Follow app.py and camera.py for this segment of code.

## Final Video of the working model 

![MaskProjectScreenVideo](https://user-images.githubusercontent.com/60249904/99911509-e76cd100-2d1a-11eb-8109-6bae2b90be3b.gif)

