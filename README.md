# COMP472_Project

## Team (NS_07)
- Jason Dinh (40129138) - Data Specialist
- Axel Dzeukou (40089940) - Training Specialist
- Vyacheslav Medvedenko (40134207) - Evaluation Specialist
- Dante Di Domenico (40125704) - Compliance Specialist

## Instruction
All our code are in 'main.py' file in the root folder. To get this code run correctly, you need to install the following packages:
- PyTorch
- TorchVision
- sklearn

#### IMPORTANT: Getting the Trained Model
###### We cannot include out trained model in the Moodle submission and on Git because the size is too large for both (350+ MB). Please download the model from our Google Drive and save it in the root folder (the same folder as the 'main.py' file). Link below:
```
Trained model
https://drive.google.com/file/d/1RteV9Hwqbqr8MRca4xAm5OW3FatnXcCT/view?usp=sharing
```

When you run 'main.py', initially, the code will ask you if you would want to train the base model.
- If ‘yes’ is entered, the code will start training the base model and will then save it as “Demo_model” when the validation accuracy is at least 60%. If it does not, it will return the variable “history” which holds the validation accuracy and loss for the evaluation.  
- If ‘no’ is entered, the code will predict the labels in our sample folder using the trained model called “finalized_model” and print the actual and predicted labels.

Currently, we specified a manual_seed(42) in line 2 of main.py. This is to make sure that we get the same split for Training Set and Test Set in dataset for every run. To split the dataset into random Training Set and Test Set, please comment out the code in line 2 of main.py
```
torch.manual_seed(42)
```

## Dataset
The Moodle submission only contains the sample dataset with 100 images (25 images per catergory). You can get the full dataset from this repository in the 'dataset' folder, which contains over 1600 images.


## Content of Submission
```
root
│   README.md				readme file
|   main.py					python file that has all the code of the project
|   NS07_COMP472_Report.pdf   	Project Part 1 Report
|   NS_07_COMP472_Expectations-of-Originality.pdf
│
└───dataset               		complete dataset
│	Image_References.txt		link references for the dataset
|	|
|	└───cloth_mask
│	|	401 images
|	|
|	└───n95_mask
│	|	410 images
|	|
|	└───no_mask
│	|	420 images
|	|
|	└───surgical_mask
│	|	400 images
|	|
└───sample-dataset			a subset of size 100 of the full dataset 
│	Image_References.txt		link references for the dataset
|	└───cloth_mask
│	|	25 images
|	└───n95_mask
│	|	25 images
|	└───no_mask
│	|	25 images
|	└───surgical_mask
│	|	25 images
```
