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
Part 1: https://drive.google.com/file/d/1RteV9Hwqbqr8MRca4xAm5OW3FatnXcCT/view
Part 2: https://drive.google.com/file/d/16sPB5_CYc3kLfUfO_001vAhASq3fKzpb/view
```

When you run 'main.py', initially, the code will ask you if you would want to train the base model.
- If ‘yes’ is entered, the code will start training the base model and will then save it as “Demo_model” when the validation accuracy converges. If it does not, it will return the variable “history” which holds the validation accuracy and loss for the evaluation.  
- If ‘no’ is entered, the code will predict the labels in our sample folder using the trained model called “finalized_model” and print the actual and predicted labels.

Currently, we specified a manual_seed(42) in line 2 of main.py. This is to make sure that we get the same split for Training Set and Test Set in dataset for every run. To split the dataset into random Training Set and Test Set, please comment out the code in line 2 of main.py
```
torch.manual_seed(42)
```

Our saved trained model in part 2 only works with 
```
(20): Linear(in_features=512, out_features=4, bias=True). But our saved trained model in part 1 only works with  
(20): Linear(in_features=512, out_features=6, bias=True)
```


## Dataset
The Moodle submission only contains the sample dataset with 100 images (25 images per catergory). You can get the full dataset from this repository in the 'dataset' folder, which contains over 1600 images.

To study the bias of the AI, we have split our dataset into two groups. For gender, we have male and female, while race is split between pale skin and dark skin.


## Content of Submission
```
root
│   README.md				readme file
|   main.py					python file that has all the code of the project
|   NS07_COMP472_Report.pdf   	Project Report
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
└─── categorized-dataset               		complete dataset split up into races and genders
│	Image_References.txt		link references for the dataset
|	|
|	└───dark
|	|	└───cloth_dark
│	|	|	52 images
|	|	|
|	|	└───n95_dark
|	│	|	53 images
|	|	|
|	|	└───no_dark
|	│	|	94 images
|	|	|
|	|	└───surgical_dark
|	│	|	25 images
|	|
|	└───female
|	|	└───cloth_female
│	|	|	236 images
|	|	|
|	|	└───n95_female
|	│	|	210 images
|	|	|
|	|	└───no_female
|	│	|	226 images
|	|	|
|	|	└───surgical_female
|	│	|	253 images
|	|
|	└───male
|	|	└───cloth_male
│	|	|	165 images
|	|	|
|	|	└───n95_male
|	│	|	199 images
|	|	|
|	|	└───no_male
|	│	|	194 images
|	|	|
|	|	└───surgical_male
|	│	|	147 images
|	|
|	└───pale
|	|	└───cloth_pale
│	|	|	349 images
|	|	|
|	|	└───n95_pale
|	│	|	356 images
|	|	|
|	|	└───no_pale
|	│	|	326 images
|	|	|
|	|	└───surgical_pale
|	│	|	375 images
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
