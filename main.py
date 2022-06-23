import torch

torch.manual_seed(42)
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import KFold
from torch.utils.data.dataset import Subset

CUDA_LAUNCH_BLOCKING = 1


# Base Model For Image Classification:
class ImageClassificationBase(nn.Module):

    def training_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        acc = accuracy(out, labels)  # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))


# CNN Model For Classification:
class FaceMaskClassification(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(

            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),
            nn.Linear(82944, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 4)
        )

    def forward(self, xb):
        return self.network(xb)


# Extra functions to aid model training and evaluation:

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


# train the model and after every epoch save the model that gives us a validation accuracy >=60%
def fit(epochs, lr, model, train_loader, val_loader, opt_func, fold_value):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    previous_valaccuracy = 0
    for epoch in range(epochs):

        model.train()
        train_losses = []

        for batch in train_loader:
            # batch[1]=torch.LongTensor(list(map(lambda x:x-1,batch[1].tolist())))
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # validation data evaluation
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        #         result['train_acc'] = evaluate(model,train_loader)['val_acc']
        model.epoch_end(epoch, result)
        history.append(result)
        #         if abs(result['val_acc']-previous_valaccuracy)<0.001:
        #             model_name = "Model_Number_" + str(fold_value)
        #             return [model_name,model,result['val_acc']]

        if epoch == epochs - 1:
            model_name = "Model_Number_" + str(fold_value)
            return [model_name, model, result['val_acc']]
        previous_valaccuracy = result['val_acc']

    print("desired accuracy wasnt obtained so return model after latest epoch ")
    model_name = "Model_Number_" + str(fold_value)
    return [model_name, model, previous_valaccuracy]


# used to print predicted labels and evaluation such as confusion matrix and f1-score
def test(model, test_dl, number_ofclasses):
    list_labels = []  # list of labels (actual class values of images)
    list_preds = []  # list of predictions (nn's guesses)

    all_preds = []
    all_outputs = []
    all_labels = []

    # collecting labels and predictions from batches into lists
    for batch in test_dl:
        images, labels = batch
        outputs = model(images)
        for i in labels:
            list_labels.append(i.item())

        _, preds = torch.max(outputs, dim=1)

        all_outputs += outputs

        all_labels += labels
        all_preds += preds

        for i in preds:
            list_preds.append(torch.max(i).item())

    same = 0
    not_same = 0

    for k in range(len(list_labels)):
        if int(list_preds[k]) == int(list_labels[k]):
            same += 1
        elif int(list_preds[k]) != int(list_labels[k]):
            not_same += 1

    y_true = list_labels
    y_pred = list_preds

    print('y_pred: ', y_pred)
    print('y_true: ', y_true)

    print('****************************************')
    print('\t\tConfusion Matrix')
    print('****************************************')
    print('Columns are predictions, rows are labels\n')

    cf_matrix = confusion_matrix(y_true, y_pred)
    print(cf_matrix)

    fn = 0  # False Negative
    fp = 0  # False Positive
    tp = 0  # True Positive
    tn = 0  # True Negative
    for c in range(number_ofclasses):  # calculating the above values from the cf_matrix
        tp += cf_matrix[c][c]
        for r in range(number_ofclasses):
            if r != c:
                fp += cf_matrix[r][c]
        for i in range(number_ofclasses):
            if i != c:
                fn += cf_matrix[c][i]

    for c in range(number_ofclasses):  # calculating the above values from the cf_matrix
        fn = 0  # False Negative
        fp = 0  # False Positive
        tp = 0  # True Positive

        tp += cf_matrix[c][c]
        for r in range(number_ofclasses):
            if r != c:
                fp += cf_matrix[r][c]
        for i in range(number_ofclasses):
            if i != c:
                fn += cf_matrix[c][i]

        tn = len(list_labels) - fn - fp - tp
        print('\nCLASS #', str(c))

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        format_float = "{:.2f}".format(accuracy)
        print('Accuracy: ', format_float)
        precision = tp / (tp + fp)
        format_float = "{:.2f}".format(precision)
        print('Precision: ', format_float)
        recall = tp / (tp + fn)
        format_float = "{:.2f}".format(recall)
        print('Recall: ', format_float)
        f1_measure = 2 * (precision * recall) / (precision + recall)
        format_float = "{:.2f}".format(f1_measure)
        print('F1_score: ', format_float)

    print('\n\nTotal (all classes together):')

    accuracy = same / (not_same + same)
    print('Accuracy: ', accuracy)
    precision = precision_score(y_true, y_pred, average='micro')
    print('Precision: ', precision)
    recall = recall_score(y_true, y_pred, average='micro')
    print('Recall: ', recall)
    score = f1_score(y_true, y_pred, average='micro')
    print('F1_score: ', score)


# bias results
def test_without_bias(model, batchsize, option):
    print('\t--- 4 tests within each subgroup (male/female/dark/pale) to avoid biases that arent related to masks')
    
    if (option == 0):
        mask_male_dir = "./categorized-dataset/male"
        mask_female_dir = "./categorized-dataset/female"
        mask_dark_dir = "./categorized-dataset/dark"
        mask_pale_dir = "./categorized-dataset/pale"
    elif (option == 1):
        mask_male_dir = "./sample-categorized-dataset/male"
        mask_female_dir = "./sample-categorized-dataset/female"
        mask_dark_dir = "./sample-categorized-dataset/dark"
        mask_pale_dir = "./sample-categorized-dataset/pale"

    
    print('### MALE ###')
    male_dataset = ImageFolder(mask_male_dir, transform=transforms.Compose([
        transforms.Resize((150, 150)), transforms.ToTensor()
    ]))
    male_test_dl = DataLoader(male_dataset, batchsize, num_workers=4, pin_memory=True)  # batch size is 64
    test(model, male_test_dl, len(male_dataset.classes))

    print('### FEMALE ###')
    female_dataset = ImageFolder(mask_female_dir, transform=transforms.Compose([
        transforms.Resize((150, 150)), transforms.ToTensor()
    ]))
    female_test_dl = DataLoader(female_dataset, batchsize, num_workers=4, pin_memory=True)
    test(model, female_test_dl, len(female_dataset.classes))

    print('### DARK ###')
    dark_dataset = ImageFolder(mask_dark_dir, transform=transforms.Compose([
        transforms.Resize((150, 150)), transforms.ToTensor()
    ]))
    dark_test_dl = DataLoader(dark_dataset, batchsize, num_workers=4, pin_memory=True)
    test(model, dark_test_dl, len(dark_dataset.classes))

    print('### PALE ###')
    pale_dataset = ImageFolder(mask_pale_dir, transform=transforms.Compose([
        transforms.Resize((150, 150)), transforms.ToTensor()
    ]))
    pale_test_dl = DataLoader(pale_dataset, batchsize, num_workers=4, pin_memory=True)
    test(model, pale_test_dl, len(pale_dataset.classes))


if __name__ == "__main__":
    # train and test directory
    data_dir = "./dataset"

    request = input('Do you want to train the base model (yes/no): ')
    # Preparing the Dataset :
    # To prepare a dataset from such a structure, PyTorch provides ImageFolder class which makes the task easy for us
    # to prepare the dataset.
    # We simply have to pass the directory of our data to it and it provides the dataset which we can use to train the model.

    # load the train and test data
    # The torchvision.transforms module provides various functionality to preprocess the images,
    # here first we resize the image for (150*150) shape and then transforms them into tensors.
    dataset = ImageFolder(data_dir, transform=transforms.Compose([
        transforms.Resize((150, 150)), transforms.ToTensor()
    ]))

    if request == 'yes':
        # The image label set according to the class index in data.classes.
        print("Follwing classes are there : \n", dataset.classes)

        # output:
        # Follwing classes are there :
        # ['cloth_mask', 'n95_mask', 'no_mask', 'surgical_mask']

        # Splitting Data and Prepare Batches:
        batch_size = 64
        val_size = 246
        test_size = 400
        train_size = len(dataset) - test_size

        train_data, test_data = random_split(dataset, [train_size, test_size])
        print(f"Length of Train+validation Data : {len(train_data)}")
        print(f"Length of test Data : {len(test_data)}")

        # load the test into batches.
        test_dl = DataLoader(test_data, batch_size, num_workers=4, pin_memory=True)

        kfold = KFold(n_splits=10, shuffle=True, random_state=None)
        fold_value = 1

        num_epochs = 10
        opt_func = torch.optim.Adam
        lr = 0.001

        for training_id, validation_id in kfold.split(train_data):
            model = FaceMaskClassification()
            print("Fold Number:", fold_value)

            training_data = Subset(train_data, training_id)
            val_data = Subset(train_data, validation_id)
            print(f"Length of Train Data : {len(training_data)}")
            print(f"Length of Validation Data : {len(val_data)}")

            # load the train and validation into batches.
            training_dl = DataLoader(training_data, batch_size, shuffle=True, num_workers=4, pin_memory=True)
            val_dl = DataLoader(val_data, batch_size, num_workers=4, pin_memory=True)

            # train and return model with its validation accuracy then append to the list
            model_data = fit(num_epochs, lr, model, training_dl, val_dl, opt_func, fold_value)
            # run the model on test
            print('\nEvaluation for trained model part 2 fold:', fold_value)
            test(model_data[1], val_dl, len(dataset.classes))
            fold_value += 1

        modeldata = fit(num_epochs, lr, FaceMaskClassification(), training_dl, test_dl, opt_func, 0)

        # save trained model:
        torch.save(modeldata[1].state_dict(),
                   "./TrainedPart2_model.sav")


        print('\nTest Evaluation part 2:')
        model = FaceMaskClassification()
        model.load_state_dict(
            torch.load("./TrainedPart2_model.sav"),
            strict=False)
        test(model, test_dl, len(dataset.classes))



        # run the model on categorized dataset
        test_without_bias(model, 64, 0)

    elif request == "no":
        # sample directory
        sample_dir = "./sample-dataset"

        # prepare sample datasets
        sample_dataset = ImageFolder(sample_dir, transform=transforms.Compose([
            transforms.Resize((150, 150)), transforms.ToTensor()
        ]))

        sample_dl = DataLoader(sample_dataset, 64, num_workers=4, pin_memory=True)

        # To restore trained part2 model model:
        model = FaceMaskClassification()
        model.load_state_dict(
            torch.load("./TrainedPart2_model.sav"),
            strict=False)

        # run the model on sample
        test(model, sample_dl, len(dataset.classes))

        # run the model on sample category
        test_without_bias(model, 64, 0)


