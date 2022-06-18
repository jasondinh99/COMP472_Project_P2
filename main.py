import torch
torch.manual_seed(42)
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

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
            nn.Linear(512, 6)
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

#train the model and after every epoch save the model that gives us a validation accuracy >=60%
def fit(epochs, lr, model, train_loader, val_loader, opt_func):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):

        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
        if result['val_acc'] >= 0.60:
            # To save a model:
            torch.save(model.state_dict(),"./demo_model.sav")
            return history

    print("desired accuracy wasnt obtained ")
    return history


#used to print predicted labels and evaluation such as confusion matrix and f1-score
def test(model, test_dl):
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
    for c in range(4):  # calculating the above values from the cf_matrix
        tp += cf_matrix[c][c]
        for r in range(4):
            if r != c:
                fp += cf_matrix[r][c]
        for i in range(4):
            if i != c:
                fn += cf_matrix[c][i]

    for c in range(4):  # calculating the above values from the cf_matrix
        fn = 0  # False Negative
        fp = 0  # False Positive
        tp = 0  # True Positive

        tp += cf_matrix[c][c]
        for r in range(4):
            if r != c:
                fp += cf_matrix[r][c]
        for i in range(4):
            if i != c:
                fn += cf_matrix[c][i]

        tn = len(list_labels) - fn - fp - tp
        print('\nCLASS #', str(c))

        accuracy = (tp + tn)/ (tp + tn + fp + fn)
        format_float = "{:.2f}".format(accuracy)
        print('Accuracy: ', format_float)
        precision = tp / (tp+fp)
        format_float = "{:.2f}".format(precision)
        print('Precision: ', format_float)
        recall = tp / (tp+fn)
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

if __name__ == "__main__":
    # train and test directory
    data_dir = "./dataset"
    #sample directory
    sample_dir = "./sample-dataset"

    request = input('Do you want to train the base model? (yes/no) : ')

    if request == 'yes':
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

        train_data, val_data = random_split(train_data, [train_size - val_size, val_size])
        print(f"Length of Train Data : {len(train_data)}")
        print(f"Length of Validation Data : {len(val_data)}")

        # load the train,validation, and test into batches.
        train_dl = DataLoader(train_data, batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_dl = DataLoader(val_data, batch_size, num_workers=4, pin_memory=True)
        test_dl = DataLoader(test_data, batch_size * 2, num_workers=4, pin_memory=True)

        model = FaceMaskClassification()

        num_epochs = 30
        opt_func = torch.optim.Adam
        lr = 0.001

        # train the model
        #history = fit(num_epochs, lr, model, train_dl, val_dl, opt_func)

        # To restore a model:
        model = FaceMaskClassification()
        model.load_state_dict(torch.load("./finalized_model.sav"), strict=False)

        # run the model on test
        print('\nEvaluation:')
        test(model, test_dl)

    elif request == "no":
        # prepare sample dataset
        sample_dataset = ImageFolder(sample_dir, transform=transforms.Compose([
            transforms.Resize((150, 150)), transforms.ToTensor()
        ]))

        sample_dl = DataLoader(sample_dataset, len(sample_dataset), num_workers=4, pin_memory=True)

        # To restore a model:
        model = FaceMaskClassification()
        model.load_state_dict(torch.load("./finalized_model.sav"),
                              strict=False)

        # run the model on sample
        test(model, sample_dl)
