import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from networks.ann import ANN

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import cross_val_score

# From the source web site, it has been stated that there are duplicates so we get rid of them here
# Thanks to Leon for mentioning that there are duplicates in this data set.
# --UCI ML Librarian

# Idx starts from 1

# row 84 and 86:   94,58,21,18,26,2.0,2
# row 141 and 318:   92,80,10,26,20,6.0,1
# row 143 and 150:   91,63,25,26,15,6.0,1
# row 170 and 176:   97,71,29,22,52,8.0,1

# We drop the misleading 7th row while we're cleaning the data

# Suggestions for navigating and analysing the BUPA dataset outlined in the paper below:
# Diagnosing a disorder in a classification benchmark
# January 2016Pattern Recognition Letters 73
# DOI: 10.1016/j.patrec.2016.01.004
# Authors:
# James Mcdermott
# University of Galway
# R. S. Forsyth


# Let the ann script cast this to a tensor - purify class functionality
class BUPA:
    def __init__(self):
        self.raw_data = pd.read_csv("dataset/bupa.data", header=None).values
        clones = [85, 317, 149, 175]

        # Drop the repetitive values and the 7th column
        self.new_data = []
        for i in range(len(self.raw_data)):
            if i not in clones:
                self.new_data.append(self.raw_data[i][:6])
        self.new_data = np.array(self.new_data)

        # The closer the value of the quotient between the number of elements per class is to 1 for some cut-off value, the better the separation
        # If the no. of elements in each class is about equal then the quotient of those 2 numbers is about 1
        # If we subtract that number from 1, the closer the result should be to 0
        # The code below tries to find the most optimal cut-off value for the current dataset

        min_diff = []
        for i in range(1, 20):
            # Find how many elements are in each class for some cut-off value
            labels = [0 if j <= i else 1 for j in self.new_data[:, 5]]
            f = np.unique(labels, return_counts=True)
            # Find the quotient's distance from 0
            curr_diff = abs(1 - f[1][0] / f[1][1])
            min_diff.append((i, curr_diff))

        # The most optimal cut-off value is the first element of the sorted list
        min_diff = sorted(min_diff, key=lambda x: x[1])
        cut_off = min_diff[0][0]

        labels = [0 if i <= cut_off else 1 for i in self.new_data[:, 5]]

        self.features = self.new_data[:, :5]
        self.labels = labels

        # Combine features and labels so we can apply kfold easily
        self.bulk = np.column_stack((self.features, self.labels))

    def __len__(self):
        # Return the length of the dataset - should be the same as the number of rows / labels
        return len(self.features)

    # Fill this member so we can get data by index
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-use-k-fold-cross-validation-with-pytorch.md
class ANNBUPA:
    def __init__(self, dataset, normalise=False):
        # The line below should be equal to self.bulk from the bupa dataset
        self.dataset = dataset

        self.features = normalize(dataset[..., :5]) if normalise else dataset[..., :5]
        self.labels = dataset[..., 5]

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Specify inputs for the neural net
        self.learning_rate = 0.001
        self.epochs = 100
        self.batch_size = 10
        self.loss = torch.nn.CrossEntropyLoss()
        self.activation = torch.nn.ReLU()
        self.io = [5, 2]
        self.hidden = [500, 250, 125, 62, 31, 15, 7]

        self.model = ANN(self.io[0], self.hidden, self.io[1], self.activation).to(
            self.device
        )

        self.main()

    def main(self):
        def reset_weights(m):
            """
            Try resetting model weights to avoid
            weight leakage.
            """
            for layer in m.children():
                if hasattr(layer, "reset_parameters"):
                    print(f"Reset trainable parameters of layer = {layer}")
                    layer.reset_parameters()

        # Define the K-fold Cross Validator
        kfold = KFold(n_splits=10, shuffle=True)
        results = {}

        tts_feat, tts_labels = (
            torch.Tensor(self.features).to(self.device),
            torch.Tensor(self.labels).long().to(self.device),
        )
        tensor_ds = TensorDataset(tts_feat, tts_labels)

        # Prepare for data logging
        training_losses = []
        training_accuracies = []

        testing_losses = []
        testing_accuracies = []

        # accuracy_scores = cross_val_score(
        #     self.model, self.features, self.labels, cv=10, scoring="accuracy"
        # )
        # precision_macro_scores = cross_val_score(
        #     self.model, self.features, self.labels, cv=10, scoring="precision_macro"
        # )

        # precision_micro_scores = cross_val_score(
        #     self.model, self.features, self.labels, cv=10, scoring="precision_micro"
        # )

        # recall_macro_scores = cross_val_score(
        #     self.model, self.features, self.labels, cv=10, scoring="recall_macro"
        # )
        # recall_micro_scores = cross_val_score(
        #     self.model, self.features, self.labels, cv=10, scoring="recall_micro"
        # )

        # print("Individual scores per fold:")

        # for fold, scores in enumerate(
        #     zip(
        #         accuracy_scores,
        #         precision_macro_scores,
        #         precision_micro_scores,
        #         recall_macro_scores,
        #         recall_micro_scores,
        #     ),
        #     start=1,
        # ):
        #     print(
        #         fold,
        #         """Accuracy: %0.2f,
        #         Precision (macro): %0.2f, Precision (micro): %0.2f,
        #            Recall (macro): %0.2f,    Recall (micro): %0.2f"""
        #         % scores,
        #     )

        # print(
        #     "\nAverage accuracy of %0.2f with a standard deviation of %0.2f"
        #     % (accuracy_scores.mean(), accuracy_scores.std())
        # )
        # print(
        #     "\nAverage precision macro of %0.2f with a standard deviation of %0.2f"
        #     % (precision_macro_scores.mean(), precision_macro_scores.std())
        # )
        # print(
        #     "\nAverage precision micro of %0.2f with a standard deviation of %0.2f"
        #     % (precision_micro_scores.mean(), precision_micro_scores.std())
        # )
        # print(
        #     "\nAverage recall macro of %0.2f with a standard deviation of %0.2f"
        #     % (recall_macro_scores.mean(), recall_macro_scores.std())
        # )
        # print(
        #     "\nAverage recall micro of %0.2f with a standard deviation of %0.2f"
        #     % (recall_micro_scores.mean(), recall_micro_scores.std())
        # )

        # K-fold Cross Validation model evaluation
        for fold, (train_ids, test_ids) in enumerate(kfold.split(self.dataset)):

            # Print
            print(f"FOLD {fold}")

            # Sample elements randomly from a given list of ids, no replacement.
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

            # Define data loaders for training and testing data in this fold
            trainloader = torch.utils.data.DataLoader(
                tensor_ds, batch_size=10, sampler=train_subsampler
            )
            testloader = torch.utils.data.DataLoader(
                tensor_ds, batch_size=10, sampler=test_subsampler
            )

            # Avoid weight leakage
            self.model.apply(reset_weights)

            # Initialise optimiser for every fold
            self.optimiser = torch.optim.Adam(self.model.parameters(), lr=0.01)

            # Run the training loop for defined number of epochs
            for epoch in range(self.epochs):

                # Print epoch
                print(f"Starting epoch {epoch+1}")

                # Set current loss value
                current_loss = 0.0
                total = 0.0
                correct = 0.0

                # Iterate over the DataLoader for training data
                for i, data in enumerate(trainloader, 0):

                    # Get inputs
                    inputs, targets = data

                    # Zero the gradients
                    self.optimiser.zero_grad()

                    # Perform forward pass
                    outputs = self.model(inputs)

                    # Compute loss
                    loss = self.loss(outputs, targets)

                    # Perform backward pass
                    loss.backward()

                    # Perform optimization
                    self.optimiser.step()

                    current_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

                    # Print statistics
                    if i % 500 == 499:
                        print(
                            "Loss after mini-batch %5d: %.3f"
                            % (i + 1, current_loss / 500)
                        )
                        current_loss = 0.0

                training_accuracy = 100.0 * correct / total
                average_loss = current_loss / len(trainloader)

                training_losses.append(average_loss)
                training_accuracies.append(training_accuracy)

            # Process is complete.
            print("Training process has finished. Saving trained model.")

            # Print about testing
            print("Starting testing")

            # Saving the model
            save_path = f"./model-fold-{fold}.pth"
            torch.save(self.model.state_dict(), save_path)

            # Evaluation for this fold
            correct, total = 0, 0
            val_loss = 0
            with torch.no_grad():

                # Iterate over the test data and generate predictions
                for i, data in enumerate(testloader, 0):

                    # Get inputs
                    inputs, targets = data

                    # Generate outputs
                    outputs = self.model(inputs)
                    loss = self.loss(outputs, targets)
                    val_loss += loss.item()
                    # Set total and correct
                    _, predicted = torch.max(outputs.data, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()

                validation_accuracy = 100 * correct / total
                avg_val_loss = val_loss / len(testloader)

                testing_losses.append(avg_val_loss)
                testing_accuracies.append(validation_accuracy)

                # Print accuracy
                print("Accuracy for fold %d: %d %%" % (fold, validation_accuracy))

                print("--------------------------------")
                results[fold] = 100.0 * (correct / total)

        # Print fold results
        print(f"K-FOLD CROSS VALIDATION RESULTS FOR {10} FOLDS")
        print("--------------------------------")
        sum = 0.0
        for key, value in results.items():
            print(f"Fold {key}: {value} %")
            sum += value
        print(f"Average: {sum/len(results.items())} %")

        # Plot history
        plt.figure()
        plt.title("Average training loss per epoch per fold")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.plot(training_losses, "r")
        # plt.plot(testing_losses, "g")
        plt.legend(["Train loss"])

        plt.figure()
        plt.title("Average testing accuracy per epoch per fold")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.plot(training_accuracies, "b")
        # plt.plot(testing_accuracies, "y")
        plt.legend(["Train accuracy"])

        plt.show()


# Reference https://scikit-learn.org/stable/modules/cross_validation.html
class SVMBUPA:
    def __init__(self, dataset, auto_grid=False, normalise=False):
        self.dataset = dataset

        self.features = normalize(dataset[..., :5]) if normalise else dataset[..., :5]
        self.labels = dataset[..., 5]

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.features, self.labels, test_size=0.3, shuffle=True
        )

        model = SVC(kernel="linear", C=1, gamma=0.0, random_state=42)

        if auto_grid:
            # Create parameters
            param_grid = {
                "C": [0.1, 1, 10, 100, 1000],
                "gamma": [1, 0.1, 0.01, 0.001, 0.0001],
                "kernel": ["rbf", "sigmoid"],
            }

            # Create grid search to find best parameters and show the best parameters
            model = GridSearchCV(
                SVC(probability=True), param_grid, refit=True, verbose=3
            )

        accuracy_scores = cross_val_score(
            model, self.features, self.labels, cv=10, scoring="accuracy"
        )
        precision_macro_scores = cross_val_score(
            model, self.features, self.labels, cv=10, scoring="precision_macro"
        )

        precision_micro_scores = cross_val_score(
            model, self.features, self.labels, cv=10, scoring="precision_micro"
        )

        recall_macro_scores = cross_val_score(
            model, self.features, self.labels, cv=10, scoring="recall_macro"
        )
        recall_micro_scores = cross_val_score(
            model, self.features, self.labels, cv=10, scoring="recall_micro"
        )

        print("Individual scores per fold:")

        for fold, scores in enumerate(
            zip(
                accuracy_scores,
                precision_macro_scores,
                precision_micro_scores,
                recall_macro_scores,
                recall_micro_scores,
            ),
            start=1,
        ):
            print(
                fold,
                """Accuracy: %0.2f, 
                Precision (macro): %0.2f, Precision (micro): %0.2f, 
                   Recall (macro): %0.2f,    Recall (micro): %0.2f"""
                % scores,
            )

        print(
            "\nAverage accuracy of %0.2f with a standard deviation of %0.2f"
            % (accuracy_scores.mean(), accuracy_scores.std())
        )
        print(
            "\nAverage precision macro of %0.2f with a standard deviation of %0.2f"
            % (precision_macro_scores.mean(), precision_macro_scores.std())
        )
        print(
            "\nAverage precision micro of %0.2f with a standard deviation of %0.2f"
            % (precision_micro_scores.mean(), precision_micro_scores.std())
        )
        print(
            "\nAverage recall macro of %0.2f with a standard deviation of %0.2f"
            % (recall_macro_scores.mean(), recall_macro_scores.std())
        )
        print(
            "\nAverage recall micro of %0.2f with a standard deviation of %0.2f"
            % (recall_micro_scores.mean(), recall_micro_scores.std())
        )

        if auto_grid:
            model.fit(self.x_train, self.y_train)
            print("\nBest parameters: {}".format(model.best_params_))


if __name__ == "__main__":
    bupa_dataset = BUPA()
    # bupa_svm = SVMBUPA(bupa_dataset.bulk, auto_grid=True, normalise=False)
    bupa_ann = ANNBUPA(bupa_dataset.bulk, normalise=True)
