import pandas as pd
import numpy as np

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
            curr_diff = abs(1 - abs(f[1][0] / f[1][1]))
            min_diff.append((i, curr_diff))

        # The most optimal cut-off value is the first element of the sorted list
        min_diff = sorted(min_diff, key=lambda x: x[1])
        cut_off = min_diff[0][0]

        labels = [0 if i <= cut_off else 1 for i in self.new_data[:, 5]]

        self.features = self.new_data[:, :5]
        self.labels = labels


    def __len__(self):
        # Return the length of the dataset - should be the same as the number of rows / labels
        return len(self.features)

    # Fill this member so we can get data by index
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


BUPA()
