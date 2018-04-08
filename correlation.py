# CSC 240 Data Mining Final Project
# Spring 2018
# Jack Dalton, Mcvvina Lin
# data cleaning - removing variables using Pearson's correlation coefficient

# import numpy

filename = "train.csv"
numlines = 0

file = open(filename,"r")
headers = file.readline().strip("\n").split(",")  # get list of variable names from first line
file.close()

data = {}  # key = variable name, value = list of data
# inconsistent = set()  # set of indices of inconsistent data
with open(filename) as f:
    next(f)  # skip first line
    for line in f:  # iterate through each line in file
        values = line.split(",")  # list of attribute values in each line
        index = 0
        for value in values:  # skip non-integer values
            header = str(headers[index])
            
        #     while True:
        #       num = -1  # initialize
        #         header = str(headers[index])
        #         try:  # integer value
        #             num = int(value)  # throws exception if not integer
        #
        #             data.setdefault(header,[]).append(num)
        #             break
        #         except:  # non-integer value
        #             if header in data:
        #                 data[header].append(str(-1))
        #                 inconsistent.add(index)
        #             break
            index = index + 1
        numlines = numlines + 1

# remove inconsistent data
# for i in inconsistent:
#     for var in data:
#         del data[var][i]

for i in data.items():
    print(str(i[0]), str(len(i[1])))
    # if len(i[1]) < numlines:


# numpy.corrcoeff()