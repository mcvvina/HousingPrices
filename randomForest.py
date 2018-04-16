import frequencies
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import pandas
import normalize
from pandas.plotting import scatter_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import correlation
import sys

##Some code and conceptual help found at https://towardsdatascience.com/random-forest-in-python-24d0893d51c0




### Organizes Data

#print('jer')
freq = frequencies.frequencies()
normData = normalize.normalize(freq.attributeData, freq.allPossData)

names = pandas.Series(data = freq.attributes)
dataOrig = pandas.DataFrame(data = normData)

#print(dataOrig.head())
dataOrig = dataOrig.transpose()
dataOrig.columns = (names)


testFreq = frequencies.frequencies(filename='https://raw.githubusercontent.com/mcvvina/HousingPrices/master/test.csv')
normTestData = normalize.normalize(testFreq.attributeData, testFreq.allPossData)

namesTest = pandas.Series(data = testFreq.attributes)
testIds = pandas.DataFrame(data = testFreq.attributeData)
testIds = testIds.transpose()
testIds.columns = namesTest
testIds = testIds['Id']
##print(testIds.head())
##sys.exit()

namesTest = pandas.Series(data = testFreq.attributes)

dataOrigTest = pandas.DataFrame(data = normTestData)
dataOrigTest = dataOrigTest.transpose()
dataOrigTest.columns = (namesTest)

dataTestId = np.array(testIds)
##print(dataTestId)
##sys.exit()
dataTest = np.array(dataOrigTest)


##print (data.head())
##print(len(data))

labels = pandas.DataFrame(freq.attributeData)
labels = labels.transpose()
labels.columns = names
labels = np.array(labels['SalePrice'])

dataOrig = dataOrig.drop('SalePrice', axis = 1)

data_list = list(dataOrig.columns)

data = np.array(dataOrig)
#print('ehe')


### Regression using all normalized Data

# Break the data into training and test (we do this so we can see if the regression is accurate)
#train_features, test_features, train_labels, test_labels = train_test_split(data, labels, test_size = 0.25, random_state = 42)

rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)

rf.fit(data, labels)

# Use the forest's predict method on the test data
predictions = rf.predict(dataTest)
predictionsTrain = rf.predict(data)





# Calculate the absolute errors
errors = abs(predictionsTrain - labels)
### Print out the mean absolute error (mae)
###print('Mean Absolute Error:', round(np.mean(errors), 2), 'dollars.')


# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / labels)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')








######## Reduced accuracy so commenting out for now
##
##
##### Regression using the correlated data
##
##
##corr = correlation.correlat(freq, .6)
##corrAttributeNames = [ freq.attributes[each[0]] for each in corr]
##del corrAttributeNames[len(corrAttributeNames)-1] #removes salesPrice
##corrAttributeNames = pandas.Series(data = corrAttributeNames)
##
##d2 = dataOrig.loc[:,corrAttributeNames]
##
##
### Break the data into training and test (we do this so we can see if the regression is accurate)
##train_features, test_features, train_labels, test_labels = train_test_split(d2, labels, test_size = 0.25, random_state = 42)
##
##rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
##
##rf.fit(d2, labels)
##
### Use the forest's predict method on the test data
##predictions = rf.predict(test_features)
### Calculate the absolute errors
##errors = abs(predictions - test_labels)
##### Print out the mean absolute error (mae)
#####print('Mean Absolute Error:', round(np.mean(errors), 2), 'dollars.')
##
##
### Calculate mean absolute percentage error (MAPE)
##mape = 100 * (errors / test_labels)
### Calculate and display accuracy
##accuracy = 100 - np.mean(mape)
##print('Accuracy:', round(accuracy, 2), '%.')




importances = list(rf.feature_importances_)



#### 11 important attibutes
##
### list of x locations for plotting
##x_values = list(range(len(importances)))
### Make a bar chart
##plt.bar(x_values, importances, orientation = 'vertical')
### Tick labels for x axis
##plt.xticks(x_values, data_list, rotation='vertical')
### Axis labels and title
##plt.ylabel('Importance');
##plt.xlabel('Variable');
##plt.title('Variable Importances');
##
##plt.show()

####### List of tuples with variable and importance

data_importances = [(data, round(importance, 2)) for data, importance in zip(data_list, importances)]
# Sort the feature importances by most important first
data_importances = sorted(data_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
##[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in data_importances];

########   OUTPUT
##Variable: OverallQual          Importance: 0.58
##Variable: GrLivArea            Importance: 0.11
##Variable: TotalBsmtSF          Importance: 0.04
##Variable: 2ndFlrSF             Importance: 0.04
##Variable: BsmtFinSF1           Importance: 0.03
##Variable: 1stFlrSF             Importance: 0.02
##Variable: GarageCars           Importance: 0.02
##Variable: LotArea              Importance: 0.01
##Variable: Neighborhood         Importance: 0.01
##Variable: YearBuilt            Importance: 0.01
##Variable: YearRemodAdd         Importance: 0.01
##Variable: BsmtQual             Importance: 0.01
##Variable: BsmtUnfSF            Importance: 0.01
##Variable: FullBath             Importance: 0.01
##Variable: TotRmsAbvGrd         Importance: 0.01
##Variable: GarageArea           Importance: 0.01
##Variable: OpenPorchSF          Importance: 0.01



#### Lowers accuracy again

##rf_most_important = RandomForestRegressor(n_estimators= 1000, random_state=42)
##
##importance=[i for i in range(11)]
##importance = np.array(importance)
##print(importance)
##
##train_important = train_features[:, importance]
##test_important = test_features[:, importance]
### Train the random forest
##rf_most_important.fit(train_important, train_labels)
##predictions = rf_most_important.predict(test_important)
##errors = abs(predictions - test_labels)
### Display the performance metrics
##print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
##mape = np.mean(100 * (errors / test_labels))
##accuracy = 100 - mape
##print('Accuracy:', round(accuracy, 2), '%.')

##print(labels.size())






#print(labels.size)
x1 = [i for i in range(labels.size)]
plt.scatter(x1, labels,color='b')
plt.scatter(x1, predictionsTrain, color = 'r')
plt.title('Prediction Vs Actual')
plt.xlabel('house')
plt.ylabel('price')
plt.show()

##y = labels
predictions_data = pandas.DataFrame(data = {'prediction':predictions})
x = [i for i in range(predictions_data.size)]
#print (predictions_data)
##plt.scatter(x,y, color= 'b')
plt.scatter(x, predictions_data, color = 'c')
plt.title('Prediction')
plt.xlabel('house')
plt.ylabel('price')
plt.show()


file = open('submission.txt','a')
file.write("Id,SalePrice\n")


printPred = np.array(predictions_data)
printId = np.array(dataTestId)
for i in range(dataTestId.size):
    file.write(str(printId[i])+","+str(printPred[i][0])+"\n")

file.close()
