import matplotlib.pyplot as plt
import pandas
import copy

def _ro(x_data,y_data):
    x_data = x_data.copy()
    outliers = set()
    temps = list(x_data)
    LineData = plt.boxplot(x_data)
##    plt.show()
    outlierYVal = [each.get_ydata() for each in LineData.get('fliers')]
    for each in outlierYVal[0]:        
        outliers.add(temps.index(each))    
    return outliers

def removeOutlier(x_data,y_data):
    x_data = x_data.copy()
    x_data = pandas.DataFrame(data = x_data)
    outliers = set()
    for i in range(x_data.shape[0]):
        outliers = outliers.union(_ro(x_data.iloc[i,:],y_data))

    return sorted(list(outliers))



def returnWithoutOutliers(x_data, y_data):
    x_data = copy.deepcopy(x_data)
    y_data = copy.deepcopy(y_data)
    #print(np.array(normData).shape)
    indicesToDrop = removeOutlier(list(x_data),y_data)
    indicesToDrop = sorted(indicesToDrop, reverse = True)
    indicesToDrop = list(indicesToDrop)
    for eachAttribute in x_data:
        for e in indicesToDrop:
            del eachAttribute[e]
    for e in indicesToDrop:
        del y_data[e]
    #print(np.array(normData).shape)
    plt.clf()

##    print( str(len(x_data)) + " " + str(len(y_data)))
    return [x_data, y_data]
### PLot regression
##plot Residuals
##remove those above certain threshold
