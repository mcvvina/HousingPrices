### help with using tkinter backend  obtained from https://pythonprogramming.net/how-to-embed-matplotlib-graph-tkinter-gui/
# The code for changing pages was derived from: http://stackoverflow.com/questions/7546050/switch-between-two-frames-in-tkinter
# License: http://creativecommons.org/licenses/by-sa/3.0/	

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
import math
import tkinter as tk
from tkinter import ttk
from tkinter import Entry
import plotter as hist
import frequencies
import normalize
import pandas
import correlation
import sys
import RandomForest as rf
import removeOutlier as ro
import copy
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostRegressor
from sklearn import linear_model
from scipy.stats import skew
from sklearn.tree import ExtraTreeRegressor
import matplotlib.pyplot as plt
from scipy.stats import boxcox
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
# import normalizeBefore

#
######## Training Data ##########
#
freq = frequencies.frequencies()

xTrainData = np.array(copy.deepcopy(freq.attributeData))


normData =normalize.normalize(freq.attributeData, freq.allPossData)

##print(normData.transpose().head)
##normData = pandas.DataFrame(data = normalizeBefore.normalize(freq.attributeData, freq.allPossData))
##print(normData.transpose().head)
##
##sys.exit()



names = pandas.Series(data = freq.attributes)

dataOrig = pandas.DataFrame(data = normData)
dataOrig = dataOrig.transpose()
dataOrig.columns = (names)

skewed = copy.deepcopy(dataOrig)
skewed = skewed.drop("SalePrice",1)
skewed = skewed.drop("Id",1)

for i in range(skewed.shape[1]):
    skewed.iloc[:,i] = skew(skewed.iloc[:,i])
skewed = skewed.drop_duplicates()
skewed = skewed.transpose()
skewed.columns = np.array(["Skew"])

skewed = skewed.sort_values(['Skew'], ascending = False)
toPrintSkew = skewed.copy()

toPrintSkew = toPrintSkew[abs(toPrintSkew.values)> 0.75]

#### Fix Skew
for atribu in toPrintSkew.index:
##    print(atribu)
    dataOrig[atribu] = dataOrig[atribu].add(2)
    dataOrig[atribu] = boxcox( dataOrig[atribu])[0]
    maxxxx = max(dataOrig[atribu])
    minnnn = min(dataOrig[atribu])
    dataOrig[atribu] = dataOrig[atribu].apply(lambda x: ((x -minnnn)/(maxxxx-minnnn))*(1+1)-1)
 

##    plt.hist(dataOrig[atribu])
##    plt.show()







target = pandas.DataFrame(data = freq.attributeData)
target = target.transpose()
target.columns = names
target = target['SalePrice']
beforetarget = target

      
       
target  = [math.log(i) for i in target] 


 ### Import Test Data

testFreq = frequencies.frequencies(filename='https://raw.githubusercontent.com/mcvvina/HousingPrices/master/test.csv')
normTestData = normalize.normalize(testFreq.attributeData, testFreq.allPossData)

#### Import data as Panda DataFrame
namesTest = pandas.Series(data = testFreq.attributes)
testData = pandas.DataFrame(data = normTestData)
testData = testData.transpose()
testData.columns = namesTest


### Extract Ids
testIds = pandas.DataFrame(data = testFreq.attributeData)
testIds = testIds.transpose()
testIds.columns = namesTest
testIds = testIds['Id']













LARGE_FONT= ("Verdana", 12)


class HousePrices(tk.Tk):

    def __init__(self, *args, **kwargs):
        
        tk.Tk.__init__(self, *args, **kwargs)

        tk.Tk.wm_title(self, "House Prices")
        
        
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand = True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        for F in (StartPage, Histograms, NormScatter,NormScatterNoOutlier,
                  NormImportantScatter,NormImportantOutlierScatter,
                  SkewVisualization,PriceVisualization
                  ,RandomForestRegressorWithOutliers,RandomForestRegressorNoOutliers,
                  RandomForestClassifierWithOutliers, RandomForestClassifierNoOutliers,
                  AdaBoostRegressorWithOutliers, AdaBoostNoOutliers, LinearModelWithOutliers,
                  LinearModelNoOutliers, RidgeWithOutliers,RidgeNoOutliers,
                  LassoWithOutliers, LassoNoOutliers,RANSACRegressorWithOutliers,
                  RANSACRegressorNoOutliers,
                  CombineRegressors):

            frame = F(container, self)

            self.frames[F] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)

    def show_frame(self, cont):

        frame = self.frames[cont]
        frame.tkraise()

        
class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self,parent)
        label = tk.Label(self, text="Welcome!", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        label = tk.Label(self, text="DataCleaning and Visualization", font=('Helvetica', 12))
        label.pack(pady=10,padx=10)
        
        button = ttk.Button(self, text="View Histograms of Attributes",
                            command=lambda: controller.show_frame(Histograms))
        button.pack()

        button2 = ttk.Button(self, text="Normalized Data Scatter Plots",
                            command=lambda: controller.show_frame(NormScatter))
        button2.pack()

        button3 = ttk.Button(self, text="Normalized Data Scatter Plots NO outliers",
                            command=lambda: controller.show_frame(NormScatterNoOutlier))
        button3.pack()

        
        button4 = ttk.Button(self, text="Normalized Important Data Plots w/outliers",
                            command=lambda: controller.show_frame(NormImportantScatter))
        button4.pack()

        button5 = ttk.Button(self, text="Normalized Important Data Plots NO outliers",
                            command=lambda: controller.show_frame(NormImportantOutlierScatter))
        button5.pack()
        button6 = ttk.Button(self, text="Skew Visualization",
                            command=lambda: controller.show_frame(SkewVisualization))
        button6.pack()


        button7 = ttk.Button(self, text="Price Visualization",
                            command=lambda: controller.show_frame(PriceVisualization))
        button7.pack()
        
        
        label = tk.Label(self, text="Techniques Used", font=('Helvetica', 12))
        label.pack(pady=10,padx=10)

        button8 = ttk.Button(self, text="RandomForestRegressor Output With Outliers",
                            command=lambda: controller.show_frame(RandomForestRegressorWithOutliers))
        button8.pack()
        
        button9 = ttk.Button(self, text="RandomForestRegressor Output No Outliers",
                            command=lambda: controller.show_frame(RandomForestRegressorNoOutliers))
        button9.pack()

        button10 = ttk.Button(self, text="RandomForestClassifier Output With Outliers",
                            command=lambda: controller.show_frame(RandomForestClassifierWithOutliers))
        button10.pack()

        button11 = ttk.Button(self, text="RandomForestClassifier Output No Outliers",
                            command=lambda: controller.show_frame(RandomForestClassifierNoOutliers))
        button11.pack()
        
        button12 = ttk.Button(self, text="AdaBoostRegressor Output With Outliers",
                            command=lambda: controller.show_frame(AdaBoostRegressorWithOutliers))
        button12.pack()

        button13 = ttk.Button(self, text="AdaBoost Output No Outliers",
                            command=lambda: controller.show_frame(AdaBoostNoOutliers))
        button13.pack()

        button14 = ttk.Button(self, text="LinearModel Output With Outliers",
                            command=lambda: controller.show_frame(LinearModelWithOutliers))
        button14.pack()

        button15 = ttk.Button(self, text="LinearModel Output No Outliers",
                            command=lambda: controller.show_frame(LinearModelNoOutliers))
        button15.pack()

        button16 = ttk.Button(self, text="Ridge Output With Outliers",
                            command=lambda: controller.show_frame(RidgeWithOutliers))
        button16.pack()

        button17 = ttk.Button(self, text="Ridge Output No Outliers",
                            command=lambda: controller.show_frame(RidgeNoOutliers))
        button17.pack()

        button18 = ttk.Button(self, text="Lasso Output With Outliers",
                            command=lambda: controller.show_frame(LassoWithOutliers))
        button18.pack()

        button19 = ttk.Button(self, text="Lasso Output No Outliers",
                            command=lambda: controller.show_frame(LassoNoOutliers))
        button19.pack()

        button20 = ttk.Button(self, text="RANSAC Output With Outliers",
                            command=lambda: controller.show_frame(RANSACRegressorWithOutliers))
        button20.pack()

        button21 = ttk.Button(self, text="RANSAC Output No Outliers",
                            command=lambda: controller.show_frame(RANSACRegressorNoOutliers))
        button21.pack()

        button22 = ttk.Button(self, text="Combining Regressors",
                            command=lambda: controller.show_frame(CombineRegressors))
        button22.pack()
        
        


        

        

class Histograms(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Histograms (click figure to change histogram)", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        button1 = ttk.Button(self, text="Back to Home",
                            command=lambda: controller.show_frame(StartPage))
        button1.pack()

        f = Figure(figsize=(5,5), dpi=100)

        canvas = FigureCanvasTkAgg(f, self)
        f.canvas = canvas
        hist.plotter(freq, fig = f, backend = tk)


class NormScatter(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Normalized Data Scatter", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        button1 = ttk.Button(self, text="Back to Home",
                            command=lambda: controller.show_frame(StartPage))
        button1.pack()


        f = Figure(figsize=(5,5), dpi=100)

        canvas = FigureCanvasTkAgg(f, self)
        f.canvas = canvas
        hist.plotter(freq, fig = f, backend = tk, typ = 'plot')

class NormScatterNoOutlier(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="All Attributes NO outliers", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        button1 = ttk.Button(self, text="Back to Home",
                            command=lambda: controller.show_frame(StartPage))
        button1.pack()


        f = Figure(figsize=(5,5), dpi=100)

        canvas = FigureCanvasTkAgg(f, self)
        f.canvas = canvas
        hist.plotter(freq, fig = f,  backend = tk, typ = 'plot', removeOutliers = True)

    
class NormImportantScatter(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Important Attributes w/outliers", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        button1 = ttk.Button(self, text="Back to Home",
                            command=lambda: controller.show_frame(StartPage))
        button1.pack()


        f = Figure(figsize=(5,5), dpi=100)

        canvas = FigureCanvasTkAgg(f, self)
        f.canvas = canvas

        
        corrData = correlation.correlat(freq, .6,  normalizedData = normData)
        corrData = pandas.DataFrame(data = corrData)
        
        indsT = names.get(list(corrData.iloc[:,0]))
        indicesToPlot = list(indsT)#['OverallQual',  'GrLivArea', 'TotalBsmtSF','2ndFlrSF',  'BsmtFinSF1', '1stFlrSF', 'GarageCars', 'LotArea', 'Neighborhood', 'YearBuilt','YearRemodAdd','BsmtQual','BsmtUnfSF','FullBath','TotRmsAbvGrd','GarageArea','OpenPorchSF']
        hist.plotter(freq, fig = f, XindicesToPlot = indicesToPlot, backend = tk, typ = 'plot')

class NormImportantOutlierScatter(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Important Attributes without outliers", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        button1 = ttk.Button(self, text="Back to Home",
                            command=lambda: controller.show_frame(StartPage))
        button1.pack()


        f = Figure(figsize=(5,5), dpi=100)
        canvas = FigureCanvasTkAgg(f, self)
        f.canvas = canvas
        
        corrData = correlation.correlat(freq, .6,removeOutliers  = True,  normalizedData = normData)
        corrData = pandas.DataFrame(data = corrData)
        
        indsT = names.get(list(corrData.iloc[:,0]))
        XindicesToPlot = list(indsT)#['OverallQual',  'GrLivArea', 'TotalBsmtSF','2ndFlrSF',  'BsmtFinSF1', '1stFlrSF', 'GarageCars', 'LotArea', 'Neighborhood', 'YearBuilt','YearRemodAdd','BsmtQual','BsmtUnfSF','FullBath','TotRmsAbvGrd','GarageArea','OpenPorchSF']
        
        hist.plotter(freq, fig = f, XindicesToPlot = XindicesToPlot,  backend = tk, typ = 'plot', removeOutliers = True)



class PriceVisualization(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="SalesPrice Skew Visualization", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        button1 = ttk.Button(self, text="Back to Home",
                            command=lambda: controller.show_frame(StartPage))
        button1.pack()
        
        f  = Figure(figsize=(5,5), dpi=100)
        ax1 = f.add_subplot(211)
        ax2 = f.add_subplot(212)
        toHist = np.array(copy.deepcopy(beforetarget))
        logSales = [math.log(i) for i in toHist]
        ax1.hist(list(toHist))
        ax1.set_xlabel("SalePrice")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Skewed")
        ax2.hist(list(logSales))
        ax2.set_xlabel("log(SalePrice)")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Gaussian")
    
       

        canvas = FigureCanvasTkAgg(f, self)
        canvas.draw()

        
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

##        toolbar = NavigationToolbar2TkAgg(canvas, self)
##        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
class SkewVisualization(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Skewed Variables", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        button1 = ttk.Button(self, text="Back to Home",
                            command=lambda: controller.show_frame(StartPage))
        button1.pack()
        

        
        
        for i in range(toPrintSkew.shape[0]): 
            gridss= tk.Label(self, text= str(list(toPrintSkew.index)[i]) +" "+ str(toPrintSkew.iloc[i,0]), font=('Helvetica', 5))    
            gridss.pack()


        
        

class RandomForestRegressorWithOutliers(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="RandomForestRegressor Output With Outliers", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        button1 = ttk.Button(self, text="Back to Home",
                            command=lambda: controller.show_frame(StartPage))
        button1.pack()
        norData = copy.deepcopy(normData)
##        norData = ro.returnWithoutOutliers(normData, target)
##        print(pandas.DataFrame(data = norData).shape)
##        corrData = correlation.correlat(freq, .6,  normalizedData = norData)
##        corrData = pandas.DataFrame(data = corrData)  
##        indsT = names.get(list(corrData.iloc[:,0]))
##        importantIndices = list(indsT)
        y = target#norData[len(norData)-1]
        x = norData
        del x[len(norData)-1]
        del x[0]
        x = np.array(x).transpose()

        
        X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.5)


        rff = rf.RandomForestRegressor(n_estimators = 1000, random_state = 42,oob_score =True, max_features = "auto", warm_start=True)

        rff.fit(X_train, y_train)
        label1 = ttk.Label(self,text = "Train Score, iter 1: " +str(rff.score(X_train,y_train)))
        label1.pack()

        
        predictions = rff.predict(X_test)
        
        label2 = ttk.Label(self,text = "Test Score, iter 1: " +str(rff.score(X_test,y_test)))
        label2.pack()

        rff.n_estimators = 2000
        rff.fit(X_train, y_train)
        label3 = ttk.Label(self,text = "Train Score, iter 2: " +str(rff.score(X_train,y_train)))
        label3.pack()
        
        rff.n_estimators = 3000
        predictions = rff.predict(X_test)
        
        label2 = ttk.Label(self,text = "Test Score, iter 2: " +str(rff.score(X_test,y_test)))
        label2.pack()

        label3 = ttk.Label(self,text = "Train Score on all training data: " +str(rff.score(x,y)))
        label3.pack()


        ##### WRITE OUT #####    
        testXData = testData.copy()
        del testXData['Id']

          
        y_prediction=rff.predict(testXData)
        y_prediction = np.exp(y_prediction)

        file = open('RandomForestRegressorWithOutliers.txt','w')
        file.write("Id,SalePrice\n")

        predictions_data = pandas.DataFrame(data = {'prediction':y_prediction})

        print("Printed: RandomForestRegressorWithOutliers")
        
        #print(predictions_data.shape)
        printPred = np.array(predictions_data)
        printId = np.array(testIds)
        for i in range(printId.size):
            file.write(str(printId[i])+","+str(printPred[i][0])+"\n")

        file.close()


        
class RandomForestRegressorNoOutliers(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="RandomForestRegressor Output No Outliers", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        button1 = ttk.Button(self, text="Back to Home",
                            command=lambda: controller.show_frame(StartPage))
        button1.pack()

        wOutOut = ro.returnWithoutOutliers(normData, target)
##        print(pandas.DataFrame(data = norData).shape)
##        corrData = correlation.correlat(freq, .6,  normalizedData = norData)
##        corrData = pandas.DataFrame(data = corrData)  
##        indsT = names.get(list(corrData.iloc[:,0]))
##        importantIndices = list(indsT)
        y = wOutOut[1]
        x = wOutOut[0]
        del x[len(x)-1]
        del x[0]
        x = np.array(x).transpose()

        
        X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.9)


        rff = rf.RandomForestRegressor(n_estimators = 1000, random_state = 42,oob_score =True, max_features = "auto", warm_start=True)

        rff.fit(X_train, y_train)
        label1 = ttk.Label(self,text = "Train Score, iter 1: " +str(rff.score(X_train,y_train)))
        label1.pack()

        
        predictions = rff.predict(X_test)
        
        label2 = ttk.Label(self,text = "Test Score, iter 1: " +str(rff.score(X_test,y_test)))
        label2.pack()

        rff.n_estimators = 2000
        rff.fit(X_train, y_train)
        label3 = ttk.Label(self,text = "Train Score, iter 2: " +str(rff.score(X_train,y_train)))
        label3.pack()
        
        rff.n_estimators = 3000
        predictions = rff.predict(X_test)
        
        label2 = ttk.Label(self,text = "Test Score, iter 2: " +str(rff.score(X_test,y_test)))
        label2.pack()

        label3 = ttk.Label(self,text = "Train Score on all training data: " +str(rff.score(x,y)))
        label3.pack()

##        ##### WRITE OUT #####    
        testXData = testData.copy()
        del testXData['Id']

         
        y_prediction=rff.predict(testXData)
        y_prediction = np.exp(y_prediction)
        
        file = open('RandomForestRegressorNoOutliers.txt','w')
        file.write("Id,SalePrice\n")

        predictions_data = pandas.DataFrame(data = {'prediction':y_prediction})

        print("Printed: RandomForestRegressorNoOutliers")
        
        #print(predictions_data.shape)
        printPred = np.array(predictions_data)
        printId = np.array(testIds)
        for i in range(printId.size):
            file.write(str(printId[i])+","+str(printPred[i][0])+"\n")

        file.close()


class RandomForestClassifierWithOutliers(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="RandomForestClassifier Output With Outliers", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        button1 = ttk.Button(self, text="Back to Home",
                            command=lambda: controller.show_frame(StartPage))
        button1.pack()

        norData = copy.deepcopy(normData)
        #ro.returnWithoutOutliers(normData, target)
##        print(pandas.DataFrame(data = norData).shape)
##        corrData = correlation.correlat(freq, .6,  normalizedData = norData)
##        corrData = pandas.DataFrame(data = corrData)  
##        indsT = names.get(list(corrData.iloc[:,0]))
##        importantIndices = list(indsT)
        y = copy.deepcopy(target)
        x = norData
        del x[len(norData)-1]
        del x[0]
        x = np.array(x).transpose()
        

        minnn = math.log(25000)
        maxxx = math.log(1000000)
        n = 2001
        bins = np.linspace(minnn, maxxx, num = n)
        ycopy = copy.deepcopy(y)
        y = []
        for indxx,each in enumerate(ycopy):
            if each < minnn:
                print("PRICE CANT BE LESS THAN MINNN")
                sys.exit()
            div = (each-minnn)/((maxxx-minnn)/(n-1))
            floored = math.floor(div)
            y.append(int(bins[floored]))


            
        cff = rf.RandomForestClassifier( n_jobs = 3, random_state=0)####################################
        
        X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
        

               
        cff.fit(X_train, y_train)
        label1 = ttk.Label(self,text = "Train Score, iter 1: " +str(cff.score(X_train,y_train)))
        label1.pack()

        
        predictions = cff.predict(X_test)
        label2 = ttk.Label(self,text = "Test Score, iter 1: " +str(cff.score(X_test,y_test)))
        label2.pack()
        
        X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.4)

        cff.n_estimators = 20
        cff.fit(X_train, y_train)
        label3 = ttk.Label(self,text = "Train Score, iter 2: " +str(cff.score(X_train,y_train)))
        label3.pack()
        
        cff.n_estimators = 30
        predictions = cff.predict(X_test)
        
        label2 = ttk.Label(self,text = "Test Score, iter 2: " +str(cff.score(X_test,y_test)))
        label2.pack()

        label3 = ttk.Label(self,text = "Train Score on all training data: " +str(cff.score(x,y)))
        label3.pack()


        

              
        testXData = testData.copy()
        del testXData['Id']


        ##### WRITE OUT #####    
        testXData = testData.copy()
        del testXData['Id']
         
        y_prediction=cff.predict(testXData)
        y_prediction = np.exp(y_prediction)

        file = open('RandomForestClassifierWithOutliers.txt','w')
        file.write("Id,SalePrice\n")

        predictions_data = pandas.DataFrame(data = {'prediction':y_prediction})

        print("Printed: RandomForestClassifierWithOutliers")
        
        #print(predictions_data.shape)
        printPred = np.array(predictions_data)
        printId = np.array(testIds)
        for i in range(printId.size):
            file.write(str(printId[i])+","+str(printPred[i][0])+"\n")

        file.close()

      
        
class RandomForestClassifierNoOutliers(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="RandomForestClassifier Output No Outliers", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        button1 = ttk.Button(self, text="Back to Home",
                            command=lambda: controller.show_frame(StartPage))
        button1.pack()

        trainAllData = copy.deepcopy(normData)
        wOutOut = ro.returnWithoutOutliers(normData, target)
        norData = wOutOut[0]
        y = wOutOut[1]
        x = norData
        del x[len(norData)-1]
        del x[0]
##        print(len(x))
        x = np.array(x).transpose()
##        print(len(y))
        
        minnn = math.log(25000)
        maxxx = math.log(1000000)
        n = 101
        bins = np.linspace(minnn, maxxx, num = n)
        ycopy = copy.deepcopy(y)
        y = []
        for indxx,each in enumerate(ycopy):
            if each < minnn:
                print("PRICE CANT BE LESS THAN MINNN")
                sys.exit()
            div = (each-minnn)/((maxxx-minnn)/(n-1))
            floored = math.floor(div)
##            print(str(each) + " " +str( bins[floored]))
            y.append(int(bins[floored]))
            
##        print('-------------------------------------------------------------------------')
        
        ytrainFinal = copy.deepcopy(target)
        xtrainFinal = trainAllData
        del xtrainFinal[len(trainAllData)-1]
        del xtrainFinal[0]
        xtrainFinal = np.array(xtrainFinal).transpose()

        ytrainFinalCopy = copy.deepcopy(ytrainFinal)
        ytrainFinal = []
        for indx,each in enumerate(ytrainFinalCopy):

            div = (each-minnn)/((maxxx-minnn)/(n-1))
            floored = math.floor(div)
            ytrainFinal.append(int( bins[floored]))


        cff = rf.RandomForestClassifier( n_jobs = 3, random_state=1)###############################
        
        X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.5)

        #print(y_train)
        
       
        
        cff.fit(X_train, y_train)
        label1 = ttk.Label(self,text = "Train Score, iter 1: " +str(cff.score(X_train,y_train)))
        label1.pack()

        
        predictions = cff.predict(X_test)
        
        label2 = ttk.Label(self,text = "Test Score, iter 1: " +str(cff.score(X_test,y_test)))
        label2.pack()

        
        cff.n_estimators = 20
        cff.fit(X_train, y_train)
        label3 = ttk.Label(self,text = "Train Score, iter 2: " +str(cff.score(X_train,y_train)))
        label3.pack()
        
        cff.n_estimators = 30
        predictions = cff.predict(X_test)
        
        label2 = ttk.Label(self,text = "Test Score, iter 2: " +str(cff.score(X_test,y_test)))
        label2.pack()

        label3 = ttk.Label(self,text = "Train Score on training data minus outliers: " +str(cff.score(x,y)))
        label3.pack()

        label4 = ttk.Label(self,text = "Train Score on  ALL training data: " +str(cff.score(xtrainFinal,ytrainFinal)))
        label4.pack()


        ##### WRITE OUT ##### 
        testXData = testData.copy()
        del testXData['Id']
            
        y_prediction=cff.predict(testXData)
        y_prediction = np.exp(y_prediction)
        file = open('RandomForestClassifierNoOutliers.txt','w')
        file.write("Id,SalePrice\n")

        predictions_data = pandas.DataFrame(data = {'prediction':y_prediction})

        print("Printed: RandomForestClassifierNoOutliers")
        #print(predictions_data.shape)
        printPred = np.array(predictions_data)
        printId = np.array(testIds)
        for i in range(printId.size):
            file.write(str(printId[i])+","+str(printPred[i][0])+"\n")

        file.close()


class AdaBoostRegressorWithOutliers(tk.Frame):
     
     def __init__(self, parent, controller):
            tk.Frame.__init__(self, parent)
            label = tk.Label(self, text="AdaBoostRegressor Output With Outliers", font=LARGE_FONT)
            label.pack(pady=10,padx=10)

            button1 = ttk.Button(self, text="Back to Home",
                                command=lambda: controller.show_frame(StartPage))
            button1.pack()

            norData = copy.deepcopy(normData)
            y = copy.deepcopy(target)
            x = norData
            del x[len(norData)-1]
            del x[0]
            x = np.array(x).transpose()
            

                
            cff = AdaBoostRegressor(base_estimator = ExtraTreeRegressor(), n_estimators = 2000)####################################
            
            X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
            

                   
            cff.fit(X_train, y_train)
            label1 = ttk.Label(self,text = "Train Score, iter 1: " +str(cff.score(X_train,y_train)))
            label1.pack()

            
            predictions = cff.predict(X_test)
            label2 = ttk.Label(self,text = "Test Score, iter 1: " +str(cff.score(X_test,y_test)))
            label2.pack()
            
            X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.4)

            cff.n_estimators = 20
            cff.fit(X_train, y_train)
            label3 = ttk.Label(self,text = "Train Score, iter 2: " +str(cff.score(X_train,y_train)))
            label3.pack()
            
            cff.n_estimators = 30
            predictions = cff.predict(X_test)
            
            label2 = ttk.Label(self,text = "Test Score, iter 2: " +str(cff.score(X_test,y_test)))
            label2.pack()

            label3 = ttk.Label(self,text = "Train Score on all training data: " +str(cff.score(x,y)))
            label3.pack()

            robAB = make_pipeline(RobustScaler(), AdaBoostRegressor(base_estimator = ExtraTreeRegressor(), n_estimators = 2000)).fit(X_train, y_train)
            label4 = ttk.Label(self,text = "Pipeline Train Score on  ALL training data: " +str(robAB.score(x,y)))
            label4.pack()
                              
            testXData = testData.copy()
            del testXData['Id']


            ##### WRITE OUT #####    
            testXData = testData.copy()
            del testXData['Id']
             
            y_prediction=robAB.predict(testXData)
            y_prediction = np.exp(y_prediction)

            file = open('PipelineAdaBoostRegressorWithOutliers.txt','w')
            file.write("Id,SalePrice\n")

            predictions_data = pandas.DataFrame(data = {'prediction':y_prediction})

            print("Printed: AdaBoostRegressorWithOutliers")
            
            #print(predictions_data.shape)
            printPred = np.array(predictions_data)
            printId = np.array(testIds)
            for i in range(printId.size):
                file.write(str(printId[i])+","+str(printPred[i][0])+"\n")

            file.close()

class AdaBoostNoOutliers(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="AdaBoost Output No Outliers", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        button1 = ttk.Button(self, text="Back to Home",
                            command=lambda: controller.show_frame(StartPage))
        button1.pack()

        trainAllData = copy.deepcopy(normData)
        wOutOut = ro.returnWithoutOutliers(normData, target)
        norData = wOutOut[0]
        y = wOutOut[1]
        x = norData
        del x[len(norData)-1]
        del x[0]
##        print(len(x))
        x = np.array(x).transpose()
##        print(len(y))
        
##        minnn = 25000
##        maxxx = 1000000
##        n = 101
##        bins = np.linspace(minnn, maxxx, num = n)
##        ycopy = copy.deepcopy(y)
##        y = []
##        for indxx,each in enumerate(ycopy):
##            if each < minnn:
##                print("PRICE CANT BE LESS THAN MINNN")
##                sys.exit()
##            div = (each-minnn)/((maxxx-minnn)/(n-1))
##            floored = math.floor(div)
####            print(str(each) + " " +str( bins[floored]))
##            y.append(int(bins[floored]))
##            
##        print('-------------------------------------------------------------------------')
        
        ytrainFinal = copy.deepcopy(target)
        xtrainFinal = trainAllData
        del xtrainFinal[len(trainAllData)-1]
        del xtrainFinal[0]
        xtrainFinal = np.array(xtrainFinal).transpose()

##        ytrainFinalCopy = copy.deepcopy(ytrainFinal)
##        ytrainFinal = []
##        for indx,each in enumerate(ytrainFinalCopy):
##
##            div = (each-minnn)/((maxxx-minnn)/(n-1))
##            floored = math.floor(div)
##            ytrainFinal.append(int( bins[floored]))


        cff = AdaBoostRegressor(base_estimator = ExtraTreeRegressor(), n_estimators = 2000)####################################
        
        X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.5)

        #print(y_train)
        
       
        
        cff.fit(X_train, y_train)
        label1 = ttk.Label(self,text = "Train Score, iter 1: " +str(cff.score(X_train,y_train)))
        label1.pack()

        
        predictions = cff.predict(X_test)
        
        label2 = ttk.Label(self,text = "Test Score, iter 1: " +str(cff.score(X_test,y_test)))
        label2.pack()

        
        cff.n_estimators = 20
        cff.fit(X_train, y_train)
        label3 = ttk.Label(self,text = "Train Score, iter 2: " +str(cff.score(X_train,y_train)))
        label3.pack()
        
        cff.n_estimators = 30
        predictions = cff.predict(X_test)
        
        label2 = ttk.Label(self,text = "Test Score, iter 2: " +str(cff.score(X_test,y_test)))
        label2.pack()

        label3 = ttk.Label(self,text = "Train Score on training data minus outliers: " +str(cff.score(x,y)))
        label3.pack()

        label4 = ttk.Label(self,text = "Train Score on  ALL training data: " +str(cff.score(xtrainFinal,ytrainFinal)))
        label4.pack()


        


        ##### WRITE OUT ##### 
        testXData = testData.copy()
        del testXData['Id']
            
        y_prediction=cff.predict(testXData)
        y_prediction = np.exp(y_prediction)

        file = open('AdaBoostNoOutliers.txt','w')
        file.write("Id,SalePrice\n")

        predictions_data = pandas.DataFrame(data = {'prediction':y_prediction})

        print("Printed: AdaBoostNoOutliers")
        #print(predictions_data.shape)
        printPred = np.array(predictions_data)
        printId = np.array(testIds)
        for i in range(printId.size):
            file.write(str(printId[i])+","+str(printPred[i][0])+"\n")

        file.close()


class LinearModelWithOutliers(tk.Frame):
     
     def __init__(self, parent, controller):
            tk.Frame.__init__(self, parent)
            label = tk.Label(self, text="LinearModel Output With Outliers", font=LARGE_FONT)
            label.pack(pady=10,padx=10)

            button1 = ttk.Button(self, text="Back to Home",
                                command=lambda: controller.show_frame(StartPage))
            button1.pack()

            norData = copy.deepcopy(normData)
            y = copy.deepcopy(target)
            x = norData
            del x[len(norData)-1]
            del x[0]
            x = np.array(x).transpose()
            

                
            cff = linear_model.LinearRegression()
            X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
            

                   
            cff.fit(X_train, y_train)
            label1 = ttk.Label(self,text = "Train Score, iter 1: " +str(cff.score(X_train,y_train)))
            label1.pack()

            
            predictions = cff.predict(X_test)
            label2 = ttk.Label(self,text = "Test Score, iter 1: " +str(cff.score(X_test,y_test)))
            label2.pack()
            
##            X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.4)
##
####            cff.n_estimators = 20
##            cff.fit(X_train, y_train)
##            label3 = ttk.Label(self,text = "Train Score, iter 2: " +str(cff.score(X_train,y_train)))
##            label3.pack()
##            
####            cff.n_estimators = 30
##            predictions = cff.predict(X_test)
##            
##            label2 = ttk.Label(self,text = "Test Score, iter 2: " +str(cff.score(X_test,y_test)))
##            label2.pack()
##
##
##            X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.7)
##
####            cff.n_estimators = 20
##            cff.fit(X_train, y_train)
##            label3 = ttk.Label(self,text = "Train Score, iter 3: " +str(cff.score(X_train,y_train)))
##            label3.pack()
##            
####            cff.n_estimators = 30
##            predictions = cff.predict(X_test)
##            
##            label2 = ttk.Label(self,text = "Test Score, iter 3: " +str(cff.score(X_test,y_test)))
##            label2.pack()

            label3 = ttk.Label(self,text = "Train Score on all training data: " +str(cff.score(x,y)))
            label3.pack()

                              
            testXData = testData.copy()
            del testXData['Id']


            ##### WRITE OUT #####    
            testXData = testData.copy()
            del testXData['Id']
             
            y_prediction=cff.predict(testXData)
            y_prediction = np.exp(y_prediction)

            file = open('LinearModelWithOutliers.txt','w')
            file.write("Id,SalePrice\n")

            predictions_data = pandas.DataFrame(data = {'prediction':y_prediction})

            print("Printed: LinearModelWithOutliers")
            
            #print(predictions_data.shape)
            printPred = np.array(predictions_data)
            printId = np.array(testIds)
            for i in range(printId.size):
                file.write(str(printId[i])+","+str(printPred[i][0])+"\n")

            file.close()

class LinearModelNoOutliers(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="RandomForestClassifier Output No Outliers", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        button1 = ttk.Button(self, text="Back to Home",
                            command=lambda: controller.show_frame(StartPage))
        button1.pack()

        trainAllData = copy.deepcopy(normData)
        wOutOut = ro.returnWithoutOutliers(normData, target)
        norData = wOutOut[0]
        y = wOutOut[1]
        x = norData
        del x[len(norData)-1]
        del x[0]
##        print(len(x))
        x = np.array(x).transpose()
##        print(len(y))
        ytrainFinal = copy.deepcopy(target)
        xtrainFinal = trainAllData
        del xtrainFinal[len(trainAllData)-1]
        del xtrainFinal[0]
        xtrainFinal = np.array(xtrainFinal).transpose()

        cff = linear_model.LinearRegression()###################################################################
        
        X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

        #print(y_train)
        
       
        
        cff.fit(X_train, y_train)
        label1 = ttk.Label(self,text = "Train Score, iter 1: " +str(cff.score(X_train,y_train)))
        label1.pack()

        
        predictions = cff.predict(X_test)
        
        label2 = ttk.Label(self,text = "Test Score, iter 1: " +str(cff.score(X_test,y_test)))
        label2.pack()
##        X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.4)
##
##        
####        cff.n_estimators = 20
##        cff.fit(X_train, y_train)
##        label3 = ttk.Label(self,text = "Train Score, iter 2: " +str(cff.score(X_train,y_train)))
##        label3.pack()
##        
####        cff.n_estimators = 30
##        predictions = cff.predict(X_test)
##        
##        label2 = ttk.Label(self,text = "Test Score, iter 2: " +str(cff.score(X_test,y_test)))
##        label2.pack()
##
##
##        X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.7)
##
##        
####        cff.n_estimators = 20
##        cff.fit(X_train, y_train)
##        label3 = ttk.Label(self,text = "Train Score, iter 3: " +str(cff.score(X_train,y_train)))
##        label3.pack()
##        
####        cff.n_estimators = 30
##        predictions = cff.predict(X_test)
##        
##        label2 = ttk.Label(self,text = "Test Score, iter 3: " +str(cff.score(X_test,y_test)))
##        label2.pack()

        label3 = ttk.Label(self,text = "Train Score on training data minus outliers: " +str(cff.score(x,y)))
        label3.pack()

        label4 = ttk.Label(self,text = "Train Score on  ALL training data: " +str(cff.score(xtrainFinal,ytrainFinal)))
        label4.pack()


        ##### WRITE OUT ##### 
        testXData = testData.copy()
        del testXData['Id']
            
        y_prediction=cff.predict(testXData)
        y_prediction = np.exp(y_prediction)

        file = open('LinearModelNoOutliers.txt','w')
        file.write("Id,SalePrice\n")

        predictions_data = pandas.DataFrame(data = {'prediction':y_prediction})

        print("Printed: LinearModelNoOutliers")
        #print(predictions_data.shape)
        printPred = np.array(predictions_data)
        printId = np.array(testIds)
        for i in range(printId.size):
            file.write(str(printId[i])+","+str(printPred[i][0])+"\n")

        file.close()



class RidgeWithOutliers(tk.Frame):
     
     def __init__(self, parent, controller):
            tk.Frame.__init__(self, parent)
            label = tk.Label(self, text="Ridge Output With Outliers", font=LARGE_FONT)
            label.pack(pady=10,padx=10)

            button1 = ttk.Button(self, text="Back to Home",
                                command=lambda: controller.show_frame(StartPage))
            button1.pack()

            norData = copy.deepcopy(normData)
            y = copy.deepcopy(target)
            x = norData
            del x[len(norData)-1]
            del x[0]
            x = np.array(x).transpose()
            cff = linear_model.RidgeCV([0.1, 1.0, 10.0])
            X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
            

                   
            cff.fit(X_train, y_train)
            label1 = ttk.Label(self,text = "Train Score, iter 1: " +str(cff.score(X_train,y_train)))
            label1.pack()

            
            predictions = cff.predict(X_test)
            label2 = ttk.Label(self,text = "Test Score, iter 1: " +str(cff.score(X_test,y_test)))
            label2.pack()

            label3 = ttk.Label(self,text = "Train Score on all training data: " +str(cff.score(x,y)))
            label3.pack()
            
            cf = linear_model.RidgeCV([0.1, 1.0,  5.0, 7.5, 10.0, 20.0])

            
            X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
            

                   
            cf.fit(X_train, y_train)
            label1 = ttk.Label(self,text = "Train ScoreCV, iter 1: " +str(cf.score(X_train,y_train)))
            label1.pack()

            
            predictions = cf.predict(X_test)
            label2 = ttk.Label(self,text = "Test ScoreCV, iter 1: " +str(cf.score(X_test,y_test)))
            label2.pack()
            
##            X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.4)
##
####            cff.n_estimators = 20
##            cff.fit(X_train, y_train)
##            label3 = ttk.Label(self,text = "Train Score, iter 2: " +str(cff.score(X_train,y_train)))
##            label3.pack()
##            
####            cff.n_estimators = 30
##            predictions = cff.predict(X_test)
##            
##            label2 = ttk.Label(self,text = "Test Score, iter 2: " +str(cff.score(X_test,y_test)))
##            label2.pack()
##
##
##            X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.7)
##
####            cff.n_estimators = 20
##            cff.fit(X_train, y_train)
##            label3 = ttk.Label(self,text = "Train Score, iter 3: " +str(cff.score(X_train,y_train)))
##            label3.pack()
##            
####            cff.n_estimators = 30
##            predictions = cff.predict(X_test)
##            
##            label2 = ttk.Label(self,text = "Test Score, iter 3: " +str(cff.score(X_test,y_test)))
##            label2.pack()

            label3 = ttk.Label(self,text = "Train Score on all training data: " +str(cf.score(x,y)))
            label3.pack()

                              
            testXData = testData.copy()
            del testXData['Id']


            ##### WRITE OUT #####    
            testXData = testData.copy()
            del testXData['Id']
             
            y_prediction=cff.predict(testXData)
            y_prediction = np.exp(y_prediction)

            file = open('RidgeWithOutliers.txt','w')
            file.write("Id,SalePrice\n")

            predictions_data = pandas.DataFrame(data = {'prediction':y_prediction})

            print("Printed: RidgeWithOutliers")
            
            #print(predictions_data.shape)
            printPred = np.array(predictions_data)
            printId = np.array(testIds)
            for i in range(printId.size):
                file.write(str(printId[i])+","+str(printPred[i][0])+"\n")

            file.close()

class RidgeNoOutliers(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Ridge Output No Outliers", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        button1 = ttk.Button(self, text="Back to Home",
                            command=lambda: controller.show_frame(StartPage))
        button1.pack()

        trainAllData = copy.deepcopy(normData)
        wOutOut = ro.returnWithoutOutliers(normData, target)
        norData = wOutOut[0]
        y = wOutOut[1]
        x = norData
        del x[len(norData)-1]
        del x[0]
##        print(len(x))
        x = np.array(x).transpose()
##        print(len(y))
        ytrainFinal = copy.deepcopy(target)
        xtrainFinal = trainAllData
        del xtrainFinal[len(trainAllData)-1]
        del xtrainFinal[0]
        xtrainFinal = np.array(xtrainFinal).transpose()

        cff = linear_model.Ridge(alpha = .5)
        
        X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

        #print(y_train)
        
       
        
        cff.fit(X_train, y_train)
        label1 = ttk.Label(self,text = "Train Score, iter 1: " +str(cff.score(X_train,y_train)))
        label1.pack()

        
        predictions = cff.predict(X_test)
        
        label2 = ttk.Label(self,text = "Test Score, iter 1: " +str(cff.score(X_test,y_test)))
        label2.pack()
##        X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.4)
##
##        
####        cff.n_estimators = 20
##        cff.fit(X_train, y_train)
##        label3 = ttk.Label(self,text = "Train Score, iter 2: " +str(cff.score(X_train,y_train)))
##        label3.pack()
##        
####        cff.n_estimators = 30
##        predictions = cff.predict(X_test)
##        
##        label2 = ttk.Label(self,text = "Test Score, iter 2: " +str(cff.score(X_test,y_test)))
##        label2.pack()
##
##
##        X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.7)
##
##        
####        cff.n_estimators = 20
##        cff.fit(X_train, y_train)
##        label3 = ttk.Label(self,text = "Train Score, iter 3: " +str(cff.score(X_train,y_train)))
##        label3.pack()
##        
####        cff.n_estimators = 30
##        predictions = cff.predict(X_test)
##        
##        label2 = ttk.Label(self,text = "Test Score, iter 3: " +str(cff.score(X_test,y_test)))
##        label2.pack()

        label3 = ttk.Label(self,text = "Train Score on training data minus outliers: " +str(cff.score(x,y)))
        label3.pack()

        label4 = ttk.Label(self,text = "Train Score on  ALL training data: " +str(cff.score(xtrainFinal,ytrainFinal)))
        label4.pack()


        ##### WRITE OUT ##### 
        testXData = testData.copy()
        del testXData['Id']
            
        y_prediction=cff.predict(testXData)
        y_prediction = np.exp(y_prediction)

        file = open('RidgeNoOutliers.txt','w')
        file.write("Id,SalePrice\n")

        predictions_data = pandas.DataFrame(data = {'prediction':y_prediction})

        print("Printed: RidgeNoOutliers")
        #print(predictions_data.shape)
        printPred = np.array(predictions_data)
        printId = np.array(testIds)
        for i in range(printId.size):
            file.write(str(printId[i])+","+str(printPred[i][0])+"\n")

        file.close()



class LassoWithOutliers(tk.Frame):
     
     def __init__(self, parent, controller):
            tk.Frame.__init__(self, parent)
            label = tk.Label(self, text="Lasso Output With Outliers", font=LARGE_FONT)
            label.pack(pady=10,padx=10)

            button1 = ttk.Button(self, text="Back to Home",
                                command=lambda: controller.show_frame(StartPage))
            button1.pack()

            norData = copy.deepcopy(normData)
            y = copy.deepcopy(target)
            x = norData
            del x[len(norData)-1]
            del x[0]
            x = np.array(x).transpose()



            
            
            X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
            #______________________________________________________________________________________________

            cff = linear_model.Lasso(alpha = 10.0)       
            cff.fit(X_train, y_train)
            
            label1 = ttk.Label(self,text = "Lasso Train Score, iter 1: " +str(cff.score(X_train,y_train)))
            label1.pack()

            
            predictions = cff.predict(X_test)
            label2 = ttk.Label(self,text = "Lasso Test Score, iter 1: " +str(cff.score(X_test,y_test)))
            label2.pack()

            label3 = ttk.Label(self,text = "Lasso Train Score on all training data: " +str(cff.score(x,y)))
            label3.pack()

            #______________________________________________________________________________________________

            lars = linear_model.LassoLars(alpha = 10.0)       
            lars.fit(X_train, y_train)
            
            label1 = ttk.Label(self,text = "LassoLars Train Score, iter 1: " +str(lars.score(X_train,y_train)))
            label1.pack()

            
            predictions = lars.predict(X_test)
            label2 = ttk.Label(self,text = "LassoLars Test Score, iter 1: " +str(lars.score(X_test,y_test)))
            label2.pack()

            label3 = ttk.Label(self,text = "LassoLars Train Score on all training data: " +str(lars.score(x,y)))
            label3.pack()
            
            #______________________________________________________________________________________________

            larsIC_bic = linear_model.LassoLarsIC(criterion='bic')
            larsIC_bic.fit(X_train, y_train)

            label1 = ttk.Label(self,text = "larsIC_bic Train Score, iter 1: " +str(larsIC_bic.score(X_train,y_train)))
            label1.pack()

            
            predictions = larsIC_bic.predict(X_test)


            label2 = ttk.Label(self,text = "larsIC_bic Test Score, iter 1: " +str(larsIC_bic.score(X_test,y_test)))
            label2.pack()

            label3 = ttk.Label(self,text = "larsIC_bic Train Score on all training data: " +str(larsIC_bic.score(x,y)))
            label3.pack()
            
            #______________________________________________________________________________________________

            larsIC_aic = linear_model.LassoLarsIC(criterion='aic')
            larsIC_aic.fit(X_train, y_train)

            label1 = ttk.Label(self,text = "larsIC_aic Train Score, iter 1: " +str(larsIC_aic.score(X_train,y_train)))
            label1.pack()

            
            predictions = larsIC_aic.predict(X_test)

            
            label2 = ttk.Label(self,text = "larsIC_aic Test Score, iter 1: " +str(larsIC_aic.score(X_test,y_test)))
            label2.pack()

            label3 = ttk.Label(self,text = "larsIC_aic Train Score on all training data: " +str(larsIC_aic.score(x,y)))
            label3.pack()

            #______________________________________________________________________________________________

            lassoCv = linear_model.LassoCV(cv = 20).fit(X_train, y_train)

            label1 = ttk.Label(self,text = "lassoCV Train Score, iter 1: " +str(lassoCv.score(X_train,y_train)))
            label1.pack()

            
            predictions = lassoCv.predict(X_test)

            
            label2 = ttk.Label(self,text = "lassoCV Test Score, iter 1: " +str(lassoCv.score(X_test,y_test)))
            label2.pack()

            label3 = ttk.Label(self,text = "lassoCV Train Score on all training data: " +str(lassoCv.score(x,y)))
            label3.pack()


            #______________________________________________________________________________________________

            lassoLarsCv = linear_model.LassoLarsCV(cv = 20).fit(X_train, y_train)

            label1 = ttk.Label(self,text = "lassoLarsCv Train Score, iter 1: " +str(lassoLarsCv.score(X_train,y_train)))
            label1.pack()

            
            predictions = lassoLarsCv.predict(X_test)

            
            label2 = ttk.Label(self,text = "lassoLarsCv Test Score, iter 1: " +str(lassoLarsCv.score(X_test,y_test)))
            label2.pack()

            label3 = ttk.Label(self,text = "lassoLarsCv Train Score on all training data: " +str(lassoLarsCv.score(x,y)))
            label3.pack()

            #################!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            lassoPipeline = make_pipeline(RobustScaler(), linear_model.Lasso(alpha = 5, random_state=123)).fit(X_train, y_train)


            label3 = ttk.Label(self,text = "lassoPipeline 5 Train Score on all training data: " +str(lassoPipeline.score(x,y)))
            label3.pack()
#################!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            lassoPipeline = make_pipeline(RobustScaler(), linear_model.Lasso(alpha = 0.5, random_state=123)).fit(X_train, y_train)


            label3 = ttk.Label(self,text = "lassoPipeline .5 Train Score on all training data: " +str(lassoPipeline.score(x,y)))
            label3.pack()
#################!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            lassoPipeline = make_pipeline(RobustScaler(), linear_model.Lasso(alpha = 0.05, random_state=123)).fit(X_train, y_train)


            label3 = ttk.Label(self,text = "lassoPipeline .05 Train Score on all training data: " +str(lassoPipeline.score(x,y)))
            label3.pack()

            #################!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            lassoPipeline = make_pipeline(RobustScaler(), linear_model.Lasso(alpha = 0.005, random_state=123)).fit(X_train, y_train)


            label3 = ttk.Label(self,text = "lassoPipeline .005 Train Score on all training data: " +str(lassoPipeline.score(x,y)))
            label3.pack()

            lassoPipeline = make_pipeline(RobustScaler(), linear_model.Lasso(alpha = 0.0005, random_state=123)).fit(X_train, y_train)


            label3 = ttk.Label(self,text = "lassoPipeline .0005 Train Score on all training data: " +str(lassoPipeline.score(x,y)))
            label3.pack()




            
            

                              
            testXData = testData.copy()
            del testXData['Id']


            ##### WRITE OUT #####    
            testXData = testData.copy()
            del testXData['Id']
             
            y_prediction=lassoPipeline.predict(testXData)
            y_prediction = np.exp(y_prediction)

            file = open('LassoPipelineOutliers.txt','w')
            file.write("Id,SalePrice\n")

            predictions_data = pandas.DataFrame(data = {'prediction':y_prediction})

            print("Printed: LassoWithOutliers")
            
            #print(predictions_data.shape)
            printPred = np.array(predictions_data)
            printId = np.array(testIds)
            for i in range(printId.size):
                file.write(str(printId[i])+","+str(printPred[i][0])+"\n")

            file.close()

class LassoNoOutliers(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Lasso Output No Outliers", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        button1 = ttk.Button(self, text="Back to Home",
                            command=lambda: controller.show_frame(StartPage))
        button1.pack()

        trainAllData = copy.deepcopy(normData)
        wOutOut = ro.returnWithoutOutliers(normData, target)
        norData = wOutOut[0]
        y = wOutOut[1]
        x = norData
        del x[len(norData)-1]
        del x[0]
##        print(len(x))
        x = np.array(x).transpose()
##        print(len(y))
        ytrainFinal = copy.deepcopy(target)
        xtrainFinal = trainAllData
        del xtrainFinal[len(trainAllData)-1]
        del xtrainFinal[0]
        xtrainFinal = np.array(xtrainFinal).transpose()


        


        #print(y_train)
        
       
            
            
        X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
        #______________________________________________________________________________________________

        cff = linear_model.Lasso(alpha = 10.0)       
        cff.fit(X_train, y_train)
        
        label1 = ttk.Label(self,text = "Lasso Train Score, iter 1: " +str(cff.score(X_train,y_train)))
        label1.pack()

        
        predictions = cff.predict(X_test)
        label2 = ttk.Label(self,text = "Lasso Test Score, iter 1: " +str(cff.score(X_test,y_test)))
        label2.pack()

        label3 = ttk.Label(self,text = "Lasso Train Score on all training data: " +str(cff.score(x,y)))
        label3.pack()

        #______________________________________________________________________________________________

        lars = linear_model.LassoLars(alpha = 10.0)       
        lars.fit(X_train, y_train)
        
        label1 = ttk.Label(self,text = "LassoLars Train Score, iter 1: " +str(lars.score(X_train,y_train)))
        label1.pack()

        
        predictions = lars.predict(X_test)
        label2 = ttk.Label(self,text = "LassoLars Test Score, iter 1: " +str(lars.score(X_test,y_test)))
        label2.pack()

        label3 = ttk.Label(self,text = "LassoLars Train Score on all training data: " +str(lars.score(x,y)))
        label3.pack()
        
        #______________________________________________________________________________________________

        larsIC_bic = linear_model.LassoLarsIC(criterion='bic')
        larsIC_bic.fit(X_train, y_train)

        label1 = ttk.Label(self,text = "larsIC_bic Train Score, iter 1: " +str(larsIC_bic.score(X_train,y_train)))
        label1.pack()

        
        predictions = larsIC_bic.predict(X_test)


        label2 = ttk.Label(self,text = "larsIC_bic Test Score, iter 1: " +str(larsIC_bic.score(X_test,y_test)))
        label2.pack()

        label3 = ttk.Label(self,text = "larsIC_bic Train Score on all training data: " +str(larsIC_bic.score(x,y)))
        label3.pack()
        
        #______________________________________________________________________________________________

        larsIC_aic = linear_model.LassoLarsIC(criterion='aic')
        larsIC_aic.fit(X_train, y_train)

        label1 = ttk.Label(self,text = "larsIC_aic Train Score, iter 1: " +str(larsIC_aic.score(X_train,y_train)))
        label1.pack()

        
        predictions = larsIC_aic.predict(X_test)

        
        label2 = ttk.Label(self,text = "larsIC_aic Test Score, iter 1: " +str(larsIC_aic.score(X_test,y_test)))
        label2.pack()

        label3 = ttk.Label(self,text = "larsIC_aic Train Score on all training data: " +str(larsIC_aic.score(x,y)))
        label3.pack()

        #______________________________________________________________________________________________

        lassoCv = linear_model.LassoCV(cv = 20).fit(X_train, y_train)

        label1 = ttk.Label(self,text = "lassoCV Train Score, iter 1: " +str(lassoCv.score(X_train,y_train)))
        label1.pack()

        
        predictions = lassoCv.predict(X_test)

        
        label2 = ttk.Label(self,text = "lassoCV Test Score, iter 1: " +str(lassoCv.score(X_test,y_test)))
        label2.pack()

        label3 = ttk.Label(self,text = "lassoCV Train Score on all training data: " +str(lassoCv.score(x,y)))
        label3.pack()


        #______________________________________________________________________________________________

        lassoLarsCv = linear_model.LassoLarsCV(cv = 20).fit(X_train, y_train)

        label1 = ttk.Label(self,text = "lassoLarsCv Train Score, iter 1: " +str(lassoLarsCv.score(X_train,y_train)))
        label1.pack()

        
        predictions = lassoLarsCv.predict(X_test)

        
        label2 = ttk.Label(self,text = "lassoLarsCv Test Score, iter 1: " +str(lassoLarsCv.score(X_test,y_test)))
        label2.pack()

        label3 = ttk.Label(self,text = "lassoLarsCv Train Score on all training data: " +str(lassoLarsCv.score(x,y)))
        label3.pack()


        lassoPipeline = make_pipeline(RobustScaler(), linear_model.Lasso(alpha = 0.5, random_state=123)).fit(X_train, y_train)

        label3 = ttk.Label(self,text = "lassoPipeline Train Score on all training data: " +str(lassoPipeline.score(x,y)))
        label3.pack()
##
##        ##### WRITE OUT ##### 
##        testXData = testData.copy()
##        del testXData['Id']
##            
##        y_prediction=cff.predict(testXData)
##
##        file = open('LassoNoOutliers.txt','w')
##        file.write("Id,SalePrice\n")
##
##        predictions_data = pandas.DataFrame(data = {'prediction':y_prediction})
##
##        print("Printed: LassoNoOutliers")
##        #print(predictions_data.shape)
##        printPred = np.array(predictions_data)
##        printId = np.array(testIds)
##        for i in range(printId.size):
##            file.write(str(printId[i])+","+str(printPred[i][0])+"\n")
##
##        file.close()

class RANSACRegressorWithOutliers(tk.Frame):
     
     def __init__(self, parent, controller):
            tk.Frame.__init__(self, parent)
            

            button1 = ttk.Button(self, text="Back to Home",
                                command=lambda: controller.show_frame(StartPage))
            button1.pack()

            norData = copy.deepcopy(normData)
            y = copy.deepcopy(target)
            x = norData
            del x[len(norData)-1]
            del x[0]
            x = np.array(x).transpose()
            

                
            

            
            X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
            

            label = tk.Label(self, text="RANSAC: LinearModel Output With Outliers", font=LARGE_FONT)
            label.pack(pady=10,padx=10)



            
            ransacL = linear_model.RANSACRegressor(stop_score = .94)
            
            ransacL.fit(x,y)

            
            label1 = ttk.Label(self,text = "Train Score, iter 1: " +str(ransacL.score(X_train,y_train)))
            label1.pack()

            
            predictions = ransacL.predict(X_test)
            label2 = ttk.Label(self,text = "Test Score, iter 1: " +str(ransacL.score(X_test,y_test)))
            label2.pack()
            
            label3 = ttk.Label(self,text = "Train Score on all training data: " +str(ransacL.score(x,y)))
            label3.pack()


            #_____________________________________________________________________________________

            label = tk.Label(self, text="RANSAC: RandomForestRegressor Output With Outliers", font=LARGE_FONT)
            label.pack(pady=10,padx=10)
            
            ransacRR = linear_model.RANSACRegressor(base_estimator= RandomForestRegressor(), stop_score = .94)
            
            ransacRR.fit(x,y)

            
            label1 = ttk.Label(self,text = "Train Score, iter 1: " +str(ransacRR.score(X_train,y_train)))
            label1.pack()

            
            predictions = ransacRR.predict(X_test)
            label2 = ttk.Label(self,text = "Test Score, iter 1: " +str(ransacRR.score(X_test,y_test)))
            label2.pack()
            
            label3 = ttk.Label(self,text = "Train Score on all training data: " +str(ransacRR.score(x,y)))
            label3.pack()


            #_____________________________________________________________________________________

            label = tk.Label(self, text="RANSAC: RANSAC RandomForestRegressor Output With Outliers", font=LARGE_FONT)
            label.pack(pady=10,padx=10)
            
            ransacRR = linear_model.RANSACRegressor(linear_model.RANSACRegressor(base_estimator= RandomForestRegressor()), stop_score = .94)
            
            ransacRR.fit(x,y)

            
            label1 = ttk.Label(self,text = "Train Score, iter 1: " +str(ransacRR.score(X_train,y_train)))
            label1.pack()

            
            predictions = ransacRR.predict(X_test)
            label2 = ttk.Label(self,text = "Test Score, iter 1: " +str(ransacRR.score(X_test,y_test)))
            label2.pack()
            
            label3 = ttk.Label(self,text = "Train Score on all training data: " +str(ransacRR.score(x,y)))
            label3.pack()


            #_____________________________________________________________________________________



                              
            testXData = testData.copy()
            del testXData['Id']



class RANSACRegressorNoOutliers(tk.Frame):
     
     def __init__(self, parent, controller):
            tk.Frame.__init__(self, parent)
            

            button1 = ttk.Button(self, text="Back to Home",
                                command=lambda: controller.show_frame(StartPage))
            button1.pack()

            trainAllData = copy.deepcopy(normData)
            wOutOut = ro.returnWithoutOutliers(normData, target)
            norData = wOutOut[0]
            y = wOutOut[1]
            x = norData
            del x[len(norData)-1]
            del x[0]
    ##        print(len(x))
            x = np.array(x).transpose()
    ##        print(len(y))
            ytrainFinal = copy.deepcopy(target)
            xtrainFinal = trainAllData
            del xtrainFinal[len(trainAllData)-1]
            del xtrainFinal[0]
            xtrainFinal = np.array(xtrainFinal).transpose()




            
            X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
            

            label = tk.Label(self, text="RANSAC: LinearModel Output No Outliers", font=LARGE_FONT)
            label.pack(pady=10,padx=10)



            
            ransacL = linear_model.RANSACRegressor(stop_score = .94)
            
            ransacL.fit(x,y)

            
            label1 = ttk.Label(self,text = "Train Score, iter 1: " +str(ransacL.score(X_train,y_train)))
            label1.pack()

            
            predictions = ransacL.predict(X_test)
            label2 = ttk.Label(self,text = "Test Score, iter 1: " +str(ransacL.score(X_test,y_test)))
            label2.pack()
            
            label3 = ttk.Label(self,text = "Train Score on all training data: " +str(ransacL.score(xtrainFinal,ytrainFinal)))
            label3.pack()


            #_____________________________________________________________________________________

            label = tk.Label(self, text="RANSAC: RandomForestRegressor Output No Outliers", font=LARGE_FONT)
            label.pack(pady=10,padx=10)
            
            ransacRR = linear_model.RANSACRegressor(base_estimator= RandomForestRegressor(), stop_score = .94)
            
            ransacRR.fit(x,y)

            
            label1 = ttk.Label(self,text = "Train Score, iter 1: " +str(ransacRR.score(X_train,y_train)))
            label1.pack()

            
            predictions = ransacRR.predict(X_test)
            label2 = ttk.Label(self,text = "Test Score, iter 1: " +str(ransacRR.score(X_test,y_test)))
            label2.pack()
            
            label3 = ttk.Label(self,text = "Train Score on all training data: " +str(ransacRR.score(xtrainFinal,ytrainFinal)))
            label3.pack()


            #_____________________________________________________________________________________

            label = tk.Label(self, text="RANSAC: RANSAC RandomForestRegressor Output No Outliers", font=LARGE_FONT)
            label.pack(pady=10,padx=10)
            
            ransacRR = linear_model.RANSACRegressor(linear_model.RANSACRegressor(base_estimator= RandomForestRegressor()), stop_score = .94)
            
            ransacRR.fit(x,y)

            
            label1 = ttk.Label(self,text = "Train Score, iter 1: " +str(ransacRR.score(X_train,y_train)))
            label1.pack()

            
            predictions = ransacRR.predict(X_test)
            label2 = ttk.Label(self,text = "Test Score, iter 1: " +str(ransacRR.score(X_test,y_test)))
            label2.pack()
            
            label3 = ttk.Label(self,text = "Train Score on all training data: " +str(ransacRR.score(xtrainFinal,ytrainFinal)))
            label3.pack()


            #_____________________________________________________________________________________



class CombineRegressors(tk.Frame):
     
     def __init__(self, parent, controller):
            tk.Frame.__init__(self, parent)
            

            button1 = ttk.Button(self, text="Back to Home",
                                command=lambda: controller.show_frame(StartPage))
            button1.pack()

            norData = copy.deepcopy(normData)
            y = copy.deepcopy(target)
            x = norData
            del x[len(norData)-1]
            del x[0]
            x = np.array(x).transpose()
            

                
            testXData = testData.copy()
            del testXData['Id']



            wOutOut = ro.returnWithoutOutliers(normData, target)
##        print(pandas.DataFrame(data = norData).shape)
##        corrData = correlation.correlat(freq, .6,  normalizedData = norData)
##        corrData = pandas.DataFrame(data = corrData)  
##        indsT = names.get(list(corrData.iloc[:,0]))
##        importantIndices = list(indsT)
            ynoOut = wOutOut[1]
            xnoOut = wOutOut[0]
            del xnoOut[len(xnoOut)-1]
            del xnoOut[0]
            xnoOut = np.array(xnoOut).transpose()

            
            X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.5)

            toTrainX = xnoOut
            toTrainY = ynoOut

            #toTest = testXData
            toTest = x

            robAB = make_pipeline(RobustScaler(), AdaBoostRegressor(base_estimator = ExtraTreeRegressor(), n_estimators = 2000)).fit(toTrainX, toTrainY)
            Adaprediction = robAB.predict(toTest)
            
            label4 = ttk.Label(self,text = "ADA Train Score on  ALL training data: " +str(robAB.score(x,y)))
            label4.pack()
            
            rff = rf.RandomForestRegressor(n_estimators = 1000, random_state = 42,oob_score =True, max_features = "auto", warm_start=True).fit(toTrainX, toTrainY)      
            
            rff.n_estimators = 3000
            RFpredictions = rff.predict(toTest)

            label3 = ttk.Label(self,text = "RF Train Score on all training data: " +str(rff.score(x,y)))
            label3.pack()

            eNetRob = make_pipeline(RobustScaler(), linear_model.ElasticNet(alpha = 0.0005, l1_ratio = .9, fit_intercept = True)).fit(toTrainX, toTrainY)
            eNetpredictions = eNetRob.predict(toTest)

            label3 = ttk.Label(self,text = "ENET Train Score on all training data: " +str(eNetRob.score(x,y)))
            label3.pack()

            kernRg = KernelRidge(alpha = 0.5, kernel = 'polynomial').fit(toTrainX, toTrainY)
            kernRgpredictions = eNetRob.predict(toTest)

            label3 = ttk.Label(self,text = "Kern Train Score on all training data: " +str(kernRg.score(x,y)))
            label3.pack()

            gradBoost = GradientBoostingRegressor(loss = 'huber', learning_rate = 0.06, n_estimators = 4000, max_depth = 4, min_samples_split = 10, min_samples_leaf = 15).fit(toTrainX, toTrainY)
            gradBoostPred = eNetRob.predict(toTest)

            label3 = ttk.Label(self,text = "Gradient Boost Train Score on all training data: " +str(kernRg.score(x,y)))
            label3.pack()

            
            


            averaged = np.column_stack([Adaprediction,RFpredictions,eNetpredictions,kernRgpredictions])
            averaged = np.mean(averaged, axis = 1)
           
            newAveraged = averaged*.2+Adaprediction*.4+gradBoostPred*.2+RFpredictions*.2
##            testXData = testData.copy()
##            del testXData['Id']


            ##### WRITE OUT #####    
##            testXData = testData.copy()
##            plt.subplot(321)
##            plt.scatter(y,Adaprediction, color ='r')
##            plt.scatter(y,y,color = 'g')
##            
##
##            plt.subplot(322)
##            
##            plt.scatter(y, RFpredictions, color ='b')
##            plt.scatter(y,y,color = 'g')
##
##            plt.subplot(323)
##            
##            plt.scatter(y,kernRgpredictions, color ='c')
##            plt.scatter(y,y,color = 'g')
##
##            plt.subplot(324)
##            plt.scatter(y,gradBoostPred, color ='c')
##            plt.scatter(y,y,color = 'g')
##            
##            plt.subplot(325)
##            plt.scatter(y,averaged, color ='y')
##            plt.scatter(y,y,color = 'g')
####           
##
##
##
##            plt.subplot(326)
##            plt.scatter(y,newAveraged, color ='y')
##            plt.scatter(y,y,color = 'g')
##            plt.show()
##           
##
##
##            plt.subplot(325)
##            plt.scatter(y,newAveraged, color ='y')
##            plt.scatter(y,y,color = 'g')
##            
##            plt.show()
            

             
            y_prediction=newAveraged
            y_prediction = np.exp(y_prediction)

            file = open('LatestSubmission.txt','w')
            file.write("Id,SalePrice\n")

            predictions_data = pandas.DataFrame(data = {'prediction':y_prediction})

            print("Printed: lATEST")
            
            #print(predictions_data.shape)
            printPred = np.array(predictions_data)
            printId = np.array(testIds)
            for i in range(printId.size):
                file.write(str(printId[i])+","+str(printPred[i][0])+"\n")

            file.close()


            
             
app = HousePrices()
app.mainloop()
 
