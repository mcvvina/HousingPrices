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
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error




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

toPrintSkew = toPrintSkew[abs(toPrintSkew.values)> 2]

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

choiceOfPrints = input("Type 'y' if you would like predictions of methods printed. Default is none printed\n")
if choiceOfPrints is "y":
    choiceOfPrints = True
else:
    choiceOfPrints = False

    
choiceOfScore = input("Type 'y' if you want the scores to be root mean squared error (note: VERY LONG COMPUTATION TIME).Default is mean accuracy\n")
def calcScore(model, X, Y):
    if choiceOfScore is "y":
        return np.sqrt(-cross_val_score(model, (X), (Y), scoring = 'neg_mean_squared_error', cv = 5))
    else:
        return model.score(X,Y)

if choiceOfScore is "y":
    print("You selected: ROOT MEAN SQUARED ERROR. Please Wait until 'FINISHED' is printed.\n")
else:
    print("You selected: MEAN ACCURACY. Please Wait until 'FINISHED' is printed.\n")


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

        print("LOADING: PLEASE WAIT")
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
            print((F.__name__)+ " Status: ", end = '')
            frame = F(container, self)

            self.frames[F] = frame
            

            frame.grid(row=0, column=0, sticky="nsew")
            
            print("Finished Loading")
            

        self.show_frame(StartPage)
        print("PROGRAM FINISHED LOADING: Feel free to interact with GUI")

    def show_frame(self, cont):

        frame = self.frames[cont]
        frame.tkraise()

        
class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self,parent)
        self.canv = tk.Canvas(self, width=600, height=400,scrollregion=(0, 0, 1200, 800))
        self.canv.grid(row=0, column=0, rowspan = 23, columnspan=1)
        self.canv.configure(scrollregion=self.canv.bbox("all"))
        

                            
##        self.scrollY = ttk.Scrollbar(self,orient=tk.VERTICAL,command=self.canv.yview)
##        self.scrollY.grid(row=0, column=1,rowspan = 23, sticky=tk.N+tk.S)
##        self.canv.config(yscrollcommand=self.scrollY.set)
##        self.scrollX = tk.Scrollbar(self, orient=tk.HORIZONTAL,command=self.canv.xview)
##        self.scrollX.grid(row=24, column=0, rowspan = 23, sticky=tk.E+tk.W)
##        self.canv.config(xscrollcommand=self.scrollX.set)
##        scrollbar.pack(side ="right")
        label = tk.Label(self, text="Welcome!",font=('Helvetica', 14))
        label.grid(row=0, column=0,sticky=tk.E+tk.W)
##        label.pack(pady=10,padx=10)

        label = tk.Label(self, text="DataCleaning and Visualization", font=('Helvetica', 12))
        label.grid(row=1, column=0)#,sticky=tk.E+tk.W)
##        label.pack(pady=10,padx=10)
        
        button = ttk.Button(self, text="View Histograms of Attributes",
                            command=lambda: controller.show_frame(Histograms))
        button.grid(row=2, column=0)#,sticky=tk.E+tk.W)
##        button.pack()
##
        button2 = ttk.Button(self, text="Normalized Data Scatter Plots",
                            command=lambda: controller.show_frame(NormScatter))
        button2.grid(row=3, column=0)#,sticky=tk.E+tk.W)
##        self.canv['xscrollcommand'] = self.scrollX.set
##        self.canv['yscrollcommand'] = self.scrollY.set

##        button2.pack()
##
        button3 = ttk.Button(self, text="Normalized Data Scatter Plots NO outliers",
                            command=lambda: controller.show_frame(NormScatterNoOutlier))
        button3.grid(row=4, column=0)
##        button3.pack()
##
##        
        button4 = ttk.Button(self, text="Normalized Important Data Plots w/outliers",
                            command=lambda: controller.show_frame(NormImportantScatter))
        button4.grid(row=5, column=0)

##        button4.pack()

        button5 = ttk.Button(self, text="Normalized Important Data Plots NO outliers",
                            command=lambda: controller.show_frame(NormImportantOutlierScatter))
        button5.grid(row=5, column=0)
        button6 = ttk.Button(self, text="Skew Visualization",
                            command=lambda: controller.show_frame(SkewVisualization))
        button6.grid(row=6, column=0)


        button7 = ttk.Button(self, text="Price Visualization",
                            command=lambda: controller.show_frame(PriceVisualization))
        button7.grid(row=7, column=0)
        
        
        label = tk.Label(self, text="Techniques Used", font=('Helvetica', 12))
##        label.pack(pady=10,padx=10)
        label.grid(row=8, column=0)

        button8 = ttk.Button(self, text="RandomForestRegressor Output With Outliers",
                            command=lambda: controller.show_frame(RandomForestRegressorWithOutliers))
        button8.grid(row=9, column=0)
        
        button9 = ttk.Button(self, text="RandomForestRegressor Output No Outliers",
                            command=lambda: controller.show_frame(RandomForestRegressorNoOutliers))
        button9.grid(row=10, column=0)

        button10 = ttk.Button(self, text="RandomForestClassifier Output With Outliers",
                            command=lambda: controller.show_frame(RandomForestClassifierWithOutliers))
        button10.grid(row=11, column=0)

        button11 = ttk.Button(self, text="RandomForestClassifier Output No Outliers",
                            command=lambda: controller.show_frame(RandomForestClassifierNoOutliers))
        button11.grid(row=12, column=0)
        
        button = ttk.Button(self, text="AdaBoostRegressor Output With Outliers",
                            command=lambda: controller.show_frame(AdaBoostRegressorWithOutliers))
        button.grid(row=13, column=0)

        button = ttk.Button(self, text="AdaBoost Output No Outliers",
                            command=lambda: controller.show_frame(AdaBoostNoOutliers))
        button.grid(row=14, column=0)

        button = ttk.Button(self, text="LinearRegression Output With Outliers",
                            command=lambda: controller.show_frame(LinearModelWithOutliers))
        button.grid(row=15, column=0)

        button = ttk.Button(self, text="LinearRegression Output No Outliers",
                            command=lambda: controller.show_frame(LinearModelNoOutliers))
        button.grid(row=16, column=0)

        button = ttk.Button(self, text="Ridge Output With Outliers",
                            command=lambda: controller.show_frame(RidgeWithOutliers))
        button.grid(row=17, column=0)

        button = ttk.Button(self, text="Ridge Output No Outliers",
                            command=lambda: controller.show_frame(RidgeNoOutliers))
        button.grid(row=18, column=0)

        button = ttk.Button(self, text="Lasso Output With Outliers",
                            command=lambda: controller.show_frame(LassoWithOutliers))
        button.grid(row=19, column=0)

        button = ttk.Button(self, text="Lasso Output No Outliers",
                            command=lambda: controller.show_frame(LassoNoOutliers))
        button.grid(row=20, column=0)

        button = ttk.Button(self, text="RANSAC Output With Outliers",
                            command=lambda: controller.show_frame(RANSACRegressorWithOutliers))
        button.grid(row=21, column=0)

        button = ttk.Button(self, text="RANSAC Output No Outliers",
                            command=lambda: controller.show_frame(RANSACRegressorNoOutliers))
        button.grid(row=22, column=0)

        button = ttk.Button(self, text="Combining Regressors",
                            command=lambda: controller.show_frame(CombineRegressors))
        button.grid(row=23, column=0)
        
        


        

        

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

        
        X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.3)

        
        rff = rf.RandomForestRegressor(n_estimators = 3000, random_state = 42,oob_score =True, max_features = "auto", warm_start=True)

        rff.fit(X_train, y_train)
##        label1 = ttk.Label(self,text = "Train Score, iter 1: " +str(calcScore(rff, X_train,y_train)))
##        label1.pack()

##        
##        predictions = rff.predict(X_test)
##        
##        label2 = ttk.Label(self,text = "Test Score, iter 1: " +str(calcScore(rff,X_test,y_test)))
##        label2.pack()
##
##        rff.n_estimators = 2000
##        rff.fit(X_train, y_train)
##        label3 = ttk.Label(self,text = "Train Score, iter 2: " +str(calcScore(rff,X_train,y_train)))
##        label3.pack()
##        
##        rff.n_estimators = 3000
##        predictions = rff.predict(X_test)
##        
##        label2 = ttk.Label(self,text = "Test Score, iter 2: " +str(calcScore(rff,X_test,y_test)))
##        label2.pack()

        label3 = ttk.Label(self,text = "Train Score on all training data: " +str(calcScore(rff,x,y)))
        label3.pack()


        ##### WRITE OUT #####
        if choiceOfPrints:
            testXData = copy.deepcopy(testData)
            del testXData['Id']

              
            y_prediction=rff.predict(testXData)
            y_prediction = np.exp(y_prediction)

            file = open('RandomForestRegressorWithOutliers.txt','w')
            file.write("Id,SalePrice\n")

            predictions_data = pandas.DataFrame(data = {'prediction':y_prediction})

##            print("Printed: RandomForestRegressorWithOutliers")
            
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

        
        X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.3)


        rff = rf.RandomForestRegressor(n_estimators = 3000, random_state = 42,oob_score =True, max_features = "auto", warm_start=True)

        rff.fit(X_train, y_train)
##        label1 = ttk.Label(self,text = "Train Score, iter 1: " +str(calcScore(rff,X_train,y_train)))
##        label1.pack()
##
##        
##        predictions = rff.predict(X_test)
##        
##        label2 = ttk.Label(self,text = "Test Score, iter 1: " +str(calcScore(rff,X_test,y_test)))
##        label2.pack()
##
##        rff.n_estimators = 2000
##        rff.fit(X_train, y_train)
##        label3 = ttk.Label(self,text = "Train Score, iter 2: " +str(calcScore(rff,X_train,y_train)))
##        label3.pack()
##        
##        rff.n_estimators = 3000
##        predictions = rff.predict(X_test)
##        
##        label2 = ttk.Label(self,text = "Test Score, iter 2: " +str(calcScore(rff,X_test,y_test)))
##        label2.pack()
##
        label3 = ttk.Label(self,text = "Train Score on all training data: " +str(calcScore(rff,x,y)))
        label3.pack()

##        ##### WRITE OUT #####
        if choiceOfPrints:
            testXData = copy.deepcopy(testData)
            del testXData['Id']

             
            y_prediction=rff.predict(testXData)
            y_prediction = np.exp(y_prediction)
            
            file = open('RandomForestRegressorNoOutliers.txt','w')
            file.write("Id,SalePrice\n")

            predictions_data = pandas.DataFrame(data = {'prediction':y_prediction})

##            print("Printed: RandomForestRegressorNoOutliers")
            
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
        
        X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.3)
        

               
        cff.fit(X_train, y_train)
        label1 = ttk.Label(self,text = "Train Score, iter 1: " +str(calcScore(cff,X_train,y_train)))
        label1.pack()

##        
##        predictions = cff.predict(X_test)
##        label2 = ttk.Label(self,text = "Test Score, iter 1: " +str(calcScore(cff,X_test,y_test)))
##        label2.pack()
##        
##        X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.3)
##
##        cff.n_estimators = 20
##        cff.fit(X_train, y_train)
##        label3 = ttk.Label(self,text = "Train Score, iter 2: " +str(calcScore(cff,X_train,y_train)))
##        label3.pack()
##        
##        cff.n_estimators = 30
##        predictions = cff.predict(X_test)
##        
##        label2 = ttk.Label(self,text = "Test Score, iter 2: " +str(calcScore(cff,X_test,y_test)))
##        label2.pack()

        label3 = ttk.Label(self,text = "Train Score on all training data: " +str(calcScore(cff,x,y)))
        label3.pack()


        

            


        ##### WRITE OUT #####
        if choiceOfPrints:
            testXData = copy.deepcopy(testData)
            del testXData['Id']
             
            y_prediction=cff.predict(testXData)
            y_prediction = np.exp(y_prediction)

            file = open('RandomForestClassifierWithOutliers.txt','w')
            file.write("Id,SalePrice\n")

            predictions_data = pandas.DataFrame(data = {'prediction':y_prediction})

##            print("Printed: RandomForestClassifierWithOutliers")
            
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
        
        X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.3)

        #print(y_train)
        
       
##        
        cff.fit(X_train, y_train)
##        label1 = ttk.Label(self,text = "Train Score, iter 1: " +str(calcScore(cff,X_train,y_train)))
##        label1.pack()
##
##        
##        predictions = cff.predict(X_test)
##        
##        label2 = ttk.Label(self,text = "Test Score, iter 1: " +str(calcScore(cff,X_test,y_test)))
##        label2.pack()
##
##        
##        cff.n_estimators = 20
##        cff.fit(X_train, y_train)
##        label3 = ttk.Label(self,text = "Train Score, iter 2: " +str(calcScore(cff,X_train,y_train)))
##        label3.pack()
##        
##        cff.n_estimators = 30
##        predictions = cff.predict(X_test)
##        
##        label2 = ttk.Label(self,text = "Test Score, iter 2: " +str(calcScore(cff,X_test,y_test)))
##        label2.pack()

        label3 = ttk.Label(self,text = "Train Score on training data minus outliers: " +str(calcScore(cff,x,y)))
        label3.pack()

        label4 = ttk.Label(self,text = "Train Score on  ALL training data: " +str(calcScore(cff,xtrainFinal,ytrainFinal)))
        label4.pack()


        ##### WRITE OUT #####
        if choiceOfPrints:
            testXData = copy.deepcopy(testData)
            del testXData['Id']
                
            y_prediction=cff.predict(testXData)
            y_prediction = np.exp(y_prediction)
            file = open('RandomForestClassifierNoOutliers.txt','w')
            file.write("Id,SalePrice\n")

            predictions_data = pandas.DataFrame(data = {'prediction':y_prediction})

##            print("Printed: RandomForestClassifierNoOutliers")
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
            
            X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.3)
            

                   
            cff.fit(X_train, y_train)
##            label1 = ttk.Label(self,text = "Train Score, iter 1: " +str(calcScore(cff,X_train,y_train)))
##            label1.pack()
##
##            
##            predictions = cff.predict(X_test)
##            label2 = ttk.Label(self,text = "Test Score, iter 1: " +str(calcScore(cff,X_test,y_test)))
##            label2.pack()
##            
##            X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.3)
##
##            cff.n_estimators = 20
##            cff.fit(X_train, y_train)
##            label3 = ttk.Label(self,text = "Train Score, iter 2: " +str(calcScore(cff,X_train,y_train)))
##            label3.pack()
##            
##            cff.n_estimators = 30
##            predictions = cff.predict(X_test)
##            
##            label2 = ttk.Label(self,text = "Test Score, iter 2: " +str(calcScore(cff,X_test,y_test)))
##            label2.pack()

            label3 = ttk.Label(self,text = "Train Score on all training data: " +str(calcScore(cff,x,y)))
            label3.pack()

            if choiceOfPrints:
                ##### WRITE OUT #####    
                testXData = copy.deepcopy(testData)
                del testXData['Id']
                 
                y_prediction=robAB.predict(testXData)
                y_prediction = np.exp(y_prediction)

                file = open('AdaBoostRegressorWithOutliers.txt','w')
                file.write("Id,SalePrice\n")

                predictions_data = pandas.DataFrame(data = {'prediction':y_prediction})

##                print("Printed: AdaBoostRegressorWithOutliers")
                
                #print(predictions_data.shape)
                printPred = np.array(predictions_data)
                printId = np.array(testIds)
                for i in range(printId.size):
                    file.write(str(printId[i])+","+str(printPred[i][0])+"\n")

                file.close()


            robAB = make_pipeline(RobustScaler(), AdaBoostRegressor(base_estimator = ExtraTreeRegressor(), n_estimators = 2000)).fit(X_train, y_train)
            label4 = ttk.Label(self,text = "Pipeline Train Score on  ALL training data: " +str(calcScore(robAB,x,y)))
            label4.pack()
                              
           

            ##### WRITE OUT #####
            if choiceOfPrints:
                testXData = copy.deepcopy(testData)
                del testXData['Id']
                 
                y_prediction=robAB.predict(testXData)
                y_prediction = np.exp(y_prediction)

                file = open('PipelineAdaBoostRegressorWithOutliers.txt','w')
                file.write("Id,SalePrice\n")

                predictions_data = pandas.DataFrame(data = {'prediction':y_prediction})

##                print("Printed: AdaBoostRegressorWithOutliers")
                
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
        
        X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.3)

        #print(y_train)
        
       
        
        cff.fit(X_train, y_train)
##        label1 = ttk.Label(self,text = "Train Score, iter 1: " +str(calcScore(cff,X_train,y_train)))
##        label1.pack()
##
##        
##        predictions = cff.predict(X_test)
##        
##        label2 = ttk.Label(self,text = "Test Score, iter 1: " +str(calcScore(cff,X_test,y_test)))
##        label2.pack()
##
##        
##        cff.n_estimators = 20
##        cff.fit(X_train, y_train)
##        label3 = ttk.Label(self,text = "Train Score, iter 2: " +str(calcScore(cff,X_train,y_train)))
##        label3.pack()
##        
##        cff.n_estimators = 30
##        predictions = cff.predict(X_test)
##        
##        label2 = ttk.Label(self,text = "Test Score, iter 2: " +str(calcScore(cff,X_test,y_test)))
##        label2.pack()
##
##        label3 = ttk.Label(self,text = "Train Score on training data minus outliers: " +str(calcScore(cff,x,y)))
##        label3.pack()

        label4 = ttk.Label(self,text = "Train Score on  ALL training data: " +str(calcScore(cff,xtrainFinal,ytrainFinal)))
        label4.pack()


        


        ##### WRITE OUT #####
        if choiceOfPrints:
            testXData = copy.deepcopy(testData)
            del testXData['Id']
                
            y_prediction=cff.predict(testXData)
            y_prediction = np.exp(y_prediction)

            file = open('AdaBoostNoOutliers.txt','w')
            file.write("Id,SalePrice\n")

            predictions_data = pandas.DataFrame(data = {'prediction':y_prediction})

##            print("Printed: AdaBoostNoOutliers")
            #print(predictions_data.shape)
            printPred = np.array(predictions_data)
            printId = np.array(testIds)
            for i in range(printId.size):
                file.write(str(printId[i])+","+str(printPred[i][0])+"\n")

            file.close()


class LinearModelWithOutliers(tk.Frame):
     
     def __init__(self, parent, controller):
            tk.Frame.__init__(self, parent)
            label = tk.Label(self, text="LinearRegression Output With Outliers", font=LARGE_FONT)
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
            X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.3)
            

                   
            cff.fit(X_train, y_train)
##            label1 = ttk.Label(self,text = "Train Score, iter 1: " +str(calcScore(cff,X_train,y_train)))
##            label1.pack()
##
##            
##            predictions = cff.predict(X_test)
##            label2 = ttk.Label(self,text = "Test Score, iter 1: " +str(calcScore(cff,X_test,y_test)))
##            label2.pack()
            
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

            label3 = ttk.Label(self,text = "Train Score on all training data: " +str(calcScore(cff,x,y)))
            label3.pack()

                              
      

            ##### WRITE OUT #####
            if choiceOfPrints:
                testXData = copy.deepcopy(testData)
                del testXData['Id']
                 
                y_prediction=cff.predict(testXData)
                y_prediction = np.exp(y_prediction)

                file = open('LinearModelWithOutliers.txt','w')
                file.write("Id,SalePrice\n")

                predictions_data = pandas.DataFrame(data = {'prediction':y_prediction})

##                print("Printed: LinearModelWithOutliers")
                
                #print(predictions_data.shape)
                printPred = np.array(predictions_data)
                printId = np.array(testIds)
                for i in range(printId.size):
                    file.write(str(printId[i])+","+str(printPred[i][0])+"\n")

                file.close()

class LinearModelNoOutliers(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="LinearRegression No Outliers", font=LARGE_FONT)
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
        
        X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.3)

        #print(y_train)
        
       
        
        cff.fit(X_train, y_train)
##        label1 = ttk.Label(self,text = "Train Score, iter 1: " +str(calcScore(cff,X_train,y_train)))
##        label1.pack()
##
##        
##        predictions = cff.predict(X_test)
##        
##        label2 = ttk.Label(self,text = "Test Score, iter 1: " +str(calcScore(cff,X_test,y_test)))
##        label2.pack()
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
##
##        label3 = ttk.Label(self,text = "Train Score on training data minus outliers: " +str(calcScore(cff,x,y)))
##        label3.pack()

        label4 = ttk.Label(self,text = "Train Score on  ALL training data: " +str(calcScore(cff,xtrainFinal,ytrainFinal)))
        label4.pack()


        ##### WRITE OUT #####
        if choiceOfPrints:
            testXData = copy.deepcopy(testData)
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
##            cff = linear_model.RidgeCV([0.1, 1.0, 10.0])
##            X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.3)
##            
##
##                   
##            cff.fit(X_train, y_train)
##            label1 = ttk.Label(self,text = "RidgeTrain Score: " +str(calcScore(cff,X_train,y_train)))
##            label1.pack()
##
##            
##            predictions = cff.predict(X_test)
##            label2 = ttk.Label(self,text = "Test Score: " +str(calcScore(cff,X_test,y_test)))
##            label2.pack()
##
##            label3 = ttk.Label(self,text = "Train Score on all training data: " +str(calcScore(cff,x,y)))
##            label3.pack()
##
##            if choiceOfPrints:
##                testXData = testData.copy()
##                del testXData['Id']
##                 
##                y_prediction=cff.predict(testXData)
##                y_prediction = np.exp(y_prediction)
##
##                file = open('RidgeWithOutliers.txt','w')
##                file.write("Id,SalePrice\n")
##
##                predictions_data = pandas.DataFrame(data = {'prediction':y_prediction})
##
####                print("Printed: RidgeCVWithOutliers")
##                
##                #print(predictions_data.shape)
##                printPred = np.array(predictions_data)
##                printId = np.array(testIds)
##                for i in range(printId.size):
##                    file.write(str(printId[i])+","+str(printPred[i][0])+"\n")
##
##                file.close()
##            
            cf = linear_model.RidgeCV([0.1, 1.0,  5.0, 7.5, 10.0, 20.0])

            
            X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.3)
            

                   
            cf.fit(X_train, y_train)
            label1 = ttk.Label(self,text = "RidgeCV([0.1, 1.0,  5.0, 7.5, 10.0, 20.0]) Train Score: " +str(calcScore(cf,X_train,y_train)))
            label1.pack()

            
            predictions = cf.predict(X_test)
            label2 = ttk.Label(self,text = "RidgeCV([0.1, 1.0,  5.0, 7.5, 10.0, 20.0]) Test Score: " +str(calcScore(cf,X_test,y_test)))
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

            label3 = ttk.Label(self,text = "Train Score on all training data: " +str(calcScore(cf,x,y)))
            label3.pack()

                    


            ##### WRITE OUT #####
            if choiceOfPrints:
                testXData = copy.deepcopy(testData)
                del testXData['Id']
                 
                y_prediction=cff.predict(testXData)
                y_prediction = np.exp(y_prediction)

                file = open('RidgeCVWithOutliers.txt','w')
                file.write("Id,SalePrice\n")

                predictions_data = pandas.DataFrame(data = {'prediction':y_prediction})

##                print("Printed: RidgeCVWithOutliers")
                
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

        cff = linear_model.RidgeCV([0.1, 1.0,  5.0, 7.5, 10.0, 20.0])
        
        X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.3)

        #print(y_train)
        
       
        
        cff.fit(X_train, y_train)
        label1 = ttk.Label(self,text = "RidgeCV([0.1, 1.0,  5.0, 7.5, 10.0, 20.0]) Train Score: " +str(calcScore(cff,X_train,y_train)))
        label1.pack()

        
        predictions = cff.predict(X_test)
        
        label2 = ttk.Label(self,text = "RidgeCV([0.1, 1.0,  5.0, 7.5, 10.0, 20.0]) Test Score: " +str(calcScore(cff,X_test,y_test)))
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

##        label3 = ttk.Label(self,text = "Train Score on training data minus outliers: " +str(calcScore(cff,x,y)))
##        label3.pack()

        label4 = ttk.Label(self,text = "Train Score on  ALL training data: " +str(calcScore(cff,xtrainFinal,ytrainFinal)))
        label4.pack()


        ##### WRITE OUT #####
        if choiceOfPrints:
            testXData = copy.deepcopy(testData)
            del toTest['Id'] 
                
            y_prediction=cff.predict(testXData)
            y_prediction = np.exp(y_prediction)

            file = open('RidgeNoOutliers.txt','w')
            file.write("Id,SalePrice\n")

            predictions_data = pandas.DataFrame(data = {'prediction':y_prediction})

##            print("Printed: RidgeNoOutliers")
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


            toTest = copy.deepcopy(testData)
            del toTest['Id'] 
            
            
            X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.3)
            #______________________________________________________________________________________________
            label = tk.Label(self, text="Lasso", font=LARGE_FONT)
            label.pack()
            label = tk.Label(self, text="Lasso(alpha = .10)",font=('Helvetica',14))
            label.pack()
            
            cff = linear_model.Lasso(alpha = .10)       
            cff.fit(X_train, y_train)
            
            label1 = ttk.Label(self,text = "Lasso(alpha = 0.10) Train Score: " +str(calcScore(cff,X_train,y_train)))
            label1.pack()

            
            predictions = cff.predict(toTest)
            label2 = ttk.Label(self,text = "Lasso(alpha = 0.10) Test Score: " +str(calcScore(cff,X_test,y_test)))
            label2.pack()

            label3 = ttk.Label(self,text = "Lasso(alpha = 0.10) Train Score on all training data: " +str(calcScore(cff,x,y)))
            label3.pack()
            if choiceOfPrints:
                              
                ##### WRITE OUT ##### 
                y_prediction=predictions
                y_prediction = np.exp(y_prediction)

                file = open('lassoOutliers.txt','w')
                file.write("Id,SalePrice\n")

                predictions_data = pandas.DataFrame(data = {'prediction':y_prediction})

##                print("Printed: LassoWithOutliers")
                
                #print(predictions_data.shape)
                printPred = np.array(predictions_data)
                printId = np.array(testIds)
                for i in range(printId.size):
                    file.write(str(printId[i])+","+str(printPred[i][0])+"\n")

                file.close()

            #______________________________________________________________________________________________
            lars = linear_model.LassoLars(alpha = .10)
            label = tk.Label(self, text="LassoLars", font=LARGE_FONT)
            label.pack()
            label = tk.Label(self, text="LassoLars(alpha = .10)",font=('Helvetica',14))
            label.pack()
            lars.fit(X_train, y_train)
            
            label1 = ttk.Label(self,text = "LassoLars(alpha = 0.10) Train Score: " +str(calcScore(lars,X_train,y_train)))
            label1.pack()

            
            predictions = lars.predict(toTest)
            label2 = ttk.Label(self,text = "LassoLars(alpha = 0.10) Test Score: " +str(calcScore(lars,X_test,y_test)))
            label2.pack()

            label3 = ttk.Label(self,text = "LassoLars(alpha = 0.10) Train Score on all training data: " +str(calcScore(lars,x,y)))
            label3.pack()
            if choiceOfPrints:
                              
                ##### WRITE OUT #####   
                 
                y_prediction=predictions
                y_prediction = np.exp(y_prediction)

                file = open('lassoLarsWOutliers.txt','w')
                file.write("Id,SalePrice\n")

                predictions_data = pandas.DataFrame(data = {'prediction':y_prediction})

##                print("Printed: LassoWithOutliers")
                
                #print(predictions_data.shape)
                printPred = np.array(predictions_data)
                printId = np.array(testIds)
                for i in range(printId.size):
                    file.write(str(printId[i])+","+str(printPred[i][0])+"\n")

                file.close()
            
            #______________________________________________________________________________________________
            label = tk.Label(self, text="LassoLarsIC", font=LARGE_FONT)
            label.pack()
            label = tk.Label(self, text="LassoLarsIC(criterion='bic')", font=('Helvetica',14))
            label.pack()
            
            larsIC_bic = linear_model.LassoLarsIC(criterion='bic')
            larsIC_bic.fit(X_train, y_train)

            label1 = ttk.Label(self,text = "larsIC_bic Train Score: " +str(calcScore(larsIC_bic,X_train,y_train)))
            label1.pack()

            
            predictions = larsIC_bic.predict(toTest)


            label2 = ttk.Label(self,text = "larsIC_bic Test Score: " +str(calcScore(larsIC_bic,X_test,y_test)))
            label2.pack()

            label3 = ttk.Label(self,text = "larsIC_bic Train Score on all training data: " +str(calcScore(larsIC_bic,x,y)))
            label3.pack()
            if choiceOfPrints:
                              
                ##### WRITE OUT #####    
                y_prediction=predictions
                y_prediction = np.exp(y_prediction)

                file = open('lassoLarsBICCvWOutliers.txt','w')
                file.write("Id,SalePrice\n")

                predictions_data = pandas.DataFrame(data = {'prediction':y_prediction})

##                print("Printed: LassoWithOutliers")
                
                #print(predictions_data.shape)
                printPred = np.array(predictions_data)
                printId = np.array(testIds)
                for i in range(printId.size):
                    file.write(str(printId[i])+","+str(printPred[i][0])+"\n")

                file.close()
            
            #______________________________________________________________________________________________

            label = tk.Label(self, text="LassoLarsIC", font=LARGE_FONT)
            label.pack()
            label = tk.Label(self, text="LassoLarsIC(criterion='aic')", font=('Helvetica',14))
            label.pack()
            
            larsIC_aic = linear_model.LassoLarsIC(criterion='aic')
            larsIC_aic.fit(X_train, y_train)

            label1 = ttk.Label(self,text = "larsIC_aic Train Score: " +str(calcScore(larsIC_aic,X_train,y_train)))
            label1.pack()

            
            predictions = larsIC_aic.predict(toTest)

            
            label2 = ttk.Label(self,text = "larsIC_aic Test Score: " +str(calcScore(larsIC_aic,X_test,y_test)))
            label2.pack()

            label3 = ttk.Label(self,text = "larsIC_aic Train Score on all training data: " +str(calcScore(larsIC_aic,x,y)))
            label3.pack()
            if choiceOfPrints:
                              
                ##### WRITE OUT #####    
                 
                y_prediction=predictions
                y_prediction = np.exp(y_prediction)

                file = open('lassoLarsAICCvWOutliers.txt','w')
                file.write("Id,SalePrice\n")

                predictions_data = pandas.DataFrame(data = {'prediction':y_prediction})

##                print("Printed: LassoWithOutliers")
                
                #print(predictions_data.shape)
                printPred = np.array(predictions_data)
                printId = np.array(testIds)
                for i in range(printId.size):
                    file.write(str(printId[i])+","+str(printPred[i][0])+"\n")

                file.close()

            #______________________________________________________________________________________________

            lassoCV = linear_model.LassoCV(cv = 20).fit(X_train, y_train)

            label = tk.Label(self, text="LassoCv", font=LARGE_FONT)
            label.pack()
            label = tk.Label(self, text="LassoCV(cv = 20)", font=('Helvetica',14))
            label.pack()
            label1 = ttk.Label(self,text = "lassoCV Train Score: " +str(calcScore(lassoCV,X_train,y_train)))
            label1.pack()

            predictions = lassoCV.predict(toTest)
            label2 = ttk.Label(self,text = "lassoCV Test Score: " +str(calcScore(lassoCV,X_test,y_test)))
            label2.pack()

            label3 = ttk.Label(self,text = "lassoCV Train Score on all training data: " +str(calcScore(lassoCV,x,y)))
            label3.pack()
            if choiceOfPrints:
                              
                ##### WRITE OUT #####    
                y_prediction=predictions
                y_prediction = np.exp(y_prediction)

                file = open('lassoCvWOutliers.txt','w')
                file.write("Id,SalePrice\n")

                predictions_data = pandas.DataFrame(data = {'prediction':y_prediction})

##                print("Printed: LassoWithOutliers")
                
                #print(predictions_data.shape)
                printPred = np.array(predictions_data)
                printId = np.array(testIds)
                for i in range(printId.size):
                    file.write(str(printId[i])+","+str(printPred[i][0])+"\n")

                file.close()


            #______________________________________________________________________________________________

            lassoLarsCv = linear_model.LassoLarsCV(cv = 20).fit(X_train, y_train)

            label = tk.Label(self, text="lassoLarsCv", font=LARGE_FONT)
            label.pack()
            label = tk.Label(self, text="LassoLarsCV(cv = 20)", font=('Helvetica',14))
            label.pack()

            
            predictions = lassoLarsCv.predict(toTest)

            label1 = ttk.Label(self,text = "lassoLarsCv Train Score: " +str(calcScore(lassoLarsCv,X_train,y_train)))
            label1.pack()
            
            label2 = ttk.Label(self,text = "lassoLarsCv Test Score: " +str(calcScore(lassoLarsCv,X_test,y_test)))
            label2.pack()

            label3 = ttk.Label(self,text = "lassoLarsCv Train Score on all training data: " +str(calcScore(lassoLarsCv,x,y)))
            label3.pack()
            if choiceOfPrints:
                              
                ##### WRITE OUT #####   
                 
                y_prediction=predictions
                y_prediction = np.exp(y_prediction)

                file = open('lassoLarsCvWOutliers.txt','w')
                file.write("Id,SalePrice\n")

                predictions_data = pandas.DataFrame(data = {'prediction':y_prediction})

##                print("Printed: LassoWithOutliers")
                
                #print(predictions_data.shape)
                printPred = np.array(predictions_data)
                printId = np.array(testIds)
                for i in range(printId.size):
                    file.write(str(printId[i])+","+str(printPred[i][0])+"\n")

                file.close()
            label = tk.Label(self, text="LassoPipeline", font=LARGE_FONT)
            label.pack()
            label = tk.Label(self, text="make_pipeline(RobustScaler(), linear_model.Lasso(alpha = ___, random_state=123))", font=('Helvetica',14))
            label.pack()
            #################!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            lassoPipeline = make_pipeline(RobustScaler(), linear_model.Lasso(alpha = 5, random_state=123)).fit(X_train, y_train)

            label1 = ttk.Label(self,text = "lassoPipeline 5 Train Score: " +str(calcScore(lassoPipeline,X_train,y_train)))
            label1.pack()
            
            label2 = ttk.Label(self,text = "lassoPipeline 5 Test Score: " +str(calcScore(lassoPipeline,X_test,y_test)))
            label2.pack()

            label3 = ttk.Label(self,text = "lassoPipeline 5 Train Score on all training data: " +str(calcScore(lassoPipeline,x,y)))
            label3.pack()
            if choiceOfPrints:
                              
                ##### WRITE OUT #####   
                 
                y_prediction=lassoPipeline.predict(toTest)
                y_prediction = np.exp(y_prediction)

                file = open('LassoPipeline-5-WOutliers.txt','w')
                file.write("Id,SalePrice\n")

                predictions_data = pandas.DataFrame(data = {'prediction':y_prediction})

##                print("Printed: LassoWithOutliers")
                
                #print(predictions_data.shape)
                printPred = np.array(predictions_data)
                printId = np.array(testIds)
                for i in range(printId.size):
                    file.write(str(printId[i])+","+str(printPred[i][0])+"\n")

                file.close()
#################!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            lassoPipeline = make_pipeline(RobustScaler(), linear_model.Lasso(alpha = 0.5, random_state=123)).fit(X_train, y_train)
            label1 = ttk.Label(self,text = "lassoPipeline .5 Train Score: " +str(calcScore(lassoPipeline,X_train,y_train)))
            label1.pack()
            
            label2 = ttk.Label(self,text = "lassoPipeline .5 Test Score: " +str(calcScore(lassoPipeline,X_test,y_test)))
            label2.pack()

            label3 = ttk.Label(self,text = "lassoPipeline .5 Train Score on all training data: " +str(calcScore(lassoPipeline,x,y)))
            label3.pack()
            if choiceOfPrints:
                              
                ##### WRITE OUT #####  
                 
                y_prediction=lassoPipeline.predict(toTest)
                y_prediction = np.exp(y_prediction)

                file = open('LassoPipeline-.5-Outliers.txt','w')
                file.write("Id,SalePrice\n")

                predictions_data = pandas.DataFrame(data = {'prediction':y_prediction})

##                print("Printed: LassoWithOutliers")
                
                #print(predictions_data.shape)
                printPred = np.array(predictions_data)
                printId = np.array(testIds)
                for i in range(printId.size):
                    file.write(str(printId[i])+","+str(printPred[i][0])+"\n")

                file.close()
#################!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            lassoPipeline = make_pipeline(RobustScaler(), linear_model.Lasso(alpha = 0.05, random_state=123)).fit(X_train, y_train)

            label1 = ttk.Label(self,text = "lassoPipeline .05 Train Score: " +str(calcScore(lassoPipeline,X_train,y_train)))
            label1.pack()
            
            label2 = ttk.Label(self,text = "lassoPipeline .05 Test Score: " +str(calcScore(lassoPipeline,X_test,y_test)))
            label2.pack()
            
            label3 = ttk.Label(self,text = "lassoPipeline .05 Train Score on all training data: " +str(calcScore(lassoPipeline,x,y)))
            label3.pack()
            if choiceOfPrints:
                              
                ##### WRITE OUT ##### 
                 
                y_prediction=lassoPipeline.predict(toTest)
                y_prediction = np.exp(y_prediction)

                file = open('LassoPipeline-05-Outliers.txt','w')
                file.write("Id,SalePrice\n")

                predictions_data = pandas.DataFrame(data = {'prediction':y_prediction})

##                print("Printed: LassoWithOutliers")
                
                #print(predictions_data.shape)
                printPred = np.array(predictions_data)
                printId = np.array(testIds)
                for i in range(printId.size):
                    file.write(str(printId[i])+","+str(printPred[i][0])+"\n")

                file.close()

            #################!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            l
            
            lassoPipeline = make_pipeline(RobustScaler(), linear_model.Lasso(alpha = 0.005, random_state=123)).fit(X_train, y_train)

            label1 = ttk.Label(self,text = "lassoPipeline .005 Train Score: " +str(calcScore(lassoPipeline,X_train,y_train)))
            label1.pack()
            
            label2 = ttk.Label(self,text = "lassoPipeline .005 Test Score: " +str(calcScore(lassoPipeline,X_test,y_test)))
            label2.pack()
            label3 = ttk.Label(self,text = "lassoPipeline .005 Train Score on all training data: " +str(calcScore(lassoPipeline,x,y)))
            label3.pack()
            if choiceOfPrints:
                              
                ##### WRITE OUT #####    
                 
                y_prediction=lassoPipeline.predict(toTest)
                y_prediction = np.exp(y_prediction)

                file = open('LassoPipeline-005-Outliers.txt','w')
                file.write("Id,SalePrice\n")

                predictions_data = pandas.DataFrame(data = {'prediction':y_prediction})

##                print("Printed: LassoWithOutliers")
                
                #print(predictions_data.shape)
                printPred = np.array(predictions_data)
                printId = np.array(testIds)
                for i in range(printId.size):
                    file.write(str(printId[i])+","+str(printPred[i][0])+"\n")

                file.close()

            #################!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            

                
            lassoPipeline = make_pipeline(RobustScaler(), linear_model.Lasso(alpha = 0.0005, random_state=123)).fit(X_train, y_train)

            label1 = ttk.Label(self,text = "lassoPipeline .0005 Train Score: " +str(calcScore(lassoPipeline,X_train,y_train)))
            label1.pack()
            
            label2 = ttk.Label(self,text = "lassoPipeline .0005 Test Score: " +str(calcScore(lassoPipeline,X_test,y_test)))
            label2.pack()
            label3 = ttk.Label(self,text = "lassoPipeline .0005 Train Score on all training data: " +str(calcScore(lassoPipeline,x,y)))
            label3.pack()




            
            
            if choiceOfPrints:
                              
                ##### WRITE OUT #####   
                 
                y_prediction=lassoPipeline.predict(toTest)
                y_prediction = np.exp(y_prediction)

                file = open('LassoPipeline-0005-Outliers.txt','w')
                file.write("Id,SalePrice\n")

                predictions_data = pandas.DataFrame(data = {'prediction':y_prediction})

##                print("Printed: LassoWithOutliers")
                
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



        toTest = copy.deepcopy(testData)
        del toTest['Id']    


        #print(y_train)
        
       
            
            
        X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.3)
        #______________________________________________________________________________________________
        label = tk.Label(self, text="Lasso", font=LARGE_FONT)
        label.pack()
        label = tk.Label(self, text="Lasso(alpha = 10.0)", font=('Helvetica',14))
        label.pack()

        cff = linear_model.Lasso(alpha = 10.0)       
        cff.fit(X_train, y_train)
        
        label1 = ttk.Label(self,text = "Lasso Train Score: " +str(calcScore(cff,X_train,y_train)))
        label1.pack()

        
        predictions = cff.predict(toTest)
        label2 = ttk.Label(self,text = "Lasso Test Score: " +str(calcScore(cff,X_test,y_test)))
        label2.pack()

        label3 = ttk.Label(self,text = "Lasso Train Score on all training data: " +str(calcScore(cff,x,y)))
        label3.pack()
        ##### WRITE OUT #####
        if choiceOfPrints:
                
            y_prediction=predictions

            file = open('lasso-NoOutliers.txt','w')
            file.write("Id,SalePrice\n")

            predictions_data = pandas.DataFrame(data = {'prediction':y_prediction})

##            print("Printed: LassoNoOutliers")
            #print(predictions_data.shape)
            printPred = np.array(predictions_data)
            printId = np.array(testIds)
            for i in range(printId.size):
                file.write(str(printId[i])+","+str(printPred[i][0])+"\n")

            file.close()

        #______________________________________________________________________________________________
        label = tk.Label(self, text="LassoLars", font=LARGE_FONT)
        label.pack()
        label = tk.Label(self, text="LassoLars(alpha = 10.0)", font=('Helvetica',14))
        label.pack()
        
        lars = linear_model.LassoLars(alpha = 10.0)       
        lars.fit(X_train, y_train)
        
        label1 = ttk.Label(self,text = "LassoLars Train Score: " +str(calcScore(lars,X_train,y_train)))
        label1.pack()

        
        predictions = lars.predict(toTest)
        label2 = ttk.Label(self,text = "LassoLars Test Score: " +str(calcScore(lars,X_test,y_test)))
        label2.pack()

        label3 = ttk.Label(self,text = "LassoLars Train Score on all training data: " +str(calcScore(lars,x,y)))
        label3.pack()
        ##### WRITE OUT #####
        if choiceOfPrints:
                
            y_prediction=predictions

            file = open('LassoLars-NoOutliers.txt','w')
            file.write("Id,SalePrice\n")

            predictions_data = pandas.DataFrame(data = {'prediction':y_prediction})

##            print("Printed: LassoNoOutliers")
            #print(predictions_data.shape)
            printPred = np.array(predictions_data)
            printId = np.array(testIds)
            for i in range(printId.size):
                file.write(str(printId[i])+","+str(printPred[i][0])+"\n")

            file.close()
        
        #______________________________________________________________________________________________

        label = tk.Label(self, text="LassoLarsIC", font=LARGE_FONT)
        label.pack()
        label = tk.Label(self, text="LassoLarsIC(criterion='bic')", font=('Helvetica',14))
        label.pack()

        larsIC_bic = linear_model.LassoLarsIC(criterion='bic')
        larsIC_bic.fit(X_train, y_train)

        label1 = ttk.Label(self,text = "larsIC_bic Train Score: " +str(calcScore(larsIC_bic,X_train,y_train)))
        label1.pack()

        
        predictions = larsIC_bic.predict(toTest)


        label2 = ttk.Label(self,text = "larsIC_bic Test Score: " +str(calcScore(larsIC_bic,X_test,y_test)))
        label2.pack()

        label3 = ttk.Label(self,text = "larsIC_bic Train Score on all training data: " +str(calcScore(larsIC_bic,x,y)))
        label3.pack()

        ##### WRITE OUT #####
        if choiceOfPrints:
                
            y_prediction=predictions

            file = open('larsIC_bic-NoOutliers.txt','w')
            file.write("Id,SalePrice\n")

            predictions_data = pandas.DataFrame(data = {'prediction':y_prediction})

##            print("Printed: LassoNoOutliers")
            #print(predictions_data.shape)
            printPred = np.array(predictions_data)
            printId = np.array(testIds)
            for i in range(printId.size):
                file.write(str(printId[i])+","+str(printPred[i][0])+"\n")

            file.close()
            
        #______________________________________________________________________________________________
        label = tk.Label(self, text="LassoLarsIC", font=LARGE_FONT)
        label.pack()
        label = tk.Label(self, text="LassoLarsIC(criterion='aic')", font=('Helvetica',14))
        label.pack()
        
        larsIC_aic = linear_model.LassoLarsIC(criterion='aic')
        larsIC_aic.fit(X_train, y_train)

        label1 = ttk.Label(self,text = "larsIC_aic Train Score: " +str(calcScore(larsIC_aic,X_train,y_train)))
        label1.pack()

        
        predictions = larsIC_aic.predict(toTest)

        
        label2 = ttk.Label(self,text = "larsIC_aic Test Score: " +str(calcScore(larsIC_aic,X_test,y_test)))
        label2.pack()

        label3 = ttk.Label(self,text = "larsIC_aic Train Score on all training data: " +str(calcScore(larsIC_aic,x,y)))
        label3.pack()

        ##### WRITE OUT #####
        if choiceOfPrints:
                
            y_prediction=predictions

            file = open('larsIC_aic-NoOutliers.txt','w')
            file.write("Id,SalePrice\n")

            predictions_data = pandas.DataFrame(data = {'prediction':y_prediction})

##            print("Printed: LassoNoOutliers")
            #print(predictions_data.shape)
            printPred = np.array(predictions_data)
            printId = np.array(testIds)
            for i in range(printId.size):
                file.write(str(printId[i])+","+str(printPred[i][0])+"\n")

            file.close()

        #______________________________________________________________________________________________


        label = tk.Label(self, text="LassoCV", font=LARGE_FONT)
        label.pack()
        label = tk.Label(self, text="Lasso(cv = 20)", font=('Helvetica',14))
        label.pack()
        
        lassoCv = linear_model.LassoCV(cv = 20).fit(X_train, y_train)

        label1 = ttk.Label(self,text = "lassoCV Train Score: " +str(calcScore(lassoCv,X_train,y_train)))
        label1.pack()

        
        predictions = lassoCv.predict(toTest)

        
        label2 = ttk.Label(self,text = "lassoCV Test Score: " +str(calcScore(lassoCv,X_test,y_test)))
        label2.pack()

        label3 = ttk.Label(self,text = "lassoCV Train Score on all training data: " +str(calcScore(lassoCv,x,y)))
        label3.pack()


        ##### WRITE OUT #####
        if choiceOfPrints:
                
            y_prediction=predictions

            file = open('lassoCV-NoOutliers.txt','w')
            file.write("Id,SalePrice\n")

            predictions_data = pandas.DataFrame(data = {'prediction':y_prediction})

##            print("Printed: LassoNoOutliers")
            #print(predictions_data.shape)
            printPred = np.array(predictions_data)
            printId = np.array(testIds)
            for i in range(printId.size):
                file.write(str(printId[i])+","+str(printPred[i][0])+"\n")

            file.close()


        #______________________________________________________________________________________________


        label = tk.Label(self, text="LassoLars", font=LARGE_FONT)
        label.pack()
        label = tk.Label(self, text="LassoLarsCV(cv = 20)", font=('Helvetica',14))
        label.pack()
        lassoLarsCv = linear_model.LassoLarsCV(cv = 20).fit(X_train, y_train)

        
        
        predictions = lassoLarsCv.predict(toTest)
        
        label1 = ttk.Label(self,text = "lassoLarsCv Train Score: " +str(calcScore(lassoLarsCv,X_train,y_train)))
        label1.pack()

        
        label2 = ttk.Label(self,text = "lassoLarsCv Test Score: " +str(calcScore(lassoLarsCv,X_test,y_test)))
        label2.pack()

        label3 = ttk.Label(self,text = "lassoLarsCv Train Score on all training data: " +str(calcScore(lassoLarsCv,x,y)))
        label3.pack()

        ##### WRITE OUT #####
        if choiceOfPrints:
                
            y_prediction=predictions

            file = open('lassoLarsCv-NoOutliers.txt','w')
            file.write("Id,SalePrice\n")

            predictions_data = pandas.DataFrame(data = {'prediction':y_prediction})

##            print("Printed: LassoNoOutliers")
            #print(predictions_data.shape)
            printPred = np.array(predictions_data)
            printId = np.array(testIds)
            for i in range(printId.size):
                file.write(str(printId[i])+","+str(printPred[i][0])+"\n")

            file.close()

         #______________________________________________________________________________________________

        label = tk.Label(self, text="LassoPipeline", font=LARGE_FONT)
        label.pack()
        label = tk.Label(self, text="make_pipeline(RobustScaler(), linear_model.Lasso(alpha = 0.5, random_state=123))", font=('Helvetica',14))
        label.pack()
        lassoPipeline = make_pipeline(RobustScaler(), linear_model.Lasso(alpha = 0.5, random_state=123)).fit(X_train, y_train)

        label1 = ttk.Label(self,text = "lassoPipeline Train Score: " +str(calcScore(lassoPipeline,X_train,y_train)))
        label1.pack()

        
        label2 = ttk.Label(self,text = "lassoPipeline Test Score: " +str(calcScore(lassoPipeline,X_test,y_test)))
        label2.pack()

        label3 = ttk.Label(self,text = "lassoPipeline Train Score on all training data: " +str(calcScore(lassoPipeline,x,y)))
        label3.pack()

        predictions = lassoPipeline.predict(toTest)
        
        ##### WRITE OUT #####
        if choiceOfPrints:
                
            y_prediction=predictions

            file = open('LassoPipeLine-NoOutliers.txt','w')
            file.write("Id,SalePrice\n")

            predictions_data = pandas.DataFrame(data = {'prediction':y_prediction})

##            print("Printed: LassoNoOutliers")
            #print(predictions_data.shape)
            printPred = np.array(predictions_data)
            printId = np.array(testIds)
            for i in range(printId.size):
                file.write(str(printId[i])+","+str(printPred[i][0])+"\n")

            file.close()

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
            

            toTest = copy.deepcopy(testData)
            del toTest['Id']            

            
            X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.3)



            

            label = tk.Label(self, text="RANSAC: LinearModel Output With Outliers", font=LARGE_FONT)
            label.pack(pady=10,padx=10)

            
            ransacL = linear_model.RANSACRegressor(stop_score = .94)
            
            ransacL.fit(x,y)

            
            label1 = ttk.Label(self,text = "Train Score: " +str(calcScore(ransacL,X_train,y_train)))
            label1.pack()

            
            predictions = ransacL.predict(toTest)
            label2 = ttk.Label(self,text = "Test Score: " +str(calcScore(ransacL,X_test,y_test)))
            label2.pack()
            
            label3 = ttk.Label(self,text = "Train Score on all training data: " +str(calcScore(ransacL,x,y)))
            label3.pack()

            ##### WRITE OUT #####
            if choiceOfPrints:
                    
                y_prediction=predictions

                file = open('LassoLars-WOutliers.txt','w')
                file.write("Id,SalePrice\n")

                predictions_data = pandas.DataFrame(data = {'prediction':y_prediction})

    ##            print("Printed: LassoNoOutliers")
                #print(predictions_data.shape)
                printPred = np.array(predictions_data)
                printId = np.array(testIds)
                for i in range(printId.size):
                    file.write(str(printId[i])+","+str(printPred[i][0])+"\n")

                file.close()


            #_____________________________________________________________________________________

            label = tk.Label(self, text="RANSAC: RandomForestRegressor Output With Outliers", font=LARGE_FONT)
            label.pack(pady=10,padx=10)
            
            ransacRR = linear_model.RANSACRegressor(base_estimator= RandomForestRegressor(), stop_score = .94)
            
            ransacRR.fit(x,y)

            
            label1 = ttk.Label(self,text = "Train Score: " +str(calcScore(ransacRR,X_train,y_train)))
            label1.pack()

            
            predictions = ransacRR.predict(toTest)
            label2 = ttk.Label(self,text = "Test Score: " +str(calcScore(ransacRR,X_test,y_test)))
            label2.pack()
            
            label3 = ttk.Label(self,text = "Train Score on all training data: " +str(calcScore(ransacRR,x,y)))
            label3.pack()

             ##### WRITE OUT #####
            if choiceOfPrints:
                    
                y_prediction=predictions

                file = open('Ransac-RandomForest-WOutliers.txt','w')
                file.write("Id,SalePrice\n")

                predictions_data = pandas.DataFrame(data = {'prediction':y_prediction})

    ##            print("Printed: LassoNoOutliers")
                #print(predictions_data.shape)
                printPred = np.array(predictions_data)
                printId = np.array(testIds)
                for i in range(printId.size):
                    file.write(str(printId[i])+","+str(printPred[i][0])+"\n")

                file.close()


            #_____________________________________________________________________________________

            label = tk.Label(self, text="RANSAC: RANSAC RandomForestRegressor Output With Outliers", font=LARGE_FONT)
            label.pack(pady=10,padx=10)
            
            ransacRR = linear_model.RANSACRegressor(linear_model.RANSACRegressor(base_estimator= RandomForestRegressor()), stop_score = .94)
            
            ransacRR.fit(x,y)

            
            label1 = ttk.Label(self,text = "Train Score: " +str(calcScore(ransacRR,X_train,y_train)))
            label1.pack()

            
            predictions = ransacRR.predict(toTest)
            label2 = ttk.Label(self,text = "Test Score: " +str(calcScore(ransacRR,X_test,y_test)))
            label2.pack()
            
            label3 = ttk.Label(self,text = "Train Score on all training data: " +str(calcScore(ransacRR,x,y)))
            label3.pack()

             ##### WRITE OUT #####
            if choiceOfPrints:
                    
                y_prediction=predictions

                file = open('Ransac_of_Ransac_of_RandomForest-WOutliers.txt','w')
                file.write("Id,SalePrice\n")

                predictions_data = pandas.DataFrame(data = {'prediction':y_prediction})

    ##            print("Printed: LassoNoOutliers")
                #print(predictions_data.shape)
                printPred = np.array(predictions_data)
                printId = np.array(testIds)
                for i in range(printId.size):
                    file.write(str(printId[i])+","+str(printPred[i][0])+"\n")

                file.close()




                


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


            toTest = copy.deepcopy(testData)
            del toTest['Id']



            
            X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.3)
            




            label = tk.Label(self, text="RANSAC: LinearModel Output No Outliers", font=LARGE_FONT)
            label.pack(pady=10,padx=10)


           
            ransacL = linear_model.RANSACRegressor(stop_score = .94)
            
            ransacL.fit(x,y)

            
            label1 = ttk.Label(self,text = "Train Score: " +str(calcScore(ransacL,X_train,y_train)))
            label1.pack()

            
            predictions = ransacL.predict(toTest)
            label2 = ttk.Label(self,text = "Test Score: " +str(calcScore(ransacL,X_test,y_test)))
            label2.pack()
            
            label3 = ttk.Label(self,text = "Train Score on all training data: " +str(calcScore(ransacL,xtrainFinal,ytrainFinal)))
            label3.pack()

             ##### WRITE OUT #####
            if choiceOfPrints:
                    
                y_prediction=predictions

                file = open('Ransac_Linear-NoOutliers.txt','w')
                file.write("Id,SalePrice\n")

                predictions_data = pandas.DataFrame(data = {'prediction':y_prediction})

    ##            print("Printed: LassoNoOutliers")
                #print(predictions_data.shape)
                printPred = np.array(predictions_data)
                printId = np.array(testIds)
                for i in range(printId.size):
                    file.write(str(printId[i])+","+str(printPred[i][0])+"\n")

                file.close()


            #_____________________________________________________________________________________

            label = tk.Label(self, text="RANSAC: RandomForestRegressor Output No Outliers", font=LARGE_FONT)
            label.pack(pady=10,padx=10)
            
            ransacRR = linear_model.RANSACRegressor(base_estimator= RandomForestRegressor(), stop_score = .94)
            
            ransacRR.fit(x,y)

            
            label1 = ttk.Label(self,text = "Train Score: " +str(calcScore(ransacRR,X_train,y_train)))
            label1.pack()

            
            predictions = ransacRR.predict(toTest)
            label2 = ttk.Label(self,text = "Test Score: " +str(calcScore(ransacRR,X_test,y_test)))
            label2.pack()
            
            label3 = ttk.Label(self,text = "Train Score on all training data: " +str(calcScore(ransacRR,xtrainFinal,ytrainFinal)))
            label3.pack()

            ##### WRITE OUT #####
            if choiceOfPrints:
                    
                y_prediction=predictions

                file = open('Ransac_RandomForest-NoOutliers.txt','w')
                file.write("Id,SalePrice\n")

                predictions_data = pandas.DataFrame(data = {'prediction':y_prediction})

    ##            print("Printed: LassoNoOutliers")
                #print(predictions_data.shape)
                printPred = np.array(predictions_data)
                printId = np.array(testIds)
                for i in range(printId.size):
                    file.write(str(printId[i])+","+str(printPred[i][0])+"\n")

                file.close()


            #_____________________________________________________________________________________

            label = tk.Label(self, text="RANSAC: RANSAC RandomForestRegressor Output No Outliers", font=LARGE_FONT)
            label.pack(pady=10,padx=10)
            
            ransacRR = linear_model.RANSACRegressor(linear_model.RANSACRegressor(base_estimator= RandomForestRegressor()), stop_score = .94)
            
            ransacRR.fit(x,y)

            
            label1 = ttk.Label(self,text = "Train Score: " +str(calcScore(ransacRR,X_train,y_train)))
            label1.pack()

            
            predictions = ransacRR.predict(toTest)
            label2 = ttk.Label(self,text = "Test Score: " +str(calcScore(ransacRR,X_test,y_test)))
            label2.pack()
            
            label3 = ttk.Label(self,text = "Train Score on all training data: " +str(calcScore(ransacRR,xtrainFinal,ytrainFinal)))
            label3.pack()
            ##### WRITE OUT #####
            if choiceOfPrints:
                    
                y_prediction=predictions

                file = open('Ransac_of_Ransac_of_RandomForest-NoOutliers.txt','w')
                file.write("Id,SalePrice\n")

                predictions_data = pandas.DataFrame(data = {'prediction':y_prediction})

    ##            print("Printed: LassoNoOutliers")
                #print(predictions_data.shape)
                printPred = np.array(predictions_data)
                printId = np.array(testIds)
                for i in range(printId.size):
                    file.write(str(printId[i])+","+str(printPred[i][0])+"\n")

                file.close()


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
            

                
            testXData = copy.deepcopy(testData)
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

            
            X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.3)

            toTrainX = X_train
            toTrainY = y_train

            #toTest = testXData
            toTest = testXData

            robAB = make_pipeline(RobustScaler(), AdaBoostRegressor(base_estimator = ExtraTreeRegressor(), n_estimators = 2000)).fit(toTrainX, toTrainY)
            Adaprediction = robAB.predict(toTest)

            label1 = ttk.Label(self,text = "Robust AdaBoost",font = LARGE_FONT)
            label1.pack()
            label1 = ttk.Label(self,text = "make_pipeline(RobustScaler(), AdaBoostRegressor(base_estimator = ExtraTreeRegressor(), n_estimators = 2000)): " )
            label1.pack()
            
            label1 = ttk.Label(self,text = "Train Score: " +str(calcScore(robAB,X_train,y_train)))
            label1.pack()
            
            label2 = ttk.Label(self,text = "Test Score: " +str(calcScore(robAB,X_test,y_test)))
            label2.pack()
            
            label4 = ttk.Label(self,text = "ADA Train Score on  ALL training data: " +str(calcScore(robAB,x,y)))
            label4.pack()


            if choiceOfPrints:
                 
                y_prediction=Adaprediction
                y_prediction = np.exp(y_prediction)

                file = open('AdaBoostRobust-WOutliers.txt','w')
                file.write("Id,SalePrice\n")

                predictions_data = pandas.DataFrame(data = {'prediction':y_prediction})

##                print("Printed: lATEST")
                
                #print(predictions_data.shape)
                printPred = np.array(predictions_data)
                printId = np.array(testIds)
                for i in range(printId.size):
                    file.write(str(printId[i])+","+str(printPred[i][0])+"\n")

                file.close()
            
            rff = rf.RandomForestRegressor(n_estimators = 3000, random_state = 42,oob_score =True, max_features = "auto", warm_start=True).fit(toTrainX, toTrainY)      
            label1 = ttk.Label(self,text = "Random Forest",font = LARGE_FONT)
            label1.pack()
            RFpredictions = rff.predict(toTest)
            
            label1 = ttk.Label(self,text = "RandomForestRegressor(n_estimators = 1000, random_state = 42,oob_score =True, max_features = \"auto\", warm_start=True): " )
            label1.pack()
            
            label1 = ttk.Label(self,text = "Train Score: " +str(calcScore(rff,X_train,y_train)))
            label1.pack()
            
            label2 = ttk.Label(self,text = "Test Score: " +str(calcScore(rff,X_test,y_test)))
            label2.pack()
            
            label3 = ttk.Label(self,text = "RF Train Score on all training data: " +str(calcScore(rff,x,y)))
            label3.pack()
            
            if choiceOfPrints:
                 
                y_prediction=RFpredictions
                y_prediction = np.exp(y_prediction)

                file = open('RandomForest2-WOutliers.txt','w')
                file.write("Id,SalePrice\n")

                predictions_data = pandas.DataFrame(data = {'prediction':y_prediction})

##                print("Printed: lATEST")
                
                #print(predictions_data.shape)
                printPred = np.array(predictions_data)
                printId = np.array(testIds)
                for i in range(printId.size):
                    file.write(str(printId[i])+","+str(printPred[i][0])+"\n")

                file.close()

            eNetRob = make_pipeline(RobustScaler(), linear_model.ElasticNet(alpha = 0.0005, l1_ratio = .9, fit_intercept = True)).fit(toTrainX, toTrainY)
            eNetpredictions = eNetRob.predict(toTest)

            label1 = ttk.Label(self,text = "Robust Elastic Net",font = LARGE_FONT)
            label1.pack()

            label1 = ttk.Label(self,text = "make_pipeline(RobustScaler(), linear_model.ElasticNet(alpha = 0.0005, l1_ratio = .9, fit_intercept = True)): " )
            label1.pack()
            
            label1 = ttk.Label(self,text = "Train Score: " +str(calcScore(eNetRob,X_train,y_train)))
            label1.pack()
            
            label2 = ttk.Label(self,text = "Test Score: " +str(calcScore(eNetRob,X_test,y_test)))
            label2.pack()
            label3 = ttk.Label(self,text = "ENET Train Score on all training data: " +str(calcScore(eNetRob,x,y)))
            label3.pack()


            if choiceOfPrints:
                 
                y_prediction=kernRgpredictions
                y_prediction = np.exp(y_prediction)

                file = open('ElasticNetRobust-WOutliers.txt','w')
                file.write("Id,SalePrice\n")

                predictions_data = pandas.DataFrame(data = {'prediction':y_prediction})

##                print("Printed: lATEST")
                
                #print(predictions_data.shape)
                printPred = np.array(predictions_data)
                printId = np.array(testIds)
                for i in range(printId.size):
                    file.write(str(printId[i])+","+str(printPred[i][0])+"\n")

                file.close()

            kernRg = KernelRidge(alpha = 0.5, kernel = 'polynomial').fit(toTrainX, toTrainY)
            kernRgpredictions = eNetRob.predict(toTest)
            label1 = ttk.Label(self,text = "KernelRidge",font = LARGE_FONT)
            label1.pack()

            label1 = ttk.Label(self,text = "KernelRidge(alpha = 0.5, kernel = 'polynomial'): " )
            label1.pack()
            
            label1 = ttk.Label(self,text = "Train Score: " +str(calcScore(kernRg,X_train,y_train)))
            label1.pack()
            
            label2 = ttk.Label(self,text = "Test Score: " +str(calcScore(kernRg,X_test,y_test)))
            label2.pack()

            label3 = ttk.Label(self,text = "Kern Train Score on all training data: " +str(calcScore(kernRg,x,y)))
            label3.pack()

            if choiceOfPrints:
                 
                y_prediction=kernRgpredictions
                y_prediction = np.exp(y_prediction)

                file = open('KernelRidge-WOutliers.txt','w')
                file.write("Id,SalePrice\n")

                predictions_data = pandas.DataFrame(data = {'prediction':y_prediction})

##                print("Printed: lATEST")
                
                #print(predictions_data.shape)
                printPred = np.array(predictions_data)
                printId = np.array(testIds)
                for i in range(printId.size):
                    file.write(str(printId[i])+","+str(printPred[i][0])+"\n")

                file.close()

                

            gradBoost = GradientBoostingRegressor(loss = 'huber', learning_rate = 0.06, n_estimators = 4000, max_depth = 4, min_samples_split = 10, min_samples_leaf = 15).fit(toTrainX, toTrainY)
            gradBoostPred = eNetRob.predict(toTest)

            label1 = ttk.Label(self,text = "GradientBoostingRegressor",font = LARGE_FONT)
            label1.pack()
            label1 = ttk.Label(self,text = "GradientBoostingRegressor(loss = 'huber', learning_rate = 0.06, n_estimators = 4000, max_depth = 4, min_samples_split = 10, min_samples_leaf = 15): " )
            label1.pack()
            
            label1 = ttk.Label(self,text = "Train Score: " +str(calcScore(gradBoost,X_train,y_train)))
            label1.pack()
            
            label2 = ttk.Label(self,text = "Test Score: " +str(calcScore(gradBoost,X_test,y_test)))
            label2.pack()

            label3 = ttk.Label(self,text = "Gradient Boost Train Score on all training data: " +str(calcScore(gradBoost,x,y)))
            label3.pack()

            if choiceOfPrints:
                 
                y_prediction=gradBoostPred
                y_prediction = np.exp(y_prediction)

                file = open('GradientBoostingRegressor-WOutliers.txt','w')
                file.write("Id,SalePrice\n")

                predictions_data = pandas.DataFrame(data = {'prediction':y_prediction})

##                print("Printed: lATEST")
                
                #print(predictions_data.shape)
                printPred = np.array(predictions_data)
                printId = np.array(testIds)
                for i in range(printId.size):
                    file.write(str(printId[i])+","+str(printPred[i][0])+"\n")

                file.close()

            
            
            
            label1 = ttk.Label(self,text = "Average of [AdaBoost,RandomForest, elasticNet, kernelRidge, GradientBoostingRegressor]: " , font = LARGE_FONT)
            label1.pack()

            

            averaged = np.column_stack([Adaprediction,RFpredictions,eNetpredictions,kernRgpredictions, gradBoostPred])
            averaged = np.mean(averaged, axis = 1)


            
            label1 = ttk.Label(self,text = "Train Score: " +str(sqrt(mean_squared_error(y_train, averaged))))
            label1.pack()
            
            label2 = ttk.Label(self,text = "Test Score: " +str(sqrt(mean_squared_error(y_test, averaged))))
            label2.pack()

            label3 = ttk.Label(self,text = "Averaged Train Score on all training data: " +str(sqrt(mean_squared_error(y, averaged))))
            label3.pack()

            if choiceOfPrints:
                 
                y_prediction=averaged
                y_prediction = np.exp(y_prediction)

                file = open('Averaged_WOutliers.txt','w')
                file.write("Id,SalePrice\n")

                predictions_data = pandas.DataFrame(data = {'prediction':y_prediction})

##                print("Printed: lATEST")
                
                #print(predictions_data.shape)
                printPred = np.array(predictions_data)
                printId = np.array(testIds)
                for i in range(printId.size):
                    file.write(str(printId[i])+","+str(printPred[i][0])+"\n")

                file.close()
            
            label1 = ttk.Label(self,text = "Averaged With Modifications: averaged*.2+Adaprediction*.4+gradBoostPred*.2+RFpredictions*.2: ", font = LARGE_FONT )
            label1.pack()
           
            newAveraged = averaged*.2+Adaprediction*.4+gradBoostPred*.2+RFpredictions*.2

            label1 = ttk.Label(self,text = "Train Score: " +str(sqrt(mean_squared_error(y_train, newAveraged))))
            label1.pack()
            
            label2 = ttk.Label(self,text = "Test Score: " +str(sqrt(mean_squared_error(y_test, newAveraged))))
            label2.pack()

            label3 = ttk.Label(self,text = "Modified Average  Train Score on all training data: " +str(sqrt(mean_squared_error(y, newAveraged))))
            label3.pack()

##            testXData = testData.copy()
##            del testXData['Id']



            if choiceOfPrints:
                 
                y_prediction=newAveraged
                y_prediction = np.exp(y_prediction)

                file = open('AveragedWithMods_WOutliers.txt','w')
                file.write("Id,SalePrice\n")

                predictions_data = pandas.DataFrame(data = {'prediction':y_prediction})

##                print("Printed: lATEST")
                
                #print(predictions_data.shape)
                printPred = np.array(predictions_data)
                printId = np.array(testIds)
                for i in range(printId.size):
                    file.write(str(printId[i])+","+str(printPred[i][0])+"\n")

                file.close()


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
            


            
             
app = HousePrices()
app.mainloop()
 
