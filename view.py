### Most of this code obtained from https://pythonprogramming.net/how-to-embed-matplotlib-graph-tkinter-gui/

# The code for changing pages was derived from: http://stackoverflow.com/questions/7546050/switch-between-two-frames-in-tkinter
# License: http://creativecommons.org/licenses/by-sa/3.0/	

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
import math
import tkinter as tk
from tkinter import ttk
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

from sklearn.tree import ExtraTreeRegressor
#
######## Training Data ##########
#
freq = frequencies.frequencies()
normData = normalize.normalize(freq.attributeData, freq.allPossData)

names = pandas.Series(data = freq.attributes)

dataOrig = pandas.DataFrame(data = normData)
dataOrig = dataOrig.transpose()
dataOrig.columns = (names)

target = pandas.DataFrame(data = freq.attributeData)
target = target.transpose()
target.columns = names
target = target['SalePrice']



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
                  NormImportantScatter, RandomForestRegressorWithOutliers,
                  NormImportantOutlierScatter,RandomForestRegressorNoOutliers,
                  RandomForestClassifierWithOutliers, RandomForestClassifierNoOutliers,
                  AdaBoostRegressorWithOutliers):

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

        label = tk.Label(self, text="Techniques Used", font=('Helvetica', 12))
        label.pack(pady=10,padx=10)

        button6 = ttk.Button(self, text="RandomForestRegressor Output With Outliers",
                            command=lambda: controller.show_frame(RandomForestRegressorWithOutliers))
        button6.pack()
        
        button7 = ttk.Button(self, text="RandomForestRegressor Output No Outliers",
                            command=lambda: controller.show_frame(RandomForestRegressorNoOutliers))
        button7.pack()

        button8 = ttk.Button(self, text="RandomForestClassifier Output With Outliers",
                            command=lambda: controller.show_frame(RandomForestClassifierWithOutliers))
        button8.pack()

        button9 = ttk.Button(self, text="RandomForestClassifier Output No Outliers",
                            command=lambda: controller.show_frame(RandomForestClassifierNoOutliers))
        button9.pack()
        
        button10 = ttk.Button(self, text="AdaBoostRegressor Output With Outliers",
                            command=lambda: controller.show_frame(AdaBoostRegressorWithOutliers))
        button10.pack()

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
        

        minnn = 25000
        maxxx = 1000000
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
        
        minnn = 25000
        maxxx = 1000000
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


            

                  
            testXData = testData.copy()
            del testXData['Id']


            ##### WRITE OUT #####    
            testXData = testData.copy()
            del testXData['Id']
             
            y_prediction=cff.predict(testXData)

            file = open('AdaBoostRegressorWithOutliers.txt','w')
            file.write("Id,SalePrice\n")

            predictions_data = pandas.DataFrame(data = {'prediction':y_prediction})

            print("Printed: AdaBoostRegressorWithOutliers")
            
            #print(predictions_data.shape)
            printPred = np.array(predictions_data)
            printId = np.array(testIds)
            for i in range(printId.size):
                file.write(str(printId[i])+","+str(printPred[i][0])+"\n")

            file.close()

      
        
    
        
app = HousePrices()
app.mainloop()
