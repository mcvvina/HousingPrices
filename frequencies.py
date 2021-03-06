# CSC 240 Data Mining Final Project
# Spring 2018
# Jack Dalton, Mcvvina Lin
# frequencies: is a module to allow the use of attributeData, freqData, attributes, csvData, allPossData, and the possibility to print the freqData from other files


import requests



## Inputs: filename, default=https://raw.githubusercontent.com/mcvvina/HousingPrices/master/train.csv
## Attributes:
#### csvData = list form of all data exluding attribute names
#### attributes = list of attribute names
#### allPossData = list of lists of all possible nominal data
#### freqData = list of lists of frequency of each nominal data
## Functions
#### printFrequencies = prints formatted list of frequencies


class frequencies:
    def __init__(self, filename='https://raw.githubusercontent.com/mcvvina/HousingPrices/master/train.csv'):
    # Read in the data and store in nxn matrix
   # filename = 

        response = requests.get(filename)
        self.csvData = response.text.split("\n")
        self.csvData = [eachLine.split(',') for eachLine in self.csvData if len(eachLine)>0]


        

        # Store attribute name list and delete from the rest of the data
        self.attributes = self.csvData[0]
        #print(csvData[0])
        del self.csvData[0]


        #To store each attribute as a row and each house as a column
        self.attributeData = [ [] for each in self.attributes]


        #Define possible values for each attribute
        posId = []
        posMSSubClass=['20', '30', '40', '45', '50', '60', '70', '75', '80', '85', '90', '120', '150', '160', '180', '190']
        posMSZoning=['A', 'C', 'FV', 'I', 'RH', 'RL', 'RP', 'RM']
        posLotFrontage=[]
        posLotArea=[]
        posStreet=['Grvl', 'Pave']
        posAlley=['Grvl', 'Pave', 'NA']
        posLotShape=['Reg', 'IR1', 'IR2', 'IR3']
        posLandContour=['Lvl', 'Bnk', 'HLS', 'Low']
        posUtilities=['AllPub', 'NoSewr', 'NoSeWa', 'ELO']
        posLotConfig=['Inside', 'Corner', 'CulDSac', 'FR2', 'FR3']
        posLandSlope=['Gtl', 'Mod', 'Sev']
        posNeighborhood=['Blmngtn', 'Blueste', 'BrDale', 'BrkSide', 'ClearCr', 'CollgCr', 'Crawfor', 'Edwards', 'Gilbert', 'IDOTRR', 'MeadowV', 'Mitchel', 'Names', 'NoRidge', 'NPkVill', 'NridgHt', 'NWAmes', 'OldTown', 'SWISU', 'Sawyer', 'SawyerW', 'Somerst', 'StoneBr', 'Timber', 'Veenker']
        posCondition1=['Artery', 'Feedr', 'Norm', 'RRNn', 'RRAn', 'PosN', 'PosA', 'RRNe', 'RRAe']
        posCondition2=['Artery', 'Feedr', 'Norm', 'RRNn', 'RRAn', 'PosN', 'PosA', 'RRNe', 'RRAe']
        posBldgType=['1Fam', '2FmCon', 'Duplx', 'TwnhsE', 'TwnhsI']
        posHouseStyle=['1Story','1.5Fin','1.5Unf','2Story','2.5Fin','2.5Unf','SFoyer', 'SLvl']
        posOverallQual=['10', '9', '8', '7', '6', '5', '4', '3', '2', '1']
        posOverallCond=['10', '9', '8', '7', '6', '5', '4', '3', '2', '1']
        posYearBuilt=[]
        posYearRemodAdd=[]
        posRoofStyle=['Flat', 'Gable', 'Gambrel', 'Hip', 'Mansard', 'Shed']
        posRoofMatl=['ClyTile', 'CompShg', 'Membran', 'Metal', 'Roll', 'Tar&Grv', 'WdShake', 'WdShngl']
        posExterior1st=['AsbShng', 'AsphShn', 'BrkComm', 'BrkFace', 'CBlock', 'CemntBd', 'HdBoard', 'ImStucc', 'MetalSd', 'Other', 'Plywood', 'PreCast', 'Stone', 'Stucco', 'VinylSd', 'Wd Sdng', 'WdShing']
        posExterior2nd=['AsbShng', 'AsphShn', 'BrkComm', 'BrkFace', 'CBlock', 'CemntBd', 'HdBoard', 'ImStucc', 'MetalSd', 'Other', 'Plywood', 'PreCast', 'Stone', 'Stucco', 'VinylSd', 'Wd Sdng', 'WdShing']
        posMasVnrType=['BrkCmn', 'BrkFace', 'CBlock', 'None', 'Stone']
        posMasVnrArea=[]
        posExterQual=['Ex', 'Gd', 'TA', 'Fa', 'Po']
        posExterCond=['Ex', 'Gd', 'TA', 'Fa', 'Po']
        posFoundation=['BrkTil', 'CBlock', 'PConc', 'Slab', 'Stone', 'Wood']
        posBsmtQual=['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA']
        posBsmtCond=['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA']
        posBsmtExposure=['Gd', 'Av', 'Mn', 'No', 'NA']
        posBsmtFinType1=['GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf', 'NA']
        posBsmtFinSF1=[]
        posBsmtFinType2=['GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf', 'NA']
        posBsmtFinSF2=[]
        posBsmtUnfSF=[]
        posTotalBsmtSF=[]
        posHeating=['Floor', 'GasA', 'GasW', 'Grav', 'OthW', 'Wall']
        posHeatingQC=['Ex', 'Gd', 'TA', 'Fa', 'Po']
        posCentralAir=['N', 'Y']
        posElectrical=['SBrkr', 'FuseA', 'FuseF', 'FuseP', 'Mix']
        pos1stFlrSF=[]
        pos2ndFlrSF=[]
        posLowQualFinSF=[]
        posGrLivArea=[]
        posBsmtFullBath=[]
        posBsmtHalfBath=[]
        posFullBath=[]
        posHalfBath=[]
        posBedroom=[]
        posKitchen=[]
        posKitchenQual=['Ex', 'Gd', 'TA', 'Fa', 'Po']
        posTotRmsAbvGrd=[]
        posFunctional=['Typ', 'Min1', 'Min2', 'Mod', 'Maj1', 'Maj2', 'Sev', 'Sal']
        posFireplaces=[]
        posFireplaceQu=['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA']
        posGarageType=['2Types', 'Attchd', 'Basment', 'BuiltIn', 'CarPort', 'Detchd', 'NA']
        posGarageYrBlt=[]
        posGarageFinish=['Fin', 'RFn', 'Unf', 'NA']
        posGarageCars=[]
        posGarageArea=[]
        posGarageQual=['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA']
        posGarageCond=['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA']
        posPavedDrive=['Y', 'P', 'N']
        posWoodDeckSF=[]
        posOpenPorchSF=[]
        posEnclosedPorch=[]
        pos3SsnPorch=[]
        posScreenPorch=[]
        posPoolArea=[]
        posPoolQC=['Ex', 'Gd', 'TA', 'Fa', 'NA']
        posFence=['GdPrv', 'MnPrv', 'GdWo', 'MnWw', 'NA']
        posMiscFeature=['Elev', 'Gar2', 'Othr', 'Shed', 'TenC', 'NA']
        posMiscVal=[]
        posMoSold=[]
        posYrSold=[]
        posSaleType=['WD', 'CWD', 'VWD', 'New', 'COD', 'Con', 'ConLw', 'ConLI', 'ConLD', 'Oth']
        posSaleCondition=['Normal', 'Abnorml', 'AdjLand', 'Alloca', 'Family', 'Partial']
        ### NEEDED FOR TRAIN FILE
        posSales = []

        # Create 2D list

        ### posSales included for train data--remove for test data file 
        self.allPossData = [posId,posMSSubClass, posMSZoning, posLotFrontage, posLotArea, posStreet, posAlley, posLotShape, posLandContour, posUtilities, posLotConfig, posLandSlope, posNeighborhood, posCondition1, posCondition2, posBldgType, posHouseStyle, posOverallQual, posOverallCond, posYearBuilt, posYearRemodAdd, posRoofStyle, posRoofMatl, posExterior1st, posExterior2nd, posMasVnrType, posMasVnrArea, posExterQual, posExterCond, posFoundation, posBsmtQual, posBsmtCond, posBsmtExposure, posBsmtFinType1, posBsmtFinSF1, posBsmtFinType2, posBsmtFinSF2, posBsmtUnfSF, posTotalBsmtSF, posHeating, posHeatingQC, posCentralAir, posElectrical, pos1stFlrSF, pos2ndFlrSF, posLowQualFinSF, posGrLivArea, posBsmtFullBath, posBsmtHalfBath, posFullBath, posHalfBath, posBedroom, posKitchen, posKitchenQual, posTotRmsAbvGrd, posFunctional, posFireplaces, posFireplaceQu, posGarageType, posGarageYrBlt, posGarageFinish, posGarageCars, posGarageArea, posGarageQual, posGarageCond, posPavedDrive, posWoodDeckSF, posOpenPorchSF, posEnclosedPorch, pos3SsnPorch, posScreenPorch, posPoolArea, posPoolQC, posFence, posMiscFeature, posMiscVal, posMoSold, posYrSold, posSaleType, posSaleCondition]

        ## adds posSales to allPossData because it is required for train data
        if filename == 'https://raw.githubusercontent.com/mcvvina/HousingPrices/master/train.csv':
            self.allPossData.append(posSales)
        
        self.freqData = [ [] for ea in self.allPossData]
        #print(allPossData[2])



        #print(csvData[0])
        for eachHouse in self.csvData:

            
            for index,eachType in enumerate(eachHouse):
                
                ### Generate Frequencise
                if len(self.allPossData[index]) == []:
                    if len(self.freqData)<=index:
                        self.freqData.append([]) 
                    continue
                posValues = self.allPossData[index]
                if len(self.freqData[index])<len(posValues):
                    self.freqData[index] = [0 for elm in posValues]
                for elInd,element in enumerate(posValues):
                    if element == eachType:
                        self.freqData[index][elInd] += 1
                    elif eachType == 'C (all)':
                        self.freqData[index][elInd] += 1

                ### Generate attributeData
                if self.allPossData[index] ==[] and eachHouse[index] == 'NA':
                    self.attributeData[index].append(-9999999999)
                elif self.allPossData[index] ==[]:
                    self.attributeData[index].append(int(eachHouse[index]))
   
                else:
                    self.attributeData[index].append(eachHouse[index].lower())

        for ix, eachRow in enumerate(self.attributeData):
            if self.allPossData[ix] == []:
                sums = 0
                for each in eachRow:
                    #print(each)
                    sums = sums+each
                
                mean = sums/len(eachRow)
                for i,each in enumerate(eachRow):
                    if each == -9999999999:
                        self.attributeData[ix][i] = mean
                    
                

    def printFrequencies(self):
        stringFormat= "{name:15} | {posValues:50} | {frequencies:50}"
        print(stringFormat.format(name = "Attribute", posValues = "Possible Values", frequencies = "Frequencies"))

        print("-"*115)

        for ind,each in enumerate(self.freqData):
            length = len(str(self.allPossData[ind]))
            printFrom = 0
            printTo = 50
            times = 0
            posVal = str(self.allPossData[ind])
            freq = str(each)
            while(length > 50):
                if times ==0:
                    print(stringFormat.format(name = self.attributes[ind], posValues = posVal[printFrom:printTo], frequencies = freq[printFrom:printTo]))
                    
                else:
                    
                    print(stringFormat.format(name = " ", posValues = posVal[printFrom:printTo], frequencies = freq[printFrom:printTo]))
                    
                printFrom +=50
                printTo+=50
                length -=50
                times+=1
            if times>0:
                
                print(stringFormat.format(name = " ", posValues = posVal[printFrom:printTo], frequencies = freq[printFrom:printTo]))
            else:
                print(stringFormat.format(name =self.attributes[ind], posValues = posVal[printFrom:printTo], frequencies = freq[printFrom:printTo]))
            print("-"*115)
    def help(self):
        print("""
## Inputs: filename, default=https://raw.githubusercontent.com/mcvvina/HousingPrices/master/train.csv
## Attributes:
#### csvData = list form of all data exluding attribute names
#### attributes = list of attribute names
#### allPossData = list of lists of all possible nominal data
#### freqData = list of lists of frequency of each nominal data
## Functions
#### printFrequencies = prints formatted list of frequencies
""")
