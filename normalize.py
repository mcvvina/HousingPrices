import math
import sys
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
def normalize(toNorm, posValues):
    ####################################### USE MEDIANS INSTEAD OF MEAN
    
    toReturn = [ [] for each in toNorm]

    ### Puts all possible strings into lowercase
    for i,eachRow in enumerate(posValues):
        for ids,each in enumerate(eachRow):
            posValues[i][ids] = posValues[i][ids].lower()

    ### Goes through each attribute in the toNorm list
    for ind, eachAttribute in enumerate(toNorm):
        ####                          ####  
        #### If it is continuous data ####
        ####                          ####
        if posValues[ind] == []: 

            ### Find Mean ###
            mean = sum(eachAttribute)/len(eachAttribute)

            ### Find standard deviation ###
            difference = []
            for each in eachAttribute:
                difference.append( pow((each - mean),2))
            
            std = math.sqrt( sum(difference)/(len(eachAttribute)-1) )


           ### Finds zscore value
            
            zscores = [((each - mean)/std) for each in eachAttribute]
            maxx = max(zscores)
            minn = min(zscores)

            ### adds list of zscores to the toReturn array (maps from -1 to 1)
            toReturn[ind] = [  ( (each -minn)/(maxx-minn))*(1+1)-1    for each in zscores]

        ####                          ####
        #### else its a nominal value ####
        ####                          ####
        else:
            # each value is ordered as in posValues
            ar = []
            toPr = []
            ## Gets the nominal->continuous value and adds to ar
            for iax,each in enumerate(eachAttribute):
                found = False

                ### If value is in the possible values
                if each in posValues[ind]:
                    for ix, ps in enumerate(posValues[ind]):
                        if each.lower() == ps.lower():
                            ar.append(ix)
                ### else its a mistype, NA, or other value with issues
                else:
                    ar.append(-9999999) ### Assigns it arbitrary value that won't likely be found
#################################################################################################################
##            ### Gets Mean of data in attribute
##            leng = len(ar)
##            sums = 0
##            for each in ar:
##                ### if value is not invalid use to find mean
##                if each != -9999999: 
##                    sums = sums+each
##                ### else skip it and 'reduce' length of array
##                else:
##                    leng=leng-1
##            mean = sums/leng
##            ### Replace each invalid value with mean
##            for xx, each in enumerate(ar):
##                if each == -9999999:
##                    ar[xx] = mean
###############################################################################################################


            ### Gets Median of data in attribute
            
            listNoInvalid = []
            for each in ar:
                ### if value is not invalid append to listNoInvalid
                if each != -9999999: 
                    listNoInvalid.append(each)
            leng = len(listNoInvalid)
            listNoInvalid = sorted(listNoInvalid)
            ### Find median ###

            ### if even
            if leng%2 == 0:
                lower = int(math.floor(leng/2))
                upper = lower+1
                median = (listNoInvalid[lower]+listNoInvalid[upper])/2
                
                
                if median.is_integer():
                    median = int(median)

                ### If median is a float (between two ordinal values)
                ### Use mode ###
                else:
                    mode = max(set(listNoInvalid), key=listNoInvalid.count)
                    #print('mode used: median = '+str(median)+" mode = "+str(mode))
                    median = mode
            ### if odd
            else:                
                midIndx = int((leng+1)/2)
                median = listNoInvalid[midIndx]
            
            ### Replace each invalid value with median
            for xx, each in enumerate(ar):
                if each == -9999999:
                    ar[xx] = median


                    

            ### Find standard deviation
            difference = []
            for each in ar:
                difference.append(pow((each - mean),2))
            
            std = math.sqrt( sum(difference)/(leng-1) )

            ### Find zscores
            zscores =[]
            for each in ar:
                ### If the values differ
                if each !=-9999999 and std !=0:
                    zscores.append(((each - mean)/std))
                ### if all the values are the same
                elif std ==0:
                    zscores.append(0)            
            maxx = max(zscores)            
            minn = min(zscores)

            ### Adds mapped zscores to toReturn
            toReturn[ind] = []
            for each in zscores:                
               if each ==0:
                   toReturn[ind].append(0)
               elif maxx-minn !=0: ### if this, then all values are the same
                    toReturn[ind].append( ( (each -minn)/(maxx-minn))*(1+1)-1 )
               else:
                   toReturn[ind].append(0)
            

    toReturn =(MinMaxScaler((-1,1)).fit_transform(toReturn)).tolist()
    return toReturn



