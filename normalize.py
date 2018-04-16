import math
import sys


def normalize(toNorm, posValues):

    toReturn = [[] for each in toNorm]
    # Look into rainforest --- 100% sure there is package
    # Map to -1 to 1
    for i, eachRow in enumerate(posValues):
        for ids, each in enumerate(eachRow):
            posValues[i][ids] = posValues[i][ids].lower()
    for ind, eachAttribute in enumerate(toNorm):
        if posValues[ind] == []:  # is an int

            mean = sum(eachAttribute) / len(eachAttribute)

            difference = []
            for each in eachAttribute:
                difference.append(pow((each - mean), 2))

            std = math.sqrt(sum(difference) / (len(eachAttribute) - 1))

            zscores = [((each - mean) / std) for each in eachAttribute]
            maxx = max(zscores)
            minn = min(zscores)

            toReturn[ind] = [((each - minn) / (maxx - minn))
                             * (1 + 1) - 1 for each in zscores]
        else:  # nominal value
            # each value is ordered as in posValues
            ar = []
            toPr = []
            # Gets the nominal->continuous value and adds to ar
            for iax, each in enumerate(eachAttribute):
                found = False
                if each in posValues[ind]:
                    for ix, ps in enumerate(posValues[ind]):
                        if each.lower() == ps.lower():
                            ar.append(ix)

                else:
                  #          print(str(ind)+" "+str(each))
                    ar.append(-9999999)
##                        found = True

# elif each.lower() == 'c (all)' and ps.lower() == 'c': #accounts for mistyped data
# ar.append(ix)
##                        found = True
# elif each.lower() == 'twnhs' and ps.lower() == 'twnhsi':
# ar.append(ix)
##                        found = True
# elif each.lower() == 'duplex' and ps.lower() == 'duplx':
# ar.append(ix)
##                        found = True
# elif each.lower() == 'brk cmn' and ps.lower() == 'brkcomm':
# ar.append(ix)
##                        found = True
# elif each.lower() == 'cmentbd' and ps.lower() == 'cemntbd':
# ar.append(ix)
##                        found = True
# elif each.lower() == 'wd shng' and ps.lower() == 'wdshing':
# ar.append(ix)
##                        found = True

# if found == False:
##                    print(str(ind) +" "+ str(each))
# sys.exit()

            leng = len(ar)
            sums = 0
      #      print(ar)
            for each in ar:
                if each != -9999999:
                    sums = sums + each
                else:
                    leng = leng - 1
            mean = sums / leng

            for xx, each in enumerate(ar):
                if each == -9999999:
                    ar[xx] = mean

            difference = []
            for each in ar:
                difference.append(pow((each - mean), 2))

            std = math.sqrt(sum(difference) / (leng - 1))
# if std ==0:
# print(posValues[ind])
# print(eachAttribute)
            zscores = []
            for each in ar:
                if each != -9999999 and std != 0:
                    zscores.append(((each - mean) / std))
                elif std == 0:
                    zscores.append(0)

          #  zscores = [  ((each - mean)/std) for each in ar if each!= -9999999]
            maxx = max(zscores)

            minn = min(zscores)

            toReturn[ind] = []
            for each in zscores:
                if each == 0:
                    toReturn[ind].append(0)
                else:
                    toReturn[ind].append(
                        ((each - minn) / (maxx - minn)) * (1 + 1) - 1)

           # toReturn[ind] = [  ( (each -minn)/(maxx-minn))*(1+1)-1    for each in zscores]

    return toReturn
