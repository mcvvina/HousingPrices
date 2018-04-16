# CSC 240 Data Mining Final Project
# Spring 2018
# Jack Dalton, Mcvvina Lin
# Plots histograms of all frequencies that are nominal values
# To change figure, click on histogram figure (not the arrow buttons)


import sys
import frequencies
import matplotlib.pyplot as plt
import itertools


def histograms(freq, indicesToPlot=[]):
    sys.setrecursionlimit(5000)

    # freq =
    # frequencies.frequencies()#'https://raw.githubusercontent.com/mcvvina/HousingPrices/master/test.csv')

    freqs = []
    poss = []
    attr = []

    if indicesToPlot == []:
        for ind, each in enumerate(freq.allPossData):
            if each != []:
                freqs.append(freq.freqData[ind])
                poss.append(freq.allPossData[ind])
                attr.append(freq.attributes[ind])
            else:
                # need all the values so that each row is an attribute and each
                # column is a value

                freqs.append(freq.attributeData[ind])
                poss.append([])
                attr.append(freq.attributes[ind])
    else:
        for ind in indicesToPlot:
            if freq.allPossData[ind] != []:
                freqs.append(freq.freqData[ind])
                poss.append(freq.allPossData[ind])
                attr.append(freq.attributes[ind])
            else:
                freqs.append(freq.attributeData[ind])
                poss.append([])
                attr.append(freq.attributes[ind])

    ys = itertools.cycle(freqs)
    xs = itertools.cycle(poss)
    ts = itertools.cycle(attr)

    def onclick(event):
        plt.clf()
        plt.gcf().clear()
        ay = next(xs)

        if ay == []:
            vals = next(ys)
            binwidth = int((max(vals) - min(vals)) / 20)
            if binwidth == 0:
                binwidth = 1
            plt.hist(vals, bins=range(int(min(vals)),
                                      int(max(vals)) + binwidth, binwidth))
        else:
            ax.set_xticklabels(ay)
            plt.bar(ay, next(ys))

        tit = next(ts)
        plt.title(tit + " Frequencies")
        plt.ylabel("Frequency")
        plt.draw()
        plt.show()

    fig, ax = plt.subplots()
    fig.canvas.mpl_connect('button_press_event', onclick)

    oy = next(xs)
    if oy == []:
        vals = next(ys)
        binwidth = int((max(vals) - min(vals)) / 20)
        if binwidth == 0:
            binwidth = 1
        plt.hist(vals, bins=range(int(min(vals)),
                                  int(max(vals)) + binwidth, binwidth))
    else:
        plt.bar(oy, next(ys))
        ax.set_xticklabels(oy)
    ax.set_ylabel("Frequency")
    ax.set_title(next(ts) + " Frequencies")

    plt.show()
    plt.draw()
