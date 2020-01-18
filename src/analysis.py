import numpy as np
import matplotlib.pyplot as plt


def plotAnalysis(valuesOfFirstDB, valuesOfSecondDB, valuesOfGround, firstDBLable, secondDBLabel, groundLabel, listOfXInfo, xLabel, yLabel, title):
    """
    :param valuesOfFirstDB, valuesOfSecondDB, valuesOfGround: list of values to plot for eache database and the ground database 
    :param listOfXInfo: list of category for each value 
    :param xLabel: string for the X-axis lable 
    :param yLabel: string for the Y-axis label 
    :param title: string for the title of the plot
    """
    # data to plot
    n_groups = len(valuesOfFirstDB)

    # create plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.28
    opacity = 0.8

    rects1 = plt.bar(index, valuesOfFirstDB, bar_width,
                     alpha=opacity,
                     color='#000099',
                     label=firstDBLable)

    rects2 = plt.bar(index + bar_width, valuesOfSecondDB, bar_width,
                     alpha=opacity,
                     color='#FF66B3',
                     label=secondDBLabel)

    rects3 = plt.bar(index + bar_width + bar_width, valuesOfGround, bar_width,
                     alpha=opacity,
                     color='#010B32',
                     label=groundLabel)

    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(title)
    plt.xticks(index + bar_width, listOfXInfo)
    plt.legend()

    plt.tight_layout()
    plt.show()


def plotWordPerKB(listOfLabel, listOfValues, listOfColors):
    labels = listOfLabel
    sizes = listOfValues
    colors = listOfColors
    # ['#4363d8', '#911eb4', '#000075']
    patches, texts = plt.pie(sizes, colors=colors, shadow=False, startangle=90)
    plt.legend(patches, labels, loc="best")
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
