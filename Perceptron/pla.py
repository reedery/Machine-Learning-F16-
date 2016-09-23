__author__ = 'ryan reede'

"""
implementation of the Perceptron Learning Algorithm
for prof. alvarez's ML course @ boston college
9/7/16

"""

import numpy as np
import matplotlib.pyplot as plt


class PLA:
    def __init__(self, data, file, maxIterations):
        self.data = data
        self.filename = file
        self.maxIterations = maxIterations
        self.dim = np.shape(data)[1]
        self.w = None
        self.iterations = 0

    def initPlot(self):
        plt.axis([-5, 5, -5, 5])
        for element in self.negative:
            plt.plot(element[0], element[1], 'b*')
        for element in self.positive:
            plt.plot(element[0], element[1], 'ro')

    def organize(self):
        # prepossessing...
        # separate xy coordinates based on grouping
        self.positive = []
        self.negative = []
        d = self.data
        for i in range(np.shape(d)[0]):
            # create new list item (x, y)
           new = [d[i][0], d[i][1]]
           if d[i][self.dim-1] == 1:
              self.positive.append(new)
           else:
               self.negative.append(new)

    def graphLine(self, fx, fi):
        x = np.array((-10,10))
        y = fx*x - fi
        plt.plot(x, y, label = ("iteration " + str(self.iterations)))

    def iterateLine(self):
        d = self.data
        w = np.array([0,0,0]) # init weights @ 0
        size = np.shape(d)[0]
        pos = 0
        while pos < size and self.maxIterations > self.iterations:
            row = d[pos]
            newArray = np.array([1, row[0], row[1]])
            # dot product --> sign() function
            if np.transpose(w).dot(newArray) < 0: val = -1
            else: val = 1
            if (val == row[2]):
                pos += 1
                continue
            self.iterations += 1
            pos = 0
            # add new weights
            w = np.add(w, row[2] * newArray)
            formulaX = -w[1]/w[2]
            formulaIntercept = w[0]/w[2]
            self.graphLine(formulaX, formulaIntercept)
        self.w = w
        self.show()

    def show(self):
        if self.iterations <= self.maxIterations:
            print("\nConverged after " + str(self.iterations) + " iterations.")
        else:
            print("\nDid not converge after " + str(self.iterations) + \
                  " iterations. Data may not be linear separable.")
        print("\nFinal weight vector:")
        print(self.w)
        plt.legend()
        plt.show()


class Main:
    # txt file, rows of the form: (x, y, bool)
    fn = "testData/sampleData1.txt"
    # fn = "testData/sampleData2.txt"
    # fn = "testData/sampleData3.txt"
    data = np.loadtxt(fn)
    maxIter = int(input("upper bound on iterations:"))
    pla = PLA(data, fn, maxIter)
    pla.organize()
    pla.initPlot()
    pla.iterateLine()



