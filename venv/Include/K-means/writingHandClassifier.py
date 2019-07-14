import numpy as np
import matplotlib.pyplot as plot
from os import listdir
import operator

def img2vector(filename):
    returnVect = np.zeros((1,1024))
    fin = open(filename)
    for i in range(32):
        lineStr = fin.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def handWritingClassTest(trainDirName,testDirName):
    hwLabels = []
    trainingFileList = listdir(trainDirName)
    m = len(trainingFileList)
    #将文件夹里面所有的数据集合到一个矩阵里面
    trainingMat = np.zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector(trainDirName+"\\%s" % fileNameStr)

    testFileList = listdir(testDirName)
    errorCount = 0.0
    mTest = len(testFileList)

    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector(testDirName+"\\%s" % fileNameStr)
        classResult = classify0(vectorUnderTest,trainingMat,hwLabels,3)
        print("the classfier come back {0},the real answer is {1}".format(classResult, classNumStr))
        if(classResult != classNumStr):
            errorCount+=1.0
    print("total number is %d" % errorCount)
    print("error rate is %f" % (errorCount/float(mTest)))


def classify0(inX,dataSet,label, k):
    dataSetSize = dataSet.shape[0]
    # print("#{0}".format(dataSetSize))
    diffMat = np.tile(inX,(dataSetSize,1)) - dataSet
    # print("##{0}".format(diffMat))
    sqDiffMat = diffMat**2
    # print("###{0}".format(sqDiffMat))
    #行相加
    sqDistances = sqDiffMat.sum(axis = 1)
    # print("####{0}".format(sqDistances))
    distances = sqDistances**0.5

    #argsort返回的是从小到大排序后的index
    sortedDistIndicies  = distances.argsort()
    # print("*{0}".format(sortedDistIndicies))
    classCount = {}
    for i in range(k):
        voteIlabel = label[sortedDistIndicies[i]]

        # print("**{0}".format(voteIlabel))
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
        # print("***{0}".format( classCount[voteIlabel]))
    #classCount字典排序
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)

    return sortedClassCount[0][0]

if __name__ == '__main__':
    dirname = 'E:\\ActionInML\\venv\\res\\K-m\\trainingDigits'
    testdirname = "E:\\ActionInML\\venv\\res\K-m\\testDigits"
    handWritingClassTest(dirname,testdirname)