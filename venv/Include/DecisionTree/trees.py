import numpy as np
import matplotlib.pyplot as plot
from math import log
import operator

decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

#信息熵的计算
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel]+=1
    shannonEnt = 0.0

    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob*log(prob,2)
    return shannonEnt

def createDataSet():
    dataSet = [[1,1,'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no'],]
    labels = ['no surfacing','flippers']
    return dataSet,labels

#选出符合特征的数据元素，并将符合特征的那一项去掉
def splitDataSet(dataSet,axis,value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            #去除它的一个特征
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

#选择信息增益最大的特征
def chooseBestFeatureToSplit(dataSet):
    #获取特征值的数量
    numFeatures = len(dataSet[0])-1
    #获取整体数据香农纯度值
    baseEntropy = calcShannonEnt(dataSet)

    bestInfoGain = 0.0
    bestFeature = -1

    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet,i,value)
            prob = len(subDataSet)/float(len(dataSet))
            #求这一特征中在结果为特定值的纯度
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        #infoGain信息增益越大，则纯度越高，信息下降最快的方向
        print("#i_{0},##infoGain_{1}".format(i,infoGain))
        if(infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

#递归建决策树
def createTree(dataSet,labels):
    #找出所有dataSet中的Y特征种类
    classList = [example[-1] for example in dataSet]
    #只有一种特征则返回此特征值，即到叶子端
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    #若dataSet只有最后一个特征且其Y特征不为一种，则选择多数表决的方式选择Y特征值
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    #找到最佳切分列的索引
    bestFeat = chooseBestFeatureToSplit(dataSet)
    #找到该索引点对应的特征
    bestLabel = labels[bestFeat]
    #以此特征建决策树
    myTree = {bestLabel:{}}
    #从特征表中删除此特征
    del(labels[bestFeat])
    #以此特征按照其特征值依次往下建立决策树
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        #递归求树
        myTree[bestLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return myTree

def plotNode(nodeText,centerPt,parentPt,nodeType):
    createPlot.ax1.annotate(nodeText,xy=parentPt,xycoords='axes fraction',xytext=centerPt,textcoords='axes fraction',
                            va='center',ha='center',bbox=nodeType,arrowprops=arrow_args)
def createPlot():
    fig = plot.figure(1,facecolor='white')
    fig.clf()
    createPlot.ax1 = plot.subplot(111,frameon=False)
    plotNode('a decision node',(0.5,0.1),(0.1,0.5),decisionNode)
    plotNode('a leaf node', (0.8, 0.1), (0.3, 0.8), leafNode)
    plot.show()

if __name__ == '__main__':
    myDat,labels = createDataSet()
    createTree(myDat, labels)
    createPlot()

