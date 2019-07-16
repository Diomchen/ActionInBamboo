import numpy as np
import operator
from math import log

def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]
    return postingList,classVec

def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = set(document) | vocabSet
    return list(vocabSet)

def setOfWords2Vec(vocabList,inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
    return returnVec

def trainNBO(trainMatrix,trainCategory):
    #文档矩阵行数
    numTrainDoc = len(trainMatrix)
    # print(numTrainDoc)
    #文档矩阵列数
    numWords = len(trainMatrix[0])
    #标记的文档在全文档所占比例
    pAbusive = sum(trainCategory)/float(numTrainDoc)
    #降低乘积为0的概率
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0

    for i in range(numTrainDoc):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])

    #不同类别中，分别计算每一个词语在所有出现的词语中的概率
    #取对数避免下溢
    p1Vect = np.log(p1Num / p1Denom)
    p0Vect = np.log(p0Num / p0Denom)
    return p0Vect,p1Vect,pAbusive

#朴素贝叶斯分类函数
def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    p1 = sum(vec2Classify*p1Vec)+log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0-pClass1)
    if p1 >p0:
        return 1
    else:
        return 0

#准备词袋
def bagOfWords2VecMN(vocabList,inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

if __name__ == '__main__':
    dataSet,listClass = loadDataSet()
    vocabList = createVocabList(dataSet)
    vec = setOfWords2Vec(vocabList,dataSet[1])

    trainMat = []
    for postinDoc in dataSet:
        trainMat.append(setOfWords2Vec(vocabList,postinDoc))

    print(vocabList)
    # print(sum(listClass))
    p0,p1,pA = trainNBO(trainMat,listClass)
    # print(p0)
    # print(p1)
    # print(pA)

    testEntry = ['love','my','dalmation']
    thisDoc = np.array(setOfWords2Vec(vocabList,testEntry))
    print("testEntry classified as {0}".format(classifyNB(thisDoc,p0,p1,pA)))

    testEntry = ['stupid','garbage']
    thisDoc = np.array(setOfWords2Vec(vocabList,testEntry))
    print("testEntry classified as {0}".format(classifyNB(thisDoc,p0,p1,pA)))

