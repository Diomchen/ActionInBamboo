import numpy as np
import bayes
import re
import random
import sys

def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W',bigString)
    return [ tok.lower() for tok in listOfTokens if len(tok)>2]

def spamTest():
    #创建单个文本的List
    docList = []
    #创建类别List
    classList = []
    #创建全局文本List
    fullText = []

    #遍历文件夹的文件
    for i in range(1,26):
        wordList = textParse(open('E:\\ActionInML\\venv\\res\\NaiveBayes\\email\\email\\spam\\%d.txt'% i).read())
        # print(wordList)
        #append每一个文本到docList
        docList.append(wordList)
        #extend每一个文本到fullText
        fullText.extend(wordList)
        classList.append(1)

        #同理
        wordList = textParse(open('E:\\ActionInML\\venv\\res\\NaiveBayes\\email\\email\\ham\\%d.txt'% i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    #选出全部单词（不重复），其实这里直接list(set(fullText))也行
    vocabList = bayes.createVocabList(docList)
    #创建训练集
    trainingSet = list(range(50))
    #创建测试集
    testSet = []
    #随机从训练集选出20%的测试集
    for i in range(10):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        #从训练集中删去选中的测试集
        del(trainingSet[randIndex])

    #数据化每个训练集的文本
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bayes.setOfWords2Vec(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    #计算出p0,p1,pA 这里不用np.array也可以
    p0,p1,pA = bayes.trainNBO(np.array(trainMat),np.array(trainClasses))
    errorCount = 0
    #测试训练结果
    for docIndex in testSet:
        print(docList[docIndex])
        wordVector = bayes.setOfWords2Vec(vocabList,docList[docIndex])
        if bayes.classifyNB(np.array(wordVector),p0,p1,pA)!=classList[docIndex]:
            errorCount += 1
    print('error rate is {0}'.format(float(errorCount)/len(testSet)))

if __name__ == '__main__':
    spamTest()