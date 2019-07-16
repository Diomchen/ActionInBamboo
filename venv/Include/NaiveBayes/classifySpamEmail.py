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
    docList = []
    classList = []
    fullText = []

    for i in range(1,26):
        wordList = textParse(open('E:\\ActionInML\\venv\\res\\NaiveBayes\\email\\email\\spam\\%d.txt'% i).read())
        # print(wordList)
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('E:\\ActionInML\\venv\\res\\NaiveBayes\\email\\email\\ham\\%d.txt'% i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = bayes.createVocabList(docList)
    trainingSet = list(range(50))
    testSet = []
    for i in range(10):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bayes.setOfWords2Vec(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0,p1,pA = bayes.trainNBO(np.array(trainMat),np.array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        print(docList[docIndex])
        wordVector = bayes.setOfWords2Vec(vocabList,docList[docIndex])
        if bayes.classifyNB(np.array(wordVector),p0,p1,pA)!=classList[docIndex]:
            errorCount += 1
    print('error rate is {0}'.format(float(errorCount)/len(testSet)))

if __name__ == '__main__':
    spamTest()