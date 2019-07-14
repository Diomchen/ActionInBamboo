import numpy as np
import matplotlib.pyplot as plt
import operator

def creatDataset():
    group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group , labels

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

def file2matrix(filename):
    fin = open(filename)
    arrayOLines = fin.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = np.zeros((numberOfLines,3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        #文本的解析
        line = line.strip()
        listFromLine = line.split('\t')
        #returnMat(类似于x)
        returnMat[index,:] = listFromLine[0:3]
        #添加末尾标记到classLabelVector(类似于y)
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector

#归一化处理 newValue = (oldValue-min)/(max-min) 除数也可以为标准差）
def autoNorm(dataSet):
    minvals = dataSet.min(0)
    maxvals = dataSet.max(0)
    ranges = maxvals - minvals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minvals,(m,1))
    normDataSet = normDataSet/np.tile(ranges,(m,1))
    return normDataSet,ranges,minvals

def datingClassTest(filename):
    hoRatio = 0.10
    datingDataMat,datingLabels = file2matrix(filename)
    normMat,ranges,minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    #这里最好是选取随机的10%数据作为测试用，一般都是60%训练数据，20%交叉验证集，20%测试
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classfileResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print("the classfier come back {0},the real answer is {1}".format(classfileResult,datingLabels[i]))
        if(classfileResult!=datingLabels[i]):
            errorCount+=1
    print("the error rate is {0}".format(errorCount/float(numTestVecs)))

def classifyPerson(filename):
    resultList = ["not at all","in small doses","in large doses"]
    percentTats = float(input("time spend on playing computergame?"))
    ffMiles = float(input("frequent fliter miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))

    datingMat,datingLabels = file2matrix(filename)
    normMat,ranges,minVals = autoNorm(datingMat)

    inArr = np.array([ffMiles,percentTats,iceCream])
    classifierResult = classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
    print("You will probably like this person:",resultList[classifierResult-1])


if __name__ == '__main__':
    filename = "E:\\ActionInML\\venv\\res\\K-m\\datingTestSet2.txt"
    group,labels = creatDataset()
    # x = group[:,0]
    # y = group[:,1]
    #
    # plt.scatter(x>=1.0,y>=1.0,marker="o")
    # plt.scatter(x <1.0, y < 1.0, marker=".")
    # plt.show()

    # print(classify0([0,0],group,labels,3))

    returnMat,classLabelVector = file2matrix(filename)

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(returnMat[:,1],returnMat[:,2],15.0*np.array(classLabelVector),15.0*np.array(classLabelVector))
    # plt.show()

    normMat,ranges,minVals = autoNorm(returnMat)
    # print(normMat)

    #测试算法
    datingClassTest(filename)

    #运用数据
    classifyPerson(filename)