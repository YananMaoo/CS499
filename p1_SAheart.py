import numpy as np
import matplotlib.pyplot as plt

def getDataSet(fileName):
    #data is equal to the sample size
    file = open(fileName)
    data = file.read().splitlines()
    dataset = []
    for lines in data:
        lines = lines.split(',')
        for words in lines:
            dataset.append(words)
    row = len(data)
    dataset = np.reshape(dataset, (len(data),cols))
    dataset = np.delete(dataset, 0, 0)#Delete first row
    dataset = np.delete(dataset, 0, 1)#Delete first col
    #Fixe= element data type
    for i in range(np.size(dataset, 0)):
        for j in range(np.size(dataset, 1)):
            if dataset[i][j]  == "Present":
                dataset[i][j] = 1.0
            elif dataset[i][j] == "Absent":
                dataset[i][j] = 0.0
            else:
                dataset[i][j] = float(dataset[i][j])
    file.close()
    return dataset

def getXY(data):
    data = data.astype(float)
    x = np.zeros((np.size(data, 0), features))
    y = []
    for row in range(np.size(data, 0)):
        for col in range(np.size(data, 1)):
            if(col != 9):
                x[row][col] = data[row][col]
            else:
                if(int(data[row][col]) == 0):
                    y.append(-1)
                else:
                    y.append(1)
    y = np.asmatrix(y)
    x = fixData(np.asmatrix(x))
    return x,y

def fixData(x):
    average = np.mean(x, axis = 0)
    x -= average
    x /= np.std(x, axis = 0)
    return x

def Driavative(x,y,theta):
    L = np.asmatrix(np.zeros((x[0].size), dtype = float))
    for i in range(y.size):
        l = (-1*y[0,i]*x[i])/(1 + np.exp(y[0,i]*np.matmul(x[i], theta.T)))
        L += l
    L = 1/(rows-1)*L
    return L

def getAllDataSet(data):
    np.random.shuffle(data)
    trainData,otherData = np.split(data, [int(samples*0.6)])
    testData,validationData = np.split(otherData, [int(np.size(otherData,0)/2)])
    return trainData,testData,validationData
    
def GradientDescent(x, y, stepSize, maxIterations):
    #print(np.zeros((x[0].size)))
    weightVector = np.asmatrix(np.zeros((x[0].size), dtype = float))
    weightMatrix = np.asmatrix(np.zeros((x[0].size, maxIterations)))
    for i in range(maxIterations):
        for j in range(x[0].size):
            weightMatrix[j, i] = weightVector[0, j]
        weightVector =  weightVector - stepSize * Driavative(x,y,weightVector)
    #print(weightMatrix)
    return weightMatrix

def PValueToBinary(pv):
    for row in range(np.size(pv, 0)):
        for col in range(np.size(pv, 1)):
            if pv[row, col] <= 0:
                pv[row, col] = 0
            else:
                pv[row, col] = 1
    return pv

def YOneZero(y):
    for row in range(np.size(y, 0)):
        for col in range(np.size(y, 1)):
            if int(y[row, col]) == -1:
                y[row, col] = 0
            else:
                y[row, col] = 1
    return y

def Substract(x1, x2):
    for i in range(np.size(x1, 0)):
        for j in range(np.size(x1, 1)):
            if int(x1[i,j]) == x2[0,i]:
                x1[i,j] = 0
            else:
                x1[i,j] = 1
    return x1

def lossFunction(pv, y):
    x = pv - y.T
    he = []
    loss = []
    for col in range(np.size(x, 1)):
        for row in range(np.size(x, 0)):
            he.append(np.square(x[row,col]))
        loss.append(sum(he)/np.size(x, 1))
        he = []
    return loss;

def lossFunction2(pv, y):
    x = pv - y.T
    he = np.zeros(maxIterations)
    loss = []
    for col in range(np.size(x, 1)):
        for row in range(np.size(x, 0)):
            #print(np.exp(pv[row, col] * y[0, row]))
            he[col] += 1/(1+np.exp(pv[row, col] * y[0, row]))
            #he.append(np.square(x[row,col]))
        loss.append(he[col]/np.size(x, 1))
    return loss;

def ErrorRate(predict, real):
    x = PValueToBinary(predict) -  real.T
    errors = []
    error = 0
    for col in range(np.size(x, 1)):
        for row in range(np.size(x, 0)):
            if x[row,col] == 0 or x[row,col] == 1:
                pass
            else:
                error += 1
        errors.append(error/np.size(x, 0))
        error = 0
    return errors
      
rows,cols = (31,11)
samples,features = (30,9)
file = "SAheart.data - Copy.txt"
dataset = getDataSet(file)
trainData,testData,validationData = getAllDataSet(dataset)
trainX,trainY = getXY(trainData)
validationX,validationY = getXY(validationData)
testDataX,testDataY = getXY(testData)

table = np.zeros((3, 2))

stepSize = 0.16
maxIterations = 2000
weightMatrix = GradientDescent(trainX, trainY, stepSize, maxIterations)

trainPValue = np.matmul(trainX, weightMatrix)
validationPValue = np.matmul(validationX, weightMatrix)

#print(PValueToBinary(trainPValue), trainY.T)
#print(PValueToBinary(trainPValue) -  trainY.T)
#print(ErrorRate(trainPValue, trainY))

#minIndex2 = np.argmin(ErrorRate(validationPValue, validationY))
#bestwv = weightMatrix[:,minIndex2]
#print(np.matmul(trainX, bestwv))
#trainDataBPValue = np.matmul(trainX, bestwv)
#validationDataBPValue = np.matmul(validationX, bestwv)
#testDataBPValue = np.matmul(testDataX, bestwv)

#table[0, 0] = np.mean(ErrorRate(np.matmul(trainX, bestwv), trainY.T))
#table[1, 0] = np.mean(ErrorRate(np.matmul(validationX, bestwv), validationY.T))
#table[2, 0] = np.mean(ErrorRate(np.matmul(testDataX, bestwv), testDataY.T))
#print(table)

#plt.title("Error Rate")
#plt.plot(list(range(0,maxIterations)), ErrorRate(trainPValue, trainY),label = "train", c = "black")
#plt.plot(list(range(0,maxIterations)), ErrorRate(validationPValue, validationY),label = "validation", c = "red")
#plt.annotate("Min", (minIndex2, ErrorRate(validationPValue, validationY)[minIndex2]))
#plt.xlabel('Iterations')
#plt.ylabel('ErrorRate')
#plt.legend()
#plt.show()

#plt.title("Loss Function")
#minIndex = np.argmin(lossFunction2(trainPValue, trainY))
#minIndex2 = np.argmin(lossFunction2(validationPValue, validationY))
#plt.plot(list(range(0,maxIterations)), lossFunction2(trainPValue, trainY),label = "train", c = "black")
#plt.plot(list(range(0,maxIterations)), lossFunction2(validationPValue, validationY),label = "validation", c = "red")
#plt.annotate("Min", (minIndex, lossFunction2(trainPValue, trainY)[minIndex]))
#plt.annotate("Min", (minIndex2, lossFunction2(validationPValue, validationY)[minIndex2]))
#plt.xlabel('Iterations')
#plt.ylabel('Loss')
#plt.legend()
#plt.show()


