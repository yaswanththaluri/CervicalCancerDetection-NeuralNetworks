import pandas as pd
import numpy as np
import csv


def logsigmoidFunction(netInput):
    return 1 / (1 + np.exp(-1 * netInput))

def resultFunction(result):
    if result>0:
        return 1
    else:
        return 0


def trainNetwork(inputs, target, a):
    weightsTrain = [0.0] * len(inputs[0])
    accuracyList=[]
    weightsList=[]
    epoch = int(input("Enter Epoch"))
    for o in range(epoch):
        accuracy = 0
        for x in range(len(inputs)):
            bias = 1
            learningRate = 0.1
            for k in range(len(inputs[x])):
                if inputs[x][k] == '?':
                    inputs[x][k] = 0
                else:
                    inputs[x][k] = float(inputs[x][k])
            netInput = np.dot(inputs[x], weightsTrain)
            result = logsigmoidFunction(netInput)
            output = resultFunction(result)
            if target[x][a-1] == output:
                accuracy = accuracy + 1
            else:
                error = target[x][a-1] - output
                for j in range(len(inputs[x])):
                    weightsTrain[j] = weightsTrain[j] + learningRate * error * inputs[x][j]

        print("Accuracy succeded", (accuracy/len(inputs)) * 100)
        print(weightsTrain)
        accuracyList.append((accuracy/len(inputs)) * 100)
        weightsList.append(weightsTrain)

    maxWgt = max(accuracyList)
    index = accuracyList.index(maxWgt)
    print("\n\n\nHighest accuracy achieved is ",max(accuracyList))
    print("at weights ",weightsList[index])

    if a==1:
        with open('hinselmann.csv', 'w') as writeFile:
            writer = csv.writer(writeFile)
            writer.writerow(weightsList[index])
        writeFile.close()
    elif a==2:
        with open('schiller.csv', 'w') as writeFile:
            writer = csv.writer(writeFile)
            writer.writerow(weightsList[index])
        writeFile.close()
    elif a==3:
        with open('citology.csv', 'w') as writeFile:
            writer = csv.writer(writeFile)
            writer.writerow(weightsList[index])
        writeFile.close()
    elif a==4:
        with open('biospy.csv', 'w') as writeFile:
            writer = csv.writer(writeFile)
            writer.writerow(weightsList[index])
        writeFile.close()


def testing(input, n):

    global d, c

    if n==1:
        data = pd.read_csv("hinselmann.csv", header=None)
        d = "hinselmann"
        c = data.values
    elif n==2:
        data = pd.read_csv("schiller.csv", header=None)
        d="schiller"
        c = data.values
    elif n==3:
        data = pd.read_csv("citology.csv", header=None)
        d="citology"
        c = data.values
    elif n==4:
        data = pd.read_csv("biospy.csv", header=None)
        d="biospy"
        c = data.values

    weights = c[0]
    print(weights)
    netInput=np.dot(input, weights)
    t=logsigmoidFunction(netInput)
    result = resultFunction(t)
    if result==1:
        print("The disease is confirmed. It is "+d)
    else:
        print("It is confirmed that disease is not " + d)

    print(result)


print("Loading data from files......")
data = pd.read_csv("risk_factors_cervical_cancer.csv")
c = data.values
inputValues = []
targetValues = []
weights = [0] * (len(c[0])-4)
for i in range(len(c)):
    d = c[i].tolist()
    inputValues.append(d[:-4])
    targetValues.append(d[-4:])
n = int(input("Enter\n 1. Train the Model\n 2. Testing"))
if n == 1:
    a = int(input("Enter\n 1. Hinselmann\n 2. Schiller\n 3. Cytology\n 4. Biospy"))
    trainNetwork(inputValues, targetValues, a)
elif n == 2:
    a = int(input("Enter the disease to detect\n 1. Hinselmann\n 2. Schiller\n 3. Cytology\n 4. Biospy"))
    listInput = list(map(float, input("\nEnter the diagnosis values with comma seperated for testing...\n").split(",")))
    testing(listInput, a)
