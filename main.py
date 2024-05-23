import PIL.Image
import datasets
import PIL
import numpy as np
import math
import random
import csv
import os
import functions
import time
import matplotlib.pyplot as plt
import getDatasets



plt.style.use('fivethirtyeight')
    

class NodeLayer():
    """A class representing a layer of nodes in a neural network. 
    Contains the wheights and biases used to calculate the activation of
    the next layer of nodes."""
    def __init__(self,
                 numberOfNodes: int, # how many nodes this layer has 
                 nextNodeLayer = None, # the next layer in the network. Used to initialize weights for this one
                 initializeWeightsRandomly = True, # whether or not to initialize weights randomly
                 randomizeMultiplier:float = 1, # the multiplier for the random weights

                 squishingFunction = functions.Sigmoid, # the function used to squish the activations of this layer of nodes
                 squishingFunctionDerivative = functions.SigmoidDerivative, # the derivative of the squishing function used to squish the activations of this layer of nodes
                 ) -> None:
        
        self.numberOfNodes = numberOfNodes
        self.nextNodeLayer = nextNodeLayer

        self.zValues = np.zeros(numberOfNodes)
        """a vector containing the z values of this layer of nodes.z = (weights @ activations) + biases"""

        self.zDerivatives = np.zeros(numberOfNodes)
        """A vector containing the derivative of the cost function with respect to the z values of this layer of nodes."""
                
        # iniialize the activations
        self.activations = np.zeros(numberOfNodes)
        """A vector containing all the activations in this layer of nodes."""

        self.squishingFunction: callable = squishingFunction
        """The function used to squish the activations of this layer of nodes."""

        self.squishingFunctionDerivative: callable = squishingFunctionDerivative
        """The derivative of the squishing function used to squish the activations of this layer of nodes."""
        
        # initialize the weights
        weightList = []
        for nextNodes in range(nextNodeLayer.numberOfNodes):
            weightRow = []
            for node in range(numberOfNodes):

                if initializeWeightsRandomly:
                    weightRow.append((random.random() * 2 -1) * randomizeMultiplier) # generate random weight
                else:
                    weightRow.append(0) # initialize to 0 if not random

            weightList.append(weightRow)

        self.weights = np.array(weightList)
        """A matrix containing all the weights connecting to the next layer of nodes."""

        # initialize the biases
        biasList = []
        for node in range(nextNodeLayer.numberOfNodes):
            if initializeWeightsRandomly:
                biasList.append((random.random() * 2 - 1) * randomizeMultiplier) # generate random bias
            else:
                biasList.append(0) # initialize to 0 if not random

        self.biases = np.array(biasList)
        """A vector containing all the biases for this layer of nodes."""

    def CalcNextActivations(self):
        """Calculate and set the activations of the next layer of nodes."""

        self.nextNodeLayer.zValues = self.CalculateActivationRaw()
        activations = self.ApplySquishingFunction(self.nextNodeLayer.zValues, self.squishingFunction) #apply the sigmoid function to the activations
        self.nextNodeLayer.activations = activations # set the activations of the next layer to the calculated activations

        # print("activations:" + str(activations))

    # functions to calculate the activations of the next layer of nodes:

    def CalculateActivationRaw(self) -> np.ndarray:
        """Calculate and return the activations of the next layer of nodes by multiplying the weights by the activations of this layer and adding the biases. DOES NOT apply a squishing function."""
        activations = (self.weights @ self.activations) + self.biases

        return activations

    def ApplySquishingFunction(self, 
                               activations: np.ndarray, 
                               function: callable
                               ) -> np.ndarray:
        """Apply a squishing function a vector of activations, and return the result."""

        newActivations = np.zeros(len(activations))

        for i,activation in enumerate(activations):
            newActivations[i] = function(activation)

        return newActivations
    
    def CalculateZDerivatives(self):
        """calculates and stores the derivative of the cost function with respect to the z values of the next layer of nodes."""
        for ia,a in enumerate(self.activations):
            derivativeSum = 0
            for iz,dz in enumerate(self.nextNodeLayer.zDerivatives):
                derivativeSum += dz * self.weights[iz][ia]

            derivativeSum *= self.squishingFunctionDerivative(self.zValues[ia])
            self.zDerivatives[ia] = derivativeSum


        
    def __str__(self) -> str:
        return "node activations:" + "\n" + f"{self.activations}" + "\n" + "node Weights:" + "\n" + f"{self.weights}" + "\n" + "node Biases:" + "\n" + f"{self.biases}"+ "\n" + "node zDerivatives:" + f"{self.zDerivatives}"




class OutputNodeLayer(NodeLayer):
    def __init__(self, numberOfNodes: int,
                 squishingFunction = functions.Sigmoid,
                 squishingFunctionDerivative = functions.SigmoidDerivative,
                 
                 ) -> None:

        # this class/row of nodes doesnt have a next layer
        # it also doesnt have weights

        self.numberOfNodes = numberOfNodes
        self.activations = np.zeros(numberOfNodes)

        self.zValues = np.zeros(numberOfNodes)
        """a vector containing the z values of this layer of nodes.z = (weights @ activations) + biases"""

        self.zDerivatives = np.zeros(numberOfNodes)
        """A vector containing the derivative of the cost function with respect to the z values of this layer of nodes."""

        self.squishingFunctionDerivative: callable = squishingFunctionDerivative
        """The derivative of the squishing function used to squish the activations of this layer of nodes."""

    def __str__(self) -> str:
        return "Output Node Layer" + "\n" + "node activations:" + "\n" + f"{self.activations}" + "\n" + "node zDerivatives:" + f"{self.zDerivatives}"
    
    def CalculateZDerivatives(self, desiredResult: np.ndarray):
        """calculates and stores the derivative of the cost function with respect to the z values of this layer of nodes."""
        dCdA = functions.costDerivative(desiredResult, self.activations)
        for i,c in enumerate(self.zDerivatives):
            self.zDerivatives[i] = dCdA[i] * self.squishingFunctionDerivative(self.zValues[i])

    



class Network():
    """Takes a list of the amount of nodes(int) (eg. [2,5,2] for a network with
    2 input nodes, 5 nodes in the only hidden layer, and 2 output nodes) 
    in each layer and creates a neural network."""
    def __init__(self,
                 LayerAndNodeList: list[int],
                 initializeWeightsRandomly = True, # whether or not to initialize weights randomly
                 randomizeMultiplier: float = 1, # the multiplier for the random weights
                 squishingFunction = functions.Sigmoid, # the function used to squish the activations of this layer of nodes
                 squishingFunctionDerivative = functions.SigmoidDerivative, # the derivative of the squishing function used to squish the activations of this layer of nodes
                 ) -> None:
        
        self.nodeLayers: list[NodeLayer] = []
        """An ordered list containing all the layers of nodes in the neural network."""

        self.outputLayer = OutputNodeLayer(LayerAndNodeList.pop(-1))
        """The output layer of the neural network."""
        self.nodeLayers.append(self.outputLayer)

        while len(LayerAndNodeList) > 0:
            self.nodeLayers.insert(0,NodeLayer(LayerAndNodeList.pop(-1), self.nodeLayers[0], initializeWeightsRandomly = initializeWeightsRandomly, randomizeMultiplier=randomizeMultiplier))



    def RunNetwork(self, inputVector: np.ndarray, 
                   flattenInput: bool = False,
                    desiredOutput: np.ndarray = None,
                   printResult = False,
                   ) -> np.ndarray: 
        """runs this network and returns an output vector.
        Also sets the activations of the layers and keeps them set."""

        if flattenInput: inputVector = inputVector.flatten()

        self.nodeLayers[0].activations = inputVector

        for layer in self.nodeLayers[:-1]: # dont run the output layer
            layer.CalcNextActivations()

        if desiredOutput is not None:
            if printResult: 
                cost = self.CalculateCost(desiredOutput)
                print("activations: "+ str(self.outputLayer.activations) + " cost:" + str(cost))
        elif printResult:
            print("activations: "+ str(self.outputLayer.activations))

        return self.outputLayer.activations
    

    def CalculateCost(self, desiredOutput: np.ndarray) -> float:
        """Calculates the cost of the network given a desired output vector."""
        return functions.cost(desiredOutput, self.outputLayer.activations)
    
    def CalculateAllZDerivatives(self, desiredOutput: np.ndarray):
        """Calculates and stores the z derivatives of all the layers in the network."""
        self.outputLayer.CalculateZDerivatives(desiredOutput)
        for layer in self.nodeLayers[:-1]:
            layer.CalculateZDerivatives()
            # if layer.zDerivatives[0] == 0: print("zero!")






    def __str__(self) -> str:
        s = ""
        for index,layer in enumerate(self.nodeLayers):
            s += "LAYER NUMBER: " + str(index) + "\n"
            s += str(layer) + "\n"
        return s
    

    def CalculateGradient(self, desiredOutput: np.ndarray, 
                          calculateZDerivatives: bool = True,
                          debug: bool = False
                          ) -> tuple[list[np.ndarray],list[np.ndarray]]:
        """Calculates the gradient of the cost function with respect to the weights and biases of the network.
        Takes in a desired output vector.
        returns biasGradients, weightGradients"""
        
        # calculate the z derivatives
        if calculateZDerivatives: self.CalculateAllZDerivatives(desiredOutput)

        # initialize the gradient lists
        biasGradients = [np.zeros(layer.numberOfNodes) for layer in self.nodeLayers[:-1]]
        weightGradients = [np.zeros(layer.weights.shape) for layer in self.nodeLayers[:-1]]

        if debug:
            print("weight gradients:")
            print(weightGradients)

        for i in range(len(self.nodeLayers)-1):
            biasGradients[i] = self.nodeLayers[i+1].zDerivatives # the bias gradient is just the z derivative

            # calculate the weight gradient
            for ri in range(len(self.nodeLayers[i].weights)):
                for ci in range(len(self.nodeLayers[i].weights[ri])):
                    weightGradients[i][ri][ci] = self.nodeLayers[i].activations[ci] * self.nodeLayers[i+1].zDerivatives[ri]


        return biasGradients, weightGradients
    
    def UpdateWeightsAndBiases(self, biasGradients: list[np.ndarray], weightGradients: list[np.ndarray], learningRate: float, debug:bool = False):
        """Updates the weights and biases of the network given the bias and weight gradients and a learning rate."""
        for i in range(len(self.nodeLayers)-1):
            if debug:
                print("bias gradients:")
                print(biasGradients[i])
                print("weight gradients:")
                print(weightGradients[i])
                print("biases:")
                print(self.nodeLayers[i].biases)
                print("weights:")
                print(self.nodeLayers[i].weights)
            self.nodeLayers[i].biases -= learningRate * biasGradients[i]
            self.nodeLayers[i].weights -= learningRate * weightGradients[i]


    def TrainNetwork(self, 
                     inputlist: list[np.ndarray], 
                     desiredOutputlist: list[np.ndarray], 
                     learningRate: float, 
                     trainigSetSize: int = 10, 
                     maxTrainingSets: int = 100,
                     epochs: int = 1, # the number of times to run through the training data
                     shuffleData: bool = True, # whether or not to shuffle the data between epochs
                     log: bool = True,
                     flattenInput: bool = False, # whether or not to flatten the input matrix
                     showdata: bool = False, # whether or not to show the data in the console,
                     saveLogTo: str = None, # the path to save the data to. Does not save if None
                     plotDataWhenDone: bool = False, # whether or not to plot the data
                     ):
        
        """Trains the network on a list of input vectors and a list of desired output vectors."""

        # lists to store the counter and cost values for plotting or logging
        counterVals = []
        costVals = []
        desiredOutputVals = []

        
        

        counter = 0 # counts up for each training example that has been run through

        AverageBiasGradient = [np.zeros(layer.biases.shape) for layer in self.nodeLayers[:-1]]
        AverageWeightGradients = [np.zeros(layer.weights.shape) for layer in self.nodeLayers[:-1]]

        for ei in range(epochs):

            print("epoch " + str(ei) + " of " + str(epochs) + "Started!")

            if shuffleData:
                # shuffle the data
                combined = list(zip(inputlist,desiredOutputlist))
                random.shuffle(combined)
                inputlist, desiredOutputlist = zip(*combined)

            exampleCounter = 0 # counts the number of training examples that have been run through in this epoch
            SetCounter = 0 # counts the number of training examples that have been run through in this epoch
            while SetCounter < maxTrainingSets:
                # for each training set


                # START TRAINING SET
                for i in range(trainigSetSize):

                    # run the network and calculate the gradient
                    self.RunNetwork(inputlist[exampleCounter], flattenInput=flattenInput)

                    counter += 1

                    # plot and log the data
                    cost = None
                    if log or plotDataWhenDone or saveLogTo != None: cost = self.CalculateCost(desiredOutputlist[exampleCounter])
                    # log the data
                    if log: print("e:"+str(ei)+", trainingSet: " + str(SetCounter) + ", nr:" + str(i) + ", cost:" + str(cost))
                    # plot the data
                    if plotDataWhenDone or saveLogTo != None:
                        counterVals.append(counter)
                        costVals.append(cost)
                        desiredOutputVals.append(desiredOutputlist[exampleCounter])


                    
                    if showdata: 
                        print("result,desired:")
                        print((self.outputLayer.activations))
                        print((desiredOutputlist[exampleCounter]))
                        plt.imshow(inputlist[exampleCounter], cmap='gray')
                        plt.show()


                    calculatedBiasGradient, calculatedWeightGradient = self.CalculateGradient(desiredOutputlist[exampleCounter])

                    # add the calculated gradients to the average gradients
                    for j in range(len(self.nodeLayers)-1):
                        # print(AverageBiasGradient[j])
                        # print("...")
                        # print(calculatedBiasGradient[j])
                        AverageBiasGradient[j] += calculatedBiasGradient[j]
                        AverageWeightGradients[j] += calculatedWeightGradient[j]

                    exampleCounter += 1

                SetCounter += 1
                    

                # divide the average gradients by the training set size
                for j in range(len(self.nodeLayers)-1):
                    AverageBiasGradient[j] /= trainigSetSize
                    AverageWeightGradients[j] /= trainigSetSize
                
                # update the weights and biases
                self.UpdateWeightsAndBiases(AverageBiasGradient, AverageWeightGradients, learningRate)

        # save the data
        if saveLogTo != None:
            try:
                with open(saveLogTo + ".csv", mode='w') as file:
                    writer = csv.writer(file)
                    writer.writerow(["exampleNR","desired","cost"])
                    for i in range(len(counterVals)):
                        writer.writerow([counterVals[i],desiredOutputVals[i],costVals[i]])
            except:
                print("could not save data to " + saveLogTo)

        if plotDataWhenDone:
            plt.plot(counterVals,costVals)
            plt.show()
    
    def TestNetwork(
                    self,
                    inputlist: list[np.ndarray],
                    desiredOutputlist: list[np.ndarray],
                    maxattempts: int = None,
                    flattenInput: bool = False,
                    printResult: bool = False,
                    ):
        """Tests the network on a list of input vectors and a list of desired output vectors.
        returns the average cost of the network on the test data."""


        # run the network on the test data and calculate the cost, add it to the sum
        counter = 0
        costSum = 0
        for i in range(len(inputlist)):
            if maxattempts is not None and counter >= maxattempts: break
            self.RunNetwork(inputlist[i], flattenInput=flattenInput, desiredOutput=desiredOutputlist[i])
            costSum += self.CalculateCost(desiredOutputlist[i])
            counter += 1

        # calculate the average cost and return it
        averageCost = costSum / counter
        if printResult: print("average cost: " + str(averageCost))
        return averageCost
        


        

    def SaveNetwork(self, folderPath: str = "Models/", fileName: str = "model"):
        """Saves this network to a folder.""" 

        if not os.path.exists(folderPath):
            os.makedirs(folderPath)

        with open(folderPath + fileName + ".csv", mode='w') as file:
            writer = csv.writer(file)

            # write the number of nodes in each layer
            writer.writerow(["layerIndex","numberOfNodes"])
            for index,layer in enumerate(self.nodeLayers):
                writer.writerow([index,layer.numberOfNodes])


            writer.writerow(["activations"])
            writer.writerow(["layerIndex","nodeIndex","activation"])    
            # write the activations
            for layerIndex,layer in enumerate(self.nodeLayers):
                for nodeIndex,activation in enumerate(layer.activations):
                    writer.writerow([layerIndex,nodeIndex,activation])

            writer.writerow(["biases"])
            writer.writerow(["layerIndex","nodeIndex","bias"])
            # write the biases
            for layerIndex,layer in enumerate(self.nodeLayers[:-1]):
                for nodeIndex,bias in enumerate(layer.biases):
                    writer.writerow([layerIndex,nodeIndex,bias])
            
            writer.writerow(["weights"])
            writer.writerow(["layerIndex","rowIndex","columnIndex","weight"])
            # write the weights
            for layerIndex,layer in enumerate(self.nodeLayers[:-1]):
                for weightRowIndex, weightRow in enumerate(layer.weights):
                    for weightColumnIndex, weight in enumerate(weightRow):
                        writer.writerow([layerIndex,weightRowIndex,weightColumnIndex,weight])

def LoadNetwork( 
                path: str = "Models/model.csv", # the path to the folder containing the network
                skippedHeaders: list[str] = ["activations","biases","weights","layerIndex"] # the default headers for the csv file, excluding the layerIndex and numberOfNodes headers
                ): 
    """
    Loads and returns a network from a folder. In the .csv file, activations come first, then biases, then weights.
    
    A network is represented in a .csv file as follows:
    
    layerIndex,numberOfNodes
    
    activations
    layerIndex,nodeIndex,activation
    
    biases
    layerIndex,nodeIndex,bias
    
    weights
    layerIndex,rowIndex,columnIndex,weight
    """
    infoList = [[],[],[],[]]
    # add the rows to a normal list
    with open(path, mode="r") as file:
        reader = csv.reader(file)

        counter = -1
        justSkipped = False
        for index,row in enumerate(reader):

            if index % 2 != 0: continue # skip every other row, they are empty
            if row[0] in skippedHeaders: 
                justSkipped = True
                continue # skip the default headers

            if justSkipped:
                counter+=1

            infoList[counter].append(row)
            justSkipped = False


    activations = []
    biases = []
    weights = []


    for i,row in enumerate(infoList[0]):



        # create empty lists for the activations, biases, and weights 
        activations.append([None]*int(row[1])) # add a new layer to the activations list

    
        if (i == len(infoList[0])-1): break
        biases.append([None]*int(infoList[0][i+1][1]))
        weights.append([[None]*int(infoList[0][i][1]) for _ in range(int(infoList[0][i+1][1]))])



    # fill the activations


    for row in infoList[1]:
            
        activations[int(row[0])][int(row[1])] = float(row[2])

    # fill the biases
    for row in infoList[2]:
        biases[int(row[0])][int(row[1])] = float(row[2])



    # fill the weights
    for row in infoList[3]:
        weights[int(row[0])][int(row[1])][int(row[2])] = float(row[3])



    # create the network
    n = Network([int(row[1]) for row in infoList[0]],initializeWeightsRandomly = False)
    for nodeLayerIndex,nodeLayer in enumerate(n.nodeLayers):
        nodeLayer.activations = np.array(activations[nodeLayerIndex])
        if nodeLayerIndex == len(n.nodeLayers)-1: break
        nodeLayer.biases = np.array(biases[nodeLayerIndex])
        nodeLayer.weights = np.array(weights[nodeLayerIndex])

    return n

        
            







# ------ load --------
# n = Network([784,16,16,10], randomizeMultiplier=1)
# n = LoadNetwork()
# networkbefore = str(n)


# ------ random train --------
# trainingData = [[random.random(), random.random()] for _ in range(1000)]
# desiredResults = [[1,1] for _ in range(1000)]
# n.TrainNetwork(trainingData,desiredResults,0.1,10,999,flattenInput=False)

# timebefore = time.time()


# ------ train --------
# import getDatasets
# data = getDatasets.GetImageTrainingData()

# n.TrainNetwork(
#                 data[0],data[1],
#                learningRate=0.1,
#                trainigSetSize=10,
#                maxTrainingSets=5900,
#                flattenInput=True,
#                log=True,
#                 plotDataWhenDone=True
#                )

# n.SaveNetwork("Models/","test24-23-5:1.csv")

# ------ test --------

n = LoadNetwork("Models/tests5-23-24/testNormal.csv")

testImage = PIL.Image.open("testImages/test4.png")
testImageArray = np.array(testImage)
testImageArray = testImageArray[:,:,0]
testImageArray = testImageArray * 1/255

n.RunNetwork(testImageArray,
             flattenInput=True,
             printResult=True)

# print the guess
max = 0
guess = 0
for i in range(len(n.outputLayer.activations)):
    if n.outputLayer.activations[i] > max: 
        max = n.outputLayer.activations[i]
        guess = i

print("guess:" + str(guess))

plt.imshow(testImageArray, cmap='gray')
plt.show()
    






# ------ train multiple --------

def TrainMultiple(
        testSpecDict: dict[str,dict[str,any]], # a dictionary containing the specifications for the networks to be trained
        data: list[list,list] = None, # the training data
):


    # amountOfnodes = [
    #     [784,16,16,10],
    #     [784,16,16,10],
    #     [784,16,16,10],
    #     [784,16,16,16,10],
    #     [784,16,16,16,10],
    #     [784,16,16,16,10],
    #     [784,16,16,16,16,10],
    #     [784,16,16,16,16,10],
    #     [784,16,16,16,16,10],
    #                  ]

    
    for test,settings in testSpecDict.items():

        timebefore = time.time()

        print("start training model " + str(test) + "...")

        n = Network(
            settings["nodeAmount"], 
            randomizeMultiplier=settings["randomizeMultiplier"],
            squishingFunction=settings["squishingFunction"],
            squishingFunctionDerivative=settings["squishingFunctionDerivative"],
            )
        n.TrainNetwork(
                        data[0],data[1],
                        learningRate=settings["learningRate"],
                        trainigSetSize=settings["trainigSetSize"],
                        maxTrainingSets=settings["maxTrainingSets"],
                        epochs=settings["epochs"],
                        shuffleData=True,
                        flattenInput=True,
                        log=True,
                        showdata=False,
                        plotDataWhenDone=False,
                        saveLogTo="LoggedData/tests24-23-5/" + test,
        )

        n.SaveNetwork("Models/tests5-23-24/", test)

        print("model " + str(test) + " done training")
        print("time taken: " + str(time.time()-timebefore) + " seconds")


    # ------ test multiple --------

    # import getDatasets
    # data = getDatasets.GetImageTrainingData(dataSetString='train')

    # for i in range(9):
    #     n = LoadNetwork("Models/ManyLetterTests/test" + str(i) + ".csv")
    #     averageCost = n.TestNetwork(data[0],data[1],maxattempts=100,flattenInput=True,printResult=False)
    #     print("model " + str(i) + " average cost: " + str(averageCost))

    # n = LoadNetwork("Models/tryNestWOrk.csv")
    # n.TestNetwork(data[0],data[1],maxattempts=100,flattenInput=True,printResult=True)




settingsDict = {
        "testNormal2":{
                "learningRate":0.1,
                "trainigSetSize":10,
                "maxTrainingSets":5995,
                "epochs":2,
                "nodeAmount":[784,16,16,10],
                "randomizeMultiplier":1,
                "squishingFunction":functions.ReLU,
                "squishingFunctionDerivative":functions.ReLUDerivative,
                },
        "testMoreNodes":{
                "learningRate":0.1,
                "trainigSetSize":10,
                "maxTrainingSets":5995,
                "epochs":2,
                "nodeAmount":[784,89,44,22,10],
                "randomizeMultiplier":1,
                "squishingFunction":functions.ReLU,
                "squishingFunctionDerivative":functions.ReLUDerivative,
                },
    }

# data = getDatasets.GetImageTrainingData(shuffleData=False)

# TrainMultiple(
#     settingsDict,
#     data,
#     )


