import datasets
import PIL
import numpy as np
import math
import random
import csv
import os
import functions




    

class NodeLayer():
    """A class representing a layer of nodes in a neural network. 
    Contains the wheights and biases used to calculate the activation of
    the next layer of nodes."""
    def __init__(self,
                 numberOfNodes: int, # how many nodes this layer has 
                 nextNodeLayer = None, # the next layer in the network. Used to initialize weights for this one
                 initializeWeightsRandomly = True # whether or not to initialize weights randomly
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

        self.squishingFunction: callable = functions.Sigmoid
        """The function used to squish the activations of this layer of nodes."""

        self.squishingFunctionDerivative: callable = functions.SigmoidDerivative
        """The derivative of the squishing function used to squish the activations of this layer of nodes."""
        
        # initialize the weights
        weightList = []
        for nextNodes in range(nextNodeLayer.numberOfNodes):
            weightRow = []
            for node in range(numberOfNodes):

                if initializeWeightsRandomly:
                    weightRow.append(random.random() * 2 -1) # generate random weight
                else:
                    weightRow.append(0) # initialize to 0 if not random

            weightList.append(weightRow)

        self.weights = np.array(weightList)
        """A matrix containing all the weights connecting to the next layer of nodes."""

        # initialize the biases
        biasList = []
        for node in range(nextNodeLayer.numberOfNodes):
            if initializeWeightsRandomly:
                biasList.append(random.random() * 2 - 1) # generate random bias
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
    def __init__(self, numberOfNodes: int) -> None:

        # this class/row of nodes doesnt have a next layer
        # it also doesnt have weights

        self.numberOfNodes = numberOfNodes
        self.activations = np.zeros(numberOfNodes)

        self.zValues = np.zeros(numberOfNodes)
        """a vector containing the z values of this layer of nodes.z = (weights @ activations) + biases"""

        self.zDerivatives = np.zeros(numberOfNodes)
        """A vector containing the derivative of the cost function with respect to the z values of this layer of nodes."""

        self.squishingFunctionDerivative: callable = functions.SigmoidDerivative
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
                 ) -> None:
        
        self.nodeLayers: list[NodeLayer] = []
        """An ordered list containing all the layers of nodes in the neural network."""

        self.outputLayer = OutputNodeLayer(LayerAndNodeList.pop(-1))
        """The output layer of the neural network."""
        self.nodeLayers.append(self.outputLayer)

        while len(LayerAndNodeList) > 0:
            self.nodeLayers.insert(0,NodeLayer(LayerAndNodeList.pop(-1), self.nodeLayers[0], initializeWeightsRandomly = initializeWeightsRandomly))



    def RunNetwork(self, inputVector: np.ndarray) -> np.ndarray: 
        """runs this network and returns an output vector.
        Also sets the activations of the layers and keeps them set."""

        self.nodeLayers[0].activations = inputVector

        for layer in self.nodeLayers[:-1]: # dont run the output layer
            layer.CalcNextActivations()

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
                     log: bool = True,
                     flattenInput: bool = False # whether or not to flatten the input matrix
                     ):
        
        """Trains the network on a list of input vectors and a list of desired output vectors."""

        counter = 0

        AverageBiasGradient = [np.zeros(layer.biases.shape) for layer in self.nodeLayers[:-1]]
        AverageWeightGradients = [np.zeros(layer.weights.shape) for layer in self.nodeLayers[:-1]]

        while counter < maxTrainingSets:
            # for each training set
            for i in range(trainigSetSize):

                # run the network and calculate the gradient
                self.RunNetwork(inputlist[i].flatten() if flattenInput else inputlist[i])
                if log: print("Attempt" + str(counter) + "cost:" + str(self.CalculateCost(desiredOutputlist[i])))
                calculatedBiasGradient, calculatedWeightGradient = self.CalculateGradient(desiredOutputlist[i])

                # add the calculated gradients to the average gradients
                for j in range(len(self.nodeLayers)-1):
                    # print(AverageBiasGradient[j])
                    # print("...")
                    # print(calculatedBiasGradient[j])
                    AverageBiasGradient[j] += calculatedBiasGradient[j]
                    AverageWeightGradients[j] += calculatedWeightGradient[j]

            # divide the average gradients by the training set size
            for j in range(len(self.nodeLayers)-1):
                AverageBiasGradient[j] /= trainigSetSize
                AverageWeightGradients[j] /= trainigSetSize
            
            # update the weights and biases
            self.UpdateWeightsAndBiases(AverageBiasGradient, AverageWeightGradients, learningRate)
            counter += 1

        

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

        
            







import getDatasets

data = getDatasets.GetImageTrainingData()

n = LoadNetwork()
networkbefore = str(n)
n.TrainNetwork(data[0],data[1],0.1,10,59000,flattenInput=True)
# print("before:")
# print(networkbefore)
# print("after:")
# print(n)
n.SaveNetwork()

# n = LoadNetwork()
# n.RunNetwork([0.1,0.1])
# print(n.outputLayer.activations)
# n.RunNetwork([0.1,0.9])
# print(n.outputLayer.activations)
# n.RunNetwork([0.5,0.5])
# print(n.outputLayer.activations)

# n = Network([2,3,2])
# print(n.outputLayer.zDerivatives)


