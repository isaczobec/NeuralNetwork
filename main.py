import datasets
import PIL
import numpy as np
import math
import random
import csv
import os



# Load the dataset
# dataset = datasets.load_dataset("mnist")

# image = dataset['train']['image'][0]
# image_array = np.array(image)
# print(image_array)

    

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
                
        # iniialize the activations
        self.activations = np.zeros(numberOfNodes)
        """A vector containing all the activations in this layer of nodes."""
        
        # initialize the weights
        weightList = []
        for nextNodes in range(nextNodeLayer.numberOfNodes):
            weightRow = []
            for node in range(numberOfNodes):
                weightRow.append(random.random()) # generate random weight
            weightList.append(weightRow)

        self.weights = np.array(weightList)
        """A matrix containing all the weights connecting to the next layer of nodes."""

        # initialize the biases
        biasList = []
        for node in range(nextNodeLayer.numberOfNodes):
            biasList.append(random.random())

        self.biases = np.array(biasList)
        """A vector containing all the biases for this layer of nodes."""

    def CalcNextActivations(self):
        """Calculate and set the activations of the next layer of nodes."""

        activations = (self.weights @ self.activations) + self.biases

        self.nextNodeLayer.activations = activations

        

        

    def __str__(self) -> str:
        return "node activations:" + "\n" + f"{self.activations}" + "\n" + "node Weights:" + "\n" + f"{self.weights}" + "\n" + "node Biases:" + "\n" + f"{self.biases}"




class OutputNodeLayer(NodeLayer):
    def __init__(self, numberOfNodes: int) -> None:

        # this class/row of nodes doesnt have a next layer
        # it also doesnt have weights

        self.numberOfNodes = numberOfNodes
        self.activations = np.zeros(numberOfNodes)

    def __str__(self) -> str:
        return "Output Node Layer" + "\n" + "node activations:" + "\n" + f"{self.activations}"



class Network():
    """Takes a list of the amount of nodes(int) (eg. [2,5,2] for a network with
    2 input nodes, 5 nodes in the only hidden layer, and 2 output nodes) 
    in each layer and creates a neural network."""
    def __init__(self,
                 LayerAndNodeList: list[int]
                 ) -> None:
        
        self.nodeLayers: list[NodeLayer] = []
        """An ordered list containing all the layers of nodes in the neural network."""

        self.outputLayer = OutputNodeLayer(LayerAndNodeList.pop(-1))
        """The output layer of the neural network."""
        self.nodeLayers.append(self.outputLayer)

        while len(LayerAndNodeList) > 0:
            self.nodeLayers.insert(0,NodeLayer(LayerAndNodeList.pop(-1), self.nodeLayers[0]))



    def RunNetwork(self, inputVector: np.ndarray) -> np.ndarray: 
        """runs this network and returns an output vector.
        Also sets the activations of the layers and keeps them set."""

        self.nodeLayers[0].activations = inputVector

        for layer in self.nodeLayers[:-1]: # dont run the output layer
            layer.CalcNextActivations()

        return self.outputLayer.activations
    

    def SaveNetwork(self, folderPath: str = "Models/", fileName: str = "model"):
        """Saves this network to a folder.""" 

        if not os.path.exists(folderPath):
            os.makedirs(folderPath)

        with open(folderPath + fileName + ".csv", mode='w') as file:
            writer = csv.writer(file)

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


    def __str__(self) -> str:
        s = ""
        for index,layer in enumerate(self.nodeLayers):
            s += "LAYER NUMBER: " + str(index) + "\n"
            s += str(layer) + "\n"
        return s


n = Network([2,2,2])
n.RunNetwork(np.array([3,5]))
print(n)
n.SaveNetwork()

