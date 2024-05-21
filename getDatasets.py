import datasets
import numpy as np

# Load the dataset
dataset = datasets.load_dataset("mnist")


def GetImageTrainingData(
        startFromIndex: int = 0
        ):
    # Access the training data
    train_data = dataset['train']

    # Extract the images and labels from the training data
    images = np.array([np.array(image) for image in train_data['image']]) * 1/255
    labels = []


    labelsNumbers = np.array(train_data['label'])
    for l in labelsNumbers:
        vec = np.zeros(10)
        vec[l] = 1
        labels.append(vec)

    images = images[startFromIndex:]
    labels = labels[startFromIndex:]
    
    print("done getting data")
        


    return images, labels


# # Display the first image and label
# import matplotlib.pyplot as plt

# plt.imshow(images[0], cmap='gray')
# plt.title(f"Label: {labels[0]}")
# plt.show()


i, l = GetImageTrainingData()

print(i[0])