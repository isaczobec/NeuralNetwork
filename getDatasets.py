import datasets
import numpy as np

# Load the dataset


def GetImageTrainingData(
        startFromIndex: int = 0,
        dataSetString: str = 'train',
        shuffleData: bool = False
        ):
    # Access the training data
    dataset = datasets.load_dataset("mnist")
    train_data = dataset[dataSetString]

    # shuffle the data
    if shuffleData: train_data = train_data.sample(frac=1).reset_index(drop=True)

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

