from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from matplotlib.pyplot import imshow
import matplotlib.pylab as plt
import pandas as pd
from PIL import Image

train_csv_file = 'data/training_labels.csv'
validation_csv_file = 'data/validation_labels.csv'

train_data_name = pd.read_csv(train_csv_file)
train_data_dir = 'data/training_data_pytorch/'


train_image_name = train_data_dir + train_data_name.iloc[1, 2]
image = Image.open(train_image_name)
plt.imshow(image)
plt.show()

train_image_name = train_data_dir + train_data_name.iloc[19, 2]
image = Image.open(train_image_name)
plt.imshow(image)
plt.show()


validation_data_name = pd.read_csv(validation_csv_file)
validation_data_dir = 'data/validation_data_pytorch/'

validation_image_name = validation_data_dir + validation_data_name.iloc[19, 2]
image = Image.open(train_image_name)
plt.imshow(image)
plt.show()


class Dataset(Dataset):

    #Constructor
    def __init__(self, csv_file, data_dir, transform=None):

        # Image directory
        self.data_dir = data_dir

        # The transform is going to be used on image
        self.transform = transform

        # Load the CSV file contains image info
        self.data_name = pd.read_csv(csv_file)

        # Number of image in dataset
        self.len = self.data_name.shape[0]

    # Get the length
    def __len__(self):
        return self.len

    # Getter
    def __getitem__(self, idx):

        # Image file path
        img_name = self.data_dir + self.data_name.iloc[idx, 2]

        # Open image file
        image = Image.open(img_name)

        # The class label for the image
        y = self.data_name.iloc[idx, 3]

        # If there is any transform method, apply it onto the image
        if self.transform:
            image = self.transform(image)

        return image, y


train_dataset = Dataset(csv_file = train_csv_file, data_dir = 'data/training_data_pytorch/')
validation_data = Dataset(csv_file= validation_csv_file, data_dir='data/validation_data_pytorch/')

samples = [53, 23, 10]

for i in samples:
    image, y = train_dataset.__getitem__(i)
    plt.imshow(image)
    plt.show()


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
composed = transforms.Compose([transforms.Resize((224, 224)),
                               transforms.ToTensor(), transforms.Normalize(mean, std)])

test_normalization = Dataset(csv_file=train_csv_file,
                             data_dir='data/training_data_pytorch/',
                             transform = composed)

print("Mean: ", test_normalization[0][0].mean(dim = 1).mean(dim = 1))
print("Std:", test_normalization[0][0].std(dim = 1).std(dim = 1))