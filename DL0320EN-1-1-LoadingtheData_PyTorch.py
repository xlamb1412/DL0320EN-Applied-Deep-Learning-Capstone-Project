import matplotlib.pyplot as plt
from PIL import Image

train_dir = 'data/training_data_pytorch'
name = '/0.jpeg'
Input = train_dir + name
img=Image.open(Input)
plt.imshow(img)
plt.show()


train_dir = 'data/training_data_pytorch'
name = '/52.jpeg'
Input = train_dir + name
img=Image.open(Input)
plt.imshow(img)
plt.show()


valid_dir = 'data/validation_data_pytorch'
name = '/0.jpeg'
Input = valid_dir + name
img=Image.open(Input)
plt.imshow(img)
plt.show()


valid_dir = 'data/validation_data_pytorch'
name = '/35.jpeg'
Input = train_dir + name
img=Image.open(Input)
plt.imshow(img)
plt.show()