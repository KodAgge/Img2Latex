import numpy as np
import cv2
from skimage.io import imread

def loadImages(path, width, height, n_test, scale = 255):
  # Preallocate memory
  data = np.zeros([height, width, n_test])

  for i in range(n_test):
    image_path = path + '/Image' + str(i) + '.png'
    img = imread(image_path) # Read image

    img_gray_scale = img[:,:,0] # Remove unessecary dimensions

    (img_height, img_width) = img_gray_scale.shape # Current shape

    # If the picture is to short compared to the required format
    
    if img_height / img_width < height / width:
      new_height = height / width * img_width
      pad = (new_height - img_height) / 2
      img_padded= cv2.copyMakeBorder(img_gray_scale, math.ceil(pad), math.floor(pad), 0, 0, cv2.BORDER_CONSTANT, value=255)
    
    # If the picture is to narrow compared to the required format
    elif img_height / img_width > height / width:
      new_width = img_height * width / height
      pad = (new_width - img_width) / 2
      img_padded= cv2.copyMakeBorder(img_gray_scale, 0, 0, math.ceil(pad),math.floor(pad), cv2.BORDER_CONSTANT, value=255)

    img_rescaled = img_padded / (scale / 2) - 1 # Rescale to [-1, 1]

    #FIXME Crashes when using [width, height]. Changed to tuple (width, height)
    #img_resize = cv2.resize(img_rescaled, [width, height]) # Resize image 
    img_resize = cv2.resize(img_rescaled, (width, height)) # Resize image

    data[:,:,i] = img_resize # Save transformed image data

    # plt.imshow(img_resize)
    # plt.show()

  return data


def normalizeData(test_images, train_images, val_images):
  # Find mean and std
  #all_data = np.concatenate([train_images, test_images, val_images], axis = 2)
  mean = np.mean(train_images)
  std = np.std(train_images)

  # Normalize
  test_images = (test_images - mean) / std
  train_images = (train_images - mean) / std
  val_images = (val_images - mean) / std

  return test_images, train_images, val_images