"""
Filter a TSI image to remove (i.e. set to black) portions of the image 
that are not part of the reflector. This includes the area
surrounding the reflector and also the reflector arm and sunblock.  

The region to removed is determined by a comparison with the mask.  The 
image and mask are compared. If a mask pixel is black then the corresponding
image pixel is set to black.

In this code, we assume the image and mask are stored in the same folder as the
python code and are named image.jpg, mask.png, respectively.  The resulting 
filtered image is saved to the same folder and called masked_image.png

Ultimately, one wants to automate this process so that an entire folder 
of images and masks can be filtered. 
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import numpy as np

# Read and display the image file
imagefileName = './image.jpg'
image = mpimg.imread(imagefileName)
plt.title("image")
plt.imshow(image)
#plt.show()
w = image.shape
print("image dimensions: ",w)

# Read and display the mask file
maskfileName = './mask.png'
mask_image = mpimg.imread(maskfileName)
plt.figure()
plt.title("mask")
plt.imshow(mask_image)
#plt.show()
wm = mask_image.shape
print("mask dimensions: ",wm)
#print(np.unique(mask_image))
print(np.unique(mask_image[:,:,0]))
print(np.unique(mask_image[:,:,1]))
print(np.unique(mask_image[:,:,2]))

# Filter the image based on the mask
indices = np.where((mask_image[:,:,0] == 0) & (mask_image[:,:,1] == 0) & (mask_image[:,:,2] == 0))

# eliminate sun reflection
sun_indices = np.where((mask_image[:,:,0] == 1) & (mask_image[:,:,1] == 1) & (mask_image[:,:,2] == 0))

# now apply that masks to the image
image[indices] = np.array([0,0,0])
image[sun_indices] = np.array([0,0,0])


# Display the resulting filtered image 
plt.figure()    
plt.title("filtered image")
plt.imshow(image)
plt.show()

# Save the resulting filtered image 
im = Image.fromarray(image)
im.save('masked_image.png')


