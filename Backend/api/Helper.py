from io import BytesIO
from PIL import Image
import numpy as np


def read_file_as_image(data, is_grayscale) -> np.ndarray:
    '''
    BytesIO(data) is creating a new binary stream from data, which is assumed to be bytes-like object representing an image.
    Image.open(BytesIO(data)) is using PIL's Image.open() function to open the image that is stored in the binary stream.
    np.array(Image.open(BytesIO(data))) is converting the image into a NumPy array, which can be used for further image processing tasks.
    '''
    image = Image.open(BytesIO(data))

    # Resize image to 224x224 pixels
    image = image.resize((224, 224))

    if is_grayscale:
        # Convert to grayscale
        image = image.convert('L')
    else:  
        # Convert to RGB format
        image = image.convert('RGB')

    # Convert to numpy array
    image = np.array(image)
    print(image.shape)
    return image
