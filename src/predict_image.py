from PIL import Image
from tensorflow import  keras
import numpy as np
from main import model


# image = Image.open('two.jpg')
# image = image.resize((28,28), Image.ANTIALIAS)
# image.save('two_red.jpg')

#the paramn target size redimenciona the size the image
image_1 = keras.preprocessing.image.load_img('learning/teste.jpg', color_mode='grayscale', target_size=(28,28))
image_2 = keras.preprocessing.image.load_img('learning/two.jpg', color_mode='grayscale', target_size=(28,28))
image_3 = keras.preprocessing.image.load_img('learning/six.jpg', color_mode='grayscale', target_size=(28,28))

# image = keras.preprocessing.image.load_img('two_red.jpg', color_mode='grayscale')

#Convert the image for array numpy
arr_1 = keras.preprocessing.image.img_to_array(image_1)
arr_2 = keras.preprocessing.image.img_to_array(image_2)
arr_3 = keras.preprocessing.image.img_to_array(image_3)

# arr_4 = keras.preprocessing.image.img_to_array(image)

input_arr = np.array([arr_1, arr_2, arr_3])  # Convert single image to a batch.

predictions = model.predict(input_arr)
print(np.argmax(predictions[0]))
