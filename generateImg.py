from keras.models import load_model
import numpy as np
from PIL import Image

noiseShape = (100, )
generator = load_model('generator.h5')
noise = np.random.normal(0, 1, (128, noiseShape[0]))

# Adjust to generate different number of images
rows = 10
cols = 10
margin = 16
imgShape = 128

image_array = np.full(( 
  margin + (rows * (imgShape+margin)), 
  margin + (cols * (imgShape+margin)), 3), 
  255, dtype=np.uint8)
generated_images = generator.predict(noise)
generated_images = 0.5 * generated_images + 0.5
image_count = 0

for row in range(rows):
	for col in range(cols):
		r = row * (imgShape+16) + margin
		c = col * (imgShape+16) + margin
		image_array[r:r+imgShape,c:c+imgShape] = generated_images[image_count] * 255
		image_count += 1

image = Image.fromarray(image_array)
image.save("sample.png")