from keras.models import Model, Sequential, load_model
from keras import models
from keras.layers import Input, BatchNormalization, Reshape, Dropout, Dense, Flatten, Activation, ZeroPadding2D
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.optimizers import Adam
import keras
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import glob
import tensorflow as tf
import os

# Change/Remove as necessary
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
from tensorflow.python.client import device_lib
device_lib.list_local_devices()

# Save rows x cols samples to image, takes in epoch number for naming purposes
def save_image(epoch):
	rows = 5
	cols = 5
	margin = 48
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

	im = Image.fromarray(image_array)
	im.save("sample" + str(epoch) + ".png")

# Load training data from image set, takes in boolean loadNewData, inputPath for images, 
# imageShape, and dataSavePath for both saving and reading numpy array 
def load_train_data(loadNewData, inputPath, imageShape, dataSavePath):
	trainingData = []
	
	if loadNewData:
		print("Loading training images")
		for filename in sorted(glob.glob(inputPath)):
			image = Image.open(filename).resize((imageShape[0], imageShape[1]), Image.ANTIALIAS)
			if np.asarray(image).shape == (128, 128, 3):
				trainingData.append(np.asarray(image))
		trainingData = np.reshape(trainingData, (-1, imageShape[0], imageShape[1], imageShape[2]))
		trainingData = trainingData / 127.5 - 1
		np.save(dataSavePath, trainingData)
	else:
		trainingData = np.load(dataSavePath)

	return trainingData

# Construct generator portion of GAN. Takes in noiseShape and returns generator model.
def build_generator(noiseShape):
	model = Sequential()
	alpha = 0.2
	momentum = 0.8

	model.add(Dense(4*4*256, activation = "relu", input_shape = noiseShape))
	model.add(Reshape((4,4,256)))

	model.add(UpSampling2D())
	model.add(Conv2D(256, kernel_size = 3, padding = "same"))
	model.add(BatchNormalization(momentum = momentum))
	model.add(Activation("relu"))

	model.add(UpSampling2D())
	model.add(Conv2D(256, kernel_size = 3, padding = "same"))
	model.add(BatchNormalization(momentum = momentum))
	model.add(Activation("relu"))

	model.add(UpSampling2D())
	model.add(Conv2D(128, kernel_size = 3, padding = "same"))
	model.add(BatchNormalization(momentum = momentum))
	model.add(Activation("relu"))

	model.add(UpSampling2D())
	model.add(Conv2D(128, kernel_size = 3, padding = "same"))
	model.add(BatchNormalization(momentum = momentum))
	model.add(Activation("relu"))

	model.add(UpSampling2D())
	model.add(Conv2D(128, kernel_size = 3, padding = "same"))
	model.add(BatchNormalization(momentum = momentum))
	model.add(Activation("relu"))
	
	model.add(Conv2D(3, kernel_size = 3, padding = "same"))
	model.add(Activation("tanh"))

	noise = Input(shape = noiseShape)
	image = model(noise)

	return Model(noise, image)

# Construct discriminator portion of GAN. Takes in imageShape and returns discriminator model
def build_discriminator(imageShape):
	model = Sequential()
	alpha = 0.2
	dropoutRate = 0.25
	momentum = 0.8

	model.add(Conv2D(32, kernel_size = 3, strides = 2, input_shape = imageShape, padding = "same"))
	model.add(LeakyReLU(alpha))

	model.add(Dropout(dropoutRate))
	model.add(Conv2D(64, kernel_size = 3, strides = 2, padding = "same"))
	model.add(ZeroPadding2D(padding = ((0, 1),(0, 1))))
	model.add(BatchNormalization(momentum = momentum))
	model.add(LeakyReLU(alpha))

	model.add(Dropout(dropoutRate))
	model.add(Conv2D(128, kernel_size = 3, strides = 2, padding = "same"))
	model.add(BatchNormalization(momentum = momentum))
	model.add(LeakyReLU(alpha))

	model.add(Dropout(dropoutRate))
	model.add(Conv2D(256, kernel_size = 3, strides = 1, padding = "same"))
	model.add(BatchNormalization(momentum = momentum))
	model.add(LeakyReLU(alpha))

	model.add(Dropout(dropoutRate))
	model.add(Conv2D(512, kernel_size = 3, strides = 1, padding = "same"))
	model.add(BatchNormalization(momentum = momentum))
	model.add(LeakyReLU(alpha))

	model.add(Dropout(dropoutRate))
	model.add(Flatten())
	model.add(Dense(1, activation= 'sigmoid'))

	image = Input(shape = imageShape)
	validity = model(image)
		
	return Model(image, validity)

# Main training function. Takes in epochs, batchSize, learningRate, 
# saveFreq (number of epochs between saves), loadNewData (set to false if no numpy file exists), and numGPUs
def train(epochs, batchSize, learningRate, saveFreq, inputPath, loadNewData, numGPUs):
	noiseShape = (100, )
	imageShape = (128, 128, 3)
	optimizer = Adam(learningRate, .5)

	print("Building Discriminator")
	discriminator = build_discriminator(imageShape)
	discriminator = keras.utils.multi_gpu_model(discriminator, gpus = numGPUs)
	discriminator.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics = ['accuracy'])

	print("Building Generator")
	generator = build_generator(noiseShape)
	generator = keras.utils.multi_gpu_model(generator, gpus = numGPUs)
	noise = Input(shape = noiseShape)
	image = generator(noise)

	discriminator.trainable = False

	valid = discriminator(image)

	print("Building Combined")
	combined = Model(noise, valid)
	combined = keras.utils.multi_gpu_model(combined, gpus = numGPUs)
	combined.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics = ["accuracy"])

	trainingData = load_train_data(loadNewData = loadNewData, inputPath = inputPath, 
		imageShape = imageShape, dataSavePath = "training_data.npy");

	keras.backend.get_session().run(tf.global_variables_initializer())

	# Run on epochs, print/save based on saveFreq
	for epoch in range(epochs):
		y_real = np.ones((batchSize, 1))
		y_fake = np.zeros((batchSize, 1))
		
		idx = np.random.randint(0, trainingData.shape[0], batchSize)
		x_real = trainingData[idx]

		noise = np.random.normal(0, 1, (batchSize, noiseShape[0]))
		x_fake = generator.predict(noise)
		
		discriminator_real = discriminator.train_on_batch(x_real, y_real)
		discriminator_fake = discriminator.train_on_batch(x_fake, y_fake)

		discriminator_metric = .5 * np.add(discriminator_real, discriminator_fake)

		valid_y = np.array([1] * batchSize)
		noise = np.random.normal(0, 1, (batchSize, noiseShape[0]))
		generator_metric = combined.train_on_batch(noise, valid_y)

		if epoch % saveFreq == 0:
			save_image(epoch)
			print("Epoch: ", epoch, "Discriminator accuracy: ", 
				discriminator_metric[1], "Generator Accuracy: ", (generator_metric[1]))

	generator.save("generator.h5")

# call train function
train(epochs = 80000, batchSize = 128, learningRate = .0002, saveFreq = 2000, 
	inputPath = "images/*.jpg", loadNewData = False, numGPUs = 2)


