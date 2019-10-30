from keras.models import Model, Sequential, load_model
from keras.layers import Input, BatchNormalization, ReShape, Dropout, Dense, Flatten, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.optimizers import Adam
import numpy as numpy
from PIL import Image
import matplotlib.pyplot as plt
import glob
import cv2

class GAN():
	def __init__(self):
		self.noiseShape = (100, )
		self.imageShape = (64, 64, 3)
		self.outputPath = ""
		self.inputPath = ""

		optimizer = Adam(.0002, .5)

		self.discriminator = self.build_discriminator()
		self.discriminator.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics = ['accuracy'])

		self.generator = self.build.generator()

		noise = Input(shape = self.noiseShape)
		image = self.generator(noise)

		self.discriminator.trainable = False

		valid = self.discriminator(image)

		self.combined = Model(noise, valid)
		self.combined.compile(loss = 'binary_crossentropy', optimizer = optimizer)

		epochs = 100
		batchSize = 32
		saveFreq = 10

		y_real = np.ones((batchSize, 1))
		y_fake = np.zeros((batchSize, 1))

		trainingData = load_train_data(self);
		for epoch in range(epochs):
			idx = np.random.randint(0, training_data.shape[0], batchSize)
			x_real = trainingData[idx]

			seed = np.random.normal(0, 1, (batchSize, noiseShape))
			x_fake = generator.predict(seed)

			discriminator_real = discriminator.train_on_batch(x_real, y_real)
			discriminator_fake = discriminator.train_on_batch(x_fake, y_fake)
			discriminator_metric = .5 * np.add(discriminator_real, discriminator_fake)

			generator_metric = combined.train_on_batch(seed, y_real)

			if epoch % saveFreq == 0:
				print("Epoch {epoch}, Discriminator accuracy: {discriminator_metric[1]}, Generator accuracy: {generator_metric[1]}")

		generator.save("generator.h5")


	def save_image(self, epoch):
		r, c = 5, 5
		noise = np.random.normal(0, 1, (r * c, noiseShape[0]))
		generateImg = self.generator.predict(noise)

		generateImg = 0.5 * generateImg + 0.5

		fig, axis = plt.subplots(r, c)
		count = 0
		for i in range(64):
			for j in range(64):
				axis[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap = 'gray')
				axis[i, j].axis('off')
				count += 1
		fig.savefig("sample%d.png" % epoch)
		plt.close()

	def load_train_data(self):
		loadNewData = True
		dataSavePath = "training_data.npy"
		trainingData = []

		if loadNewData:
			for filename in sorted(glob.glob(self.inputPath)):
				image = Image.open(filename)
				origSize = image.size
				squareSize = max(origSize[0], origSize[1])
				newImage = Image.new("RGB", (squareSize, squareSize), (0, 0, 0, 0))
				newImage.paste(image, (int((squareSize - origSize[0])/2), int((squareSize - origSize[1])/2)))
				image = newImage
				image = image.resize((64, 64), Image.ANTIALIAS)
				trainingData.append(np.asarray(image))
			trainingData = np.reshape(trainingData, (-1, imageShape))
			trainingData = trainingData / 127.5 - 1

			np.save(trainingData, trainingData)
		else:
			trainingData = np.load(trainingData)

		return trainingData

	def build_generator(self):
		model = Sequential()
		alpha = 0.0002
		momentum = 0.5

		model.add(Dense(4*4*256, activation = "relu", input_shape = self.noiseShape))
		model.add(Reshape(self.imageShape))

		model.add(UpSampling2D())
		model.add(Conv2D(256, kernel_size = 3, padding = "same"))
		model.add(BatchNormalization(momentum))
		model.add(Activation("relu"))

		model.add(UpSampling2D())
		model.add(Conv2D(256, kernel_size = 3, padding = "same"))

		model.add(BatchNormalization(momentum))
		model.add(Activation("relu"))

		model.add(UpSampling2D())
		model.add(Conv2D(128, kernel_size = 3, padding = "same"))
		model.add(BatchNormalization(momentum))
		model.add(Activation("relu"))

		model.add(UpSampling2D())
		model.add(Conv2D(128, kernel_size = 3, padding = "same"))
		model.add(BatchNormalization(momentum))
		model.add(Activation("relu"))

		model.add(UpSampling2D())
		model.add(Conv2D(128, kernel_size = 3, padding = "same"))
		model.add(BatchNormalization(momentum))
		model.add(Activation("relu"))

		model.add(Conv2D(3, kernel_size = 3, padding = "same"))
		model.add(Activation("tanh"))

		model.summary()

		noise = Input(shape = noiseShape)
		image = model(noise)

		return Model(noise, image)

	def build_discriminator(self):
		model = Sequential()
		alpha = 0.0002
		dropoutRate = 0.25
		momentum = 0.5

		model.add(Conv2D(32, kernel_size = 3, strides = 2, input_shape = imageShape, padding = "same"))
		model.add(LeakyReLU(alpha))

		model.add(Dropout(dropoutRate))
		model.add(Conv2D(64, kernel_size = 3, strides = 2, padding = "same"))
		model.add(ZeroPadding2D(padding = ((0, 1),(0, 1))))
		model.add(BatchNormalization(momentum))
		model.add(LeakyReLU(alpha))

		model.add(Dropout(dropoutRate))
		model.add(Conv2D(128, kernel_size = 3, strides = 2, padding = "same"))
		model.add(BatchNormalization(momentum))
		model.add(LeakyReLU(alpha))

		model.add(Dropout(dropoutRate))
		model.add(Conv2D(256, kernel_size = 3, strides = 1, padding = "same"))
		model.add(BatchNormalization(momentum))
		model.add(LeakyReLU(alpha))

		model.add(Dropout(dropoutRate))
		model.add(Conv2D(512, kernel_size = 3, strides = 1, padding = "same"))
		model.add(BatchNormalization(momentum))
		model.add(LeakyReLU(alpha))

		model.add(Dropout(dropoutRate))
		model.add(Flatten())
		model.add(Dense(1, activation= 'sigmoid'))

		image = Input(shape = self.imageShape)
		validity = model(image)

		return Model(image, validity)







