###################################### Variational Autoencoder ############################################
## Author: Sara Regina Ferreira de Faria
## Email: sarareginaff@gmail.com

#Needed libraries
import numpy
import matplotlib.pyplot as plt
import pandas
import math
import scipy.io as spio
import scipy.ndimage
from scipy.stats import norm
from keras.models import Model, Sequential
from keras.layers import Input, Dense, LSTM, RepeatVector, Lambda, Layer
from sklearn.metrics import mean_squared_error
from keras import backend as K
from keras import metrics
from keras.datasets import mnist

# fix random seed for reproducibility
numpy.random.seed(7)

# load the dataset
def loadData(file, dictName):
	matfile = file
	matdata = spio.loadmat(matfile)
	dataset = numpy.ndarray(shape=(matdata[dictName].shape[1]), dtype=type(matdata[dictName][0,0]))
	for i in range(matdata[dictName].shape[1]):
		dataset[i] = matdata[dictName][0, i]
	return dataset

# normalize dataset
def normalizeData(data):
	maxVal = numpy.amax(data)
	minVal = numpy.amin(data)
	normalizedData = ((data-minVal)/(maxVal-minVal))
	return normalizedData

# based on http://machinelearningmastery.com/time-series-prediction-with-deep-learning-in-python-with-keras/
# convert an array of values into a dataset matrix
def createMatrix(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back)]
		dataX.append(a)
	return numpy.array(dataX)

# based on https://blog.keras.io/building-autoencoders-in-keras.html
# based on http://machinelearningmastery.com/time-series-prediction-with-deep-learning-in-python-with-keras/
# create lstm-based autoencoder
def trainVariationalAutoencoder(dataset, timesteps, input_dim, bottleneckDim, lossEvaluation, optimizer, epochs, batchSize, verbose=False):
	# split into train and test sets
	train_size = int(len(dataset) * 0.67)
	test_size = len(dataset) - train_size
	train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

	# encoder
	inputs = Input(shape=(timesteps, input_dim))
	encoded = LSTM(bottleneckDim)(inputs)

	# decoder
	decoded = RepeatVector(timesteps)(encoded)
	decoded = LSTM(input_dim, return_sequences=True)(decoded)

	# autoencoder
	model = Model(inputs, decoded)
	model.compile(loss=lossEvaluation, optimizer=optimizer)
	model.fit(train, train, epochs=epochs, batch_size=batchSize, verbose=verbose)

	# Estimate model performance
	trainScore = model.evaluate(train, train, verbose=0)
	print('Train Score: %.6f MSE (%.6f RMSE)' % (trainScore, math.sqrt(trainScore)))
	testScore = model.evaluate(test, test, verbose=0)
	print('Test Score: %.6f MSE (%.6f RMSE)' % (testScore, math.sqrt(testScore)))

	return model

def sampling(args):
    z_mean, z_log_var = args
    x_train_latent_shape = (original_dim[0], latent_dim)
    epsilon = K.random_normal(shape=((batchSizeModel,) + x_train_latent_shape), mean=0., #40, 480, 3, 2
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon


class CustomVariationalLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(CustomVariationalLayer, self).__init__(**kwargs)

    def vae_loss(self, x, x_decoded_mean):
        xent_loss = original_dim[1] * metrics.binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(xent_loss + kl_loss)

    def call(self, inputs):
        x = inputs[0]
        x_decoded_mean = inputs[1]
        loss = self.vae_loss(x, x_decoded_mean)
        self.add_loss(loss, inputs=inputs)
        # We won't actually use the output.
        return x

#************* MAIN *****************#
# variables
look_back = 3
bottleneckDim = 10
lossEvaluation = 'mean_squared_error'
optimizer = 'adam'
fault = False
batchSizeData = 5
epsilon_std = 1.0
batchSizeModel = 3
latent_dim = 2
epochs = 15

# load dataset with all fault simulation
originalDataset = loadData('DadosTodasFalhas.mat', 'Xsep')

# prepare dataset
filteredDataset = scipy.ndimage.filters.gaussian_filter(originalDataset[0][:,:], 4.0)
#filteredDataset = originalDataset[i][:,0]
normalizedDataset = normalizeData(filteredDataset)
dataset = createMatrix(normalizedDataset, look_back)

# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
x_train, x_test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

# get sample size
original_dim = (x_train.shape[1], x_train.shape[2])

# encoder
x = Input(batch_shape=(batchSizeModel,) + original_dim) #batchSizeModel, original_dim (22)
h = Dense(bottleneckDim, activation='relu')(x) #batchSizeModel,bottleneckDim
z_mean = Dense(latent_dim)(h) #batchSizeModel,latent_dim
z_log_var = Dense(latent_dim)(h) #batchSizeModel,latent_dim

z = Lambda(sampling)([z_mean, z_log_var])

# decoder
# we instantiate these layers separately so as to reuse them later
decoder_h = Dense(bottleneckDim, activation='relu') 
decoder_mean = Dense(original_dim[1], activation='sigmoid') 
h_decoded = decoder_h(z) #batchSizeModel,bottleneckDim
x_decoded_mean = decoder_mean(h_decoded) #batchSizeModel,original_dim
print(x_decoded_mean.shape)


y = CustomVariationalLayer()([x, x_decoded_mean]) #batchSizeModel,original_dim 
Model = Model(x, y)
Model.compile(optimizer='rmsprop', loss=None)

Model.fit(x_train, shuffle=True, epochs=epochs, batch_size=batchSizeModel, validation_data=(x_test, x_test))

'''
# using the model
# build a model to project inputs on the latent space
encoder = Model(x, z_mean)

# display a 2D plot of the digit classes in the latent space
x_test_encoded = encoder.predict(x_test, batch_size=batchSizeModel)
plt.figure(figsize=(6, 6))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
plt.colorbar()
plt.show()

# build a digit generator that can sample from the learned distribution
decoder_input = Input(shape=(latent_dim,))
_h_decoded = decoder_h(decoder_input)
_x_decoded_mean = decoder_mean(_h_decoded)
generator = Model(decoder_input, _x_decoded_mean)

# display a 2D manifold of the digits
n = 15  # figure with 15x15 digits
digit_size = 28
figure = numpy.zeros((digit_size * n, digit_size * n))
# linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
# to produce values of the latent variables z, since the prior of the latent space is Gaussian
grid_x = norm.ppf(numpy.linspace(0.05, 0.95, n))
grid_y = norm.ppf(numpy.linspace(0.05, 0.95, n))

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = numpy.array([[xi, yi]])
        x_decoded = generator.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')
plt.show()
'''
'''
# load dataset with all fault simulation
originalDataset = loadData('DadosTodasFalhas.mat', 'Xsep')

# prepare dataset to input model training
filteredDataset = scipy.ndimage.filters.gaussian_filter(originalDataset[0][:,:], 4.0)
#filteredDataset = originalDataset[0][:,:]
normalizedDataset = normalizeData(filteredDataset)
dataset = createMatrix(normalizedDataset, look_back)
#dataset = numpy.reshape(dataset, (dataset.shape[0], dataset.shape[1], 22)) # reshape input to be [samples, time steps, features]

#***** Train model with normal data *****#
# Variables
timesteps = dataset.shape[1]
input_dim = dataset.shape[2]
normalPredict = []
normalError = []
j = 0

# train model
Model = trainVariationalAutoencoder(dataset, timesteps, input_dim, bottleneckDim, lossEvaluation, optimizer, epochs, batchSizeModel, verbose=2)
'''

# get error for each batch of normal data
normalPredict = []
normalError = []
j = 0
for k in range(0,len(dataset),batchSizeModel):
	dataBatch = dataset[k:k+batchSizeModel]	
	normalPredict.append(Model.predict(dataBatch))
	normalError.append(mean_squared_error(dataBatch[:,0,:], normalPredict[j][:,0,:]))
	j += 1

#***** Testing if it is a fault or not *****#
for i in range(1,len(originalDataset)):
	#local variables
	j = 0
	faults = []
	trainPredict = []
	faultError = []
	predicted = []

	# prepare dataset
	filteredDataset = scipy.ndimage.filters.gaussian_filter(originalDataset[i][:,:], 4.0)
	#filteredDataset = originalDataset[i][:,0]
	normalizedDataset = normalizeData(filteredDataset)
	dataset = createMatrix(normalizedDataset, look_back)
	#dataset = numpy.reshape(dataset, (dataset.shape[0], dataset.shape[1], 22)) # reshape input to be [samples, time steps, features]
	
	# get error for each batch of data
	for k in range(0,len(dataset),batchSizeModel):
		dataBatch = dataset[k:k+batchSizeModel]	

		# generate predictions using model
		trainPredict.append(Model.predict(dataBatch))
		predicted.append(trainPredict[j][:,0,:])
		faultError.append(mean_squared_error(dataBatch[:,0,:], predicted[j]))

		# check if it is a fault or not
		if (faultError[j] > normalError[j]*10):
			faults.append(1)
		else:
			faults.append(0)

		j = j + 1

	print("Dataset", i, ". IsFaultVector: ", faults)
	
	#plot baseline and predictions
	#plt.plot(normalizedDataset)
	#plt.plot(numpy.concatenate( predicted, axis=0 ))
	#plt.show()
