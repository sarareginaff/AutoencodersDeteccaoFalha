###################################### Stacked Autoencoder ############################################
## Author: Sara Regina Ferreira de Faria
## Email: sarareginaff@gmail.com

#Needed libraries
import numpy
import matplotlib.pyplot as plt
import pandas
import math
import scipy.io as spio
import scipy.ndimage
from keras.models import Model, Sequential
from keras.layers import Input, Dense, LSTM, RepeatVector
from sklearn.metrics import mean_squared_error

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
def trainStackedAutoencoder(dataset, timesteps, input_dim, bottleneckDim, lossEvaluation, optimizer, epochs, batchSize, verbose=False):
	# split noise and normal data into train and test sets
	train_size = int(len(dataset) * 0.67)
	test_size = len(dataset) - train_size
	train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
	
 	# encoder
	inputs = Input(shape=(timesteps, input_dim))
	encoded = LSTM(bottleneckDim)(inputs)
	encoded = Dense(2, activation='relu')(encoded)
	encoded = Dense(1, activation='relu')(encoded)

	# decoder
	decoded = RepeatVector(timesteps)(encoded)
	decoded = Dense(2, activation='relu')(decoded)
	decoded = Dense(bottleneckDim, activation='relu')(decoded)
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


#************* MAIN *****************#
# variables
look_back = 3
bottleneckDim = 5
epochs = 5
batchSizeModel = 1
lossEvaluation = 'mean_squared_error'
optimizer = 'adam'
fault = False
batchSizeData = 5

# load dataset with all fault simulation
originalDataset = loadData('DadosTodasFalhas.mat', 'Xsep')

# prepare dataset to input model training
filteredDataset = scipy.ndimage.filters.gaussian_filter(originalDataset[0][:,:], 4.0)
#filteredDataset = originalDataset[0][:,:]
normalizedDataset = normalizeData(filteredDataset)
dataset = createMatrix(normalizedDataset, look_back)

#***** Train model with noise normal data *****#
# Variables
timesteps = dataset.shape[1]
input_dim = dataset.shape[2]
normalPredict = []
normalError = []
j = 0

# train model
Model = trainStackedAutoencoder(dataset, timesteps, input_dim, bottleneckDim, lossEvaluation, optimizer, epochs, batchSizeModel, verbose=2)

# get error for each batch of normal data
for k in range(0,len(dataset),batchSizeData):
	dataBatch = dataset[k:k+batchSizeData]	
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
	
	# get error for each batch of data
	for k in range(0,len(dataset),batchSizeData):
		dataBatch = dataset[k:k+batchSizeData]	

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
