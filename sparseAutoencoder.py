###################################### Sparse Autoencoder ############################################
## Author: Sara Regina Ferreira de Faria
## Email: sarareginaff@gmail.com

#Needed libraries
import numpy
import matplotlib.pyplot as plt
import pandas
import math
import scipy.io as spio
import scipy.ndimage
from sklearn.metrics import mean_squared_error, roc_curve, auc
from keras import regularizers

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
def trainSparseAutoencoder(dataset, timesteps, input_dim, bottleneckDim, lossEvaluation, optimizer, epochs, batchSize, regularizer, verbose=False):
	from keras.layers import Input, Dense, LSTM, RepeatVector
	from keras.models import Model

	# split noise and normal data into train and test sets
	train_size = int(len(dataset) * 0.67)
	test_size = len(dataset) - train_size
	train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
	
	# encoder
	inputs = Input(shape=(timesteps, input_dim))
	encoded = LSTM(int(bottleneckDim), activity_regularizer=regularizer)(inputs)

	# decoder
	decoded = RepeatVector(timesteps)(encoded)
	decoded = LSTM(input_dim, return_sequences=True)(decoded)

	# autoencoder
	model = Model(inputs, decoded)
	model.compile(loss=lossEvaluation, optimizer=optimizer)
	model.fit(train, train, epochs=epochs, batch_size=batchSize, verbose=verbose, validation_data=(test, test))

	# Estimate model performance
	#trainScore = model.evaluate(train, train, verbose=0)
	#print('Train Score: %.6f MSE (%.6f RMSE)' % (trainScore, math.sqrt(trainScore)))
	#testScore = model.evaluate(test, test, verbose=0)
	#print('Test Score: %.6f MSE (%.6f RMSE)' % (testScore, math.sqrt(testScore)))

	return model

# based on https://edouardfouche.com/Neural-based-Outlier-Discovery/
def calculateFprTpr (predicted, labels):
	dist = numpy.zeros(len(predicted))
	for i in range(len(predicted)):
	    dist[i] = numpy.linalg.norm(predicted[i])

	fpr, tpr, thresholds = roc_curve(labels, dist)

	return fpr, tpr


#************* MAIN *****************#
# variables
best_roc_auc = 0
best_epochs = 0
best_limit = 0
best_bottleneckDim = 0
best_look_back = 0
best_regularizerIndex = 0

for epochs in range(16,17): #16
	print("epochs", epochs)
	for limitAux in range(11,12):
		limit = limitAux/10
		print("limit", limit)
		for bottleneckDim in range (9,10):
			print("bottleneckDim", bottleneckDim)
			for look_back in range(6,7):
				print("look_back", look_back)
				for regularizerIndex in range (5,6):
					regularizer=regularizers.l1(pow(10,-regularizerIndex))
					print("regularizer", regularizerIndex)
				
					batchSizeData = 1
					#bottleneckDim = batchSizeData/2
					batchSizeModel = 5
					lossEvaluation = 'mean_squared_error'
					optimizer = 'adam'
					roc_auc = []
					FPRs = []
					TPRs = []

					# load dataset with all fault simulation
					originalDataset = loadData('DadosTodasFalhas.mat', 'Xsep')

					# prepare dataset to input model training
					filteredDataset = scipy.ndimage.filters.gaussian_filter(originalDataset[0][:,:], 4.0)
					#filteredDataset = originalDataset[0][:,:]
					normalizedDataset = normalizeData(filteredDataset)
					dataset = createMatrix(normalizedDataset, look_back)
					
					#***** Train model with normal data *****#
					# Variables
					timesteps = dataset.shape[1]
					input_dim = dataset.shape[2]
					normalPredict = []
					normalError = []
					j = 0

					# train model
					Model = trainSparseAutoencoder(dataset, timesteps, input_dim, bottleneckDim, lossEvaluation, optimizer, epochs, batchSizeModel, regularizer, verbose=False)
					
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
						#dataset = numpy.reshape(dataset, (dataset.shape[0], dataset.shape[1], 22)) # reshape input to be [samples, time steps, features]
						
						# get error for each batch of data
						for k in range(0,len(dataset),batchSizeData):
							dataBatch = dataset[k:k+batchSizeData]	

							# generate predictions using model
							trainPredict.append(Model.predict(dataBatch))
							predicted.append(trainPredict[j][:,0,:])
							faultError.append(mean_squared_error(dataBatch[:,0,:], predicted[j]))

							# check if it is a fault or not
							if (faultError[j] > normalError[j]*limit):
								faults.append(1)
							else:
								faults.append(0)

							j = j + 1
						#print("Dataset", i, ". IsFaultVector: ", faults)
						
						# define labels to ROC curve
						labels = []
						for k in range(0,len(dataset),batchSizeData):
							if (k >= 100):
								labels.append(1)
							if (k < 100):
								labels.append(0)

						# calculate AUC, fpr and tpr
						fpr, tpr = calculateFprTpr(faults, labels)
						FPRs.append(fpr)
						TPRs.append(tpr)
						roc_auc.append(auc(fpr, tpr))

						sum_roc_auc = 0
						for i in range(len(roc_auc)):
							sum_roc_auc += roc_auc[i]

						if (sum_roc_auc > best_roc_auc):
							best_roc_auc = sum_roc_auc
							best_epochs = epochs
							best_limit = limit
							best_bottleneckDim = bottleneckDim
							best_look_back = look_back
							best_regularizerIndex = regularizerIndex
						
						#plot baseline and predictions
						#plt.plot(normalizedDataset)
						#plt.plot(numpy.concatenate( predicted, axis=0 ))
						#plt.show()

						#plt.plot(roc_auc)
						#plt.show()


sum_selected_roc_auc = 0
for j in range(len(FPRs)):
	i = j+1
	if(i == 1 or i == 2 or i == 5 or i == 7 or i == 8 or i == 9 or i == 10 or i == 11 or i == 12 or i == 14 or i == 15 or i == 19):
		plt.plot(FPRs[j], TPRs[j], label="AUC{0}= {1:0.2f}".format(i+1, roc_auc[j]))
		sum_selected_roc_auc += roc_auc[j]


plt.xlim((0,1))
plt.ylim((0,1))
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive rate')
plt.ylabel('True Positive rate')
plt.title('ROC curve - Sparse Autoencoder')
plt.legend(loc="lower right")
plt.show()

print("bests parameters")
print("best_limit", best_limit)
print("best_epochs", best_epochs)
print("best_roc_auc", best_roc_auc)
print("best_look_back", best_look_back)
print("best_bottleneckDim", best_bottleneckDim)
print("best_regularizerIndex", best_regularizerIndex)
print("sum_selected_roc_auc", sum_selected_roc_auc)
