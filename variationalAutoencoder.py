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
from sklearn.metrics import mean_squared_error, roc_curve, auc
from keras import backend as K
from keras import metrics

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

# based on https://edouardfouche.com/Neural-based-Outlier-Discovery/
def calculateFprTpr (predicted, labels):
	dist = numpy.zeros(len(predicted))
	for i in range(len(predicted)):
	    dist[i] = numpy.linalg.norm(predicted[i])

	fpr, tpr, thresholds = roc_curve(labels, dist)

	return fpr, tpr


#************* MAIN *****************#
# variables
look_back = 3
bottleneckDim = 10
lossEvaluation = 'mean_squared_error'
optimizer = 'adam'
epsilon_std = 1.0
batchSizeModel = 3
latent_dim = 2
epochs = 15
roc_auc = []
FPRs = []
TPRs = []

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

# autoencodoer
y = CustomVariationalLayer()([x, x_decoded_mean]) #batchSizeModel,original_dim 
Model = Model(x, y)
Model.compile(optimizer='rmsprop', loss=None)

# Train model with normal data
Model.fit(x_train, shuffle=True, epochs=epochs, batch_size=batchSizeModel, validation_data=(x_test, x_test), verbose=2)


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

	# get error for each batch of data
	for k in range(0,len(dataset),batchSizeModel):
		dataBatch = dataset[k:k+batchSizeModel]	

		# generate predictions using model
		trainPredict.append(Model.predict(dataBatch))
		predicted.append(trainPredict[j][:,0,:])
		faultError.append(mean_squared_error(dataBatch[:,0,:], predicted[j]))

		# check if it is a fault or not
		if (faultError[j] > normalError[j]*1.3):
			faults.append(1)
		else:
			faults.append(0)

		j = j + 1

	#print("Dataset", i, ". IsFaultVector: ", faults)
	

	# define labels to ROC curve
	labels = []
	for k in range(0,len(dataset),batchSizeModel):
		if (k >= 100):
			labels.append(1)
		if (k < 100):
			labels.append(0)

	# calculate AUC, fpr and tpr
	fpr, tpr = calculateFprTpr(faults, labels)
	FPRs.append(fpr)
	TPRs.append(tpr)
	roc_auc.append(auc(fpr, tpr))

for i in range(len(FPRs)):
	plt.plot(FPRs[i], TPRs[i], label="AUC{0}= {1:0.2f}".format(i+1, roc_auc[i]))
plt.xlim((0,1))
plt.ylim((0,1))
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive rate')
plt.ylabel('True Positive rate')
plt.title('ROC curve')
plt.legend(loc="lower right")
plt.show()

	#plot baseline and predictions
	#plt.plot(normalizedDataset)
	#plt.plot(numpy.concatenate( predicted, axis=0 ))
	#plt.show()

#plt.plot(roc_auc)
#plt.show()