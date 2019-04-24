
import matplotlib.pylab as plt
import seaborn as sns
#sns.despine()

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
#from keras.layers import Merge
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping
from keras.optimizers import RMSprop, Adam, SGD, Nadam
from keras.layers.advanced_activations import *
from keras.layers import Convolution1D, MaxPooling1D, AtrousConvolution1D
from keras.layers.recurrent import LSTM, GRU
from keras import regularizers

from PyQt5 import QtCore, QtGui, QtWidgets

import pandas as pd, numpy as np


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import theano
theano.config.compute_test_value = "ignore"

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
from matplotlib.figure import Figure

class MplCanvas(Canvas):
	def __init__(self):
		self.fig = Figure()
		Canvas.__init__(self, self.fig)
		Canvas.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
		Canvas.updateGeometry(self)

	def plot1(self, data):
		self.ax = self.fig.add_subplot(111)
		self.ax.plot(data)

	def plot2(self, history):
		self.ax = self.fig.add_subplot(111)
		self.ax.plot(history.history['loss'])
		self.ax.plot(history.history['val_loss'])
		self.ax.set_title('model loss')
		self.ax.set_ylabel('loss')
		self.ax.set_xlabel('epoch')
		self.ax.legend(['train', 'test'], loc='best')

	def plot3(self, history):
		self.ax = self.fig.add_subplot(111)
		self.ax.plot(history.history['acc'])
		self.ax.plot(history.history['val_acc'])
		self.ax.set_title('model accuracy')
		self.ax.set_ylabel('accuracy')
		self.ax.set_xlabel('epoch')
		self.ax.legend(['train', 'test'], loc='best')

	def plot4(self, orig, pred, FROM, TO):
		self.ax = self.fig.add_subplot(111)
		self.ax.plot(orig, color='black', label = 'Original data')
		self.ax.plot(pred, color='blue', label = 'Predicted data')
		self.ax.legend(loc='best')
		self.ax.set_title('Actual and predicted from point %d to point %d of test set' % (FROM, TO))

class Model():
	WINDOW = 30
	EMB_SIZE = 1
	STEP = 1
	FORECAST = 5
	EPOCHS = 0
	X, Y = [], []

	def __init__(self, widgets):
		self.w = widgets

	def set_num_epoch(self, num):
		self.EPOCHS = num

	def setFile(self, temp, lineEdit):
		self.filepath = temp
		lineEdit.setText(temp)
		self.data = pd.read_csv(self.filepath)[::-1]
		self.data = self.data.ix[:, 'Adj Close'].tolist()
		self.canvas = MplCanvas() 
		self.canvas.plot1(self.data)
		self.w[0].addWidget(self.canvas)


	def learn(self, memo):
		#self.setLayout(widget)
		for i in range(0, len(self.data), self.STEP): 
			try:
				x_i = self.data[i:i+self.WINDOW]
				y_i = self.data[i+self.WINDOW+self.FORECAST]  

				last_close = x_i[self.WINDOW-1]
				next_close = y_i

				if last_close < next_close:
					y_i = [1, 0]
				else:
					y_i = [0, 1] 

			except Exception as e:
				print(e)
				break

			self.X.append(x_i)
			self.Y.append(y_i)

		self.X = [(np.array(x) - np.mean(x)) / np.std(x) for x in self.X] # comment it to remove normalization
		self.X, self.Y = np.array(self.X), np.array(self.Y)

		self.X_train, self.X_test, self.Y_train, self.Y_test = self.create_Xt_Yt(self.X, self.Y)

		self.model = Sequential()
		self.model.add(Dense(64, input_dim=30,
						activity_regularizer=regularizers.l2(0.01)))
		self.model.add(BatchNormalization())
		self.model.add(LeakyReLU())

		self.model.add(Dropout(0.5))
		self.model.add(Dense(16,
						activity_regularizer=regularizers.l2(0.01)))
		self.model.add(BatchNormalization())
		self.model.add(LeakyReLU())
		self.model.add(Dense(2))
		self.model.add(Activation('softmax'))

		opt = Nadam(lr=0.001)

		self.reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.9, patience=25, min_lr=0.000001, verbose=1)
		self.checkpointer = ModelCheckpoint(filepath="test.hdf5", verbose=1, save_best_only=True)
		self.model.compile(optimizer=opt, 
					  loss='categorical_crossentropy',
					  metrics=['accuracy'])

		self.history = self.model.fit(self.X_train, self.Y_train, 
				nb_epoch = self.EPOCHS, 
				batch_size = 128, 
				verbose=1, 
				validation_data=(self.X_test, self.Y_test),
				callbacks=[self.reduce_lr, self.checkpointer],
				shuffle=True)
		self.canvas1 = MplCanvas()
		self.canvas1.plot2(self.history)
		self.canvas2 = MplCanvas()
		self.canvas2.plot3(self.history)
		self.w[1].addWidget(self.canvas1)
		self.w[2].addWidget(self.canvas2)

		pred = self.model.predict(np.array(self.X_test))
		C = confusion_matrix([np.argmax(y) for y in self.Y_test], [np.argmax(y) for y in pred])

		memo.setPlainText(str(C / C.astype(np.float).sum(axis=1)))

		FROM = 0
		TO = FROM + 500
		self.original = self.Y_test[FROM:TO]
		self.predicted = pred[FROM:TO]

		self.canvas3 = MplCanvas()
		self.canvas3.plot4(self.original, self.predicted, FROM, TO)
		self.w[3].addWidget(self.canvas3)

	def shuffle_in_unison(self, a, b):
		"""courtsey http://stackoverflow.com/users/190280/josh-bleecher-snyder"""
		assert len(a) == len(b)
		shuffled_a = np.empty(a.shape, dtype=a.dtype)
		shuffled_b = np.empty(b.shape, dtype=b.dtype)
		permutation = np.random.permutation(len(a))
		for old_index, new_index in enumerate(permutation):
			shuffled_a[new_index] = a[old_index]
			shuffled_b[new_index] = b[old_index]
		return shuffled_a, shuffled_b
 
	def create_Xt_Yt(self, X, y, percentage=0.9):
		p = int(len(X) * percentage)
		self.X_train = X[0:p]
		self.Y_train = y[0:p]
		 
		self.X_train, self.Y_train = self.shuffle_in_unison(self.X_train, self.Y_train)

		self.X_test = X[p:]
		self.Y_test = y[p:]

		return self.X_train, self.X_test, self.Y_train, self.Y_test
