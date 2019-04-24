from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QSize, Qt, pyqtSlot
from PyQt5.QtGui import QBrush, QPen, QPixmap
from PyQt5.QtWidgets import (QFileDialog, QGraphicsPixmapItem, QTableWidgetItem, QGraphicsScene, QDialogButtonBox,
							 QLabel, QMainWindow, QPushButton, QDialog)


from PyQt5.uic import loadUi
from model import Model

class Window(QMainWindow):

	def __init__(self):
		QMainWindow.__init__(self)
		loadUi('ui.ui', self)
		self.netw = Model([self.verticalLayout_2, self.verticalLayout_3, self.verticalLayout_4, self.verticalLayout_6])


	@pyqtSlot()
	def on_ln_clicked(self):
		self.netw.set_num_epoch(self.spinBox.value())
		self.netw.learn(self.plainTextEdit)

	@pyqtSlot()
	def on_file_clicked(self):
		temp = QFileDialog.getOpenFileName(self, 'Open file', '', 'CSV Tables (*.csv)')[0]
		self.netw.setFile(temp, self.lineEdit)