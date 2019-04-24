import sys, os

from PIL import Image, ImageDraw
from PyQt5 import QtWidgets, QtGui

from window import Window

if __name__ == "__main__":
	app = QtWidgets.QApplication(sys.argv)
	mainWin = Window()
	mainWin.show()
	sys.exit(app.exec_())
