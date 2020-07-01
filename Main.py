# -*- coding: utf-8 -*-
import sys
from UI import Ui_MainWindow
from PyQt5 import QtWidgets

print("********************注意********************")
print("************先选择网络再进行识别检测************")
app = QtWidgets.QApplication(sys.argv)
ui = Ui_MainWindow()
ui.show()
sys.exit(app.exec_())