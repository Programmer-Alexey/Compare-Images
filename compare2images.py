# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'compare2images.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(951, 639)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        MainWindow.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setSpacing(5)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        spacerItem = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem)
        self.left_label = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.left_label.sizePolicy().hasHeightForWidth())
        self.left_label.setSizePolicy(sizePolicy)
        self.left_label.setMinimumSize(QtCore.QSize(311, 321))
        self.left_label.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"border-radius: 10px;")
        self.left_label.setText("")
        self.left_label.setObjectName("left_label")
        self.horizontalLayout_3.addWidget(self.left_label)
        self.right_label = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.right_label.sizePolicy().hasHeightForWidth())
        self.right_label.setSizePolicy(sizePolicy)
        self.right_label.setMinimumSize(QtCore.QSize(311, 321))
        self.right_label.setStyleSheet("QLabel{\n"
"    background-color: rgb(255, 255, 255);\n"
"    border-radius: 10px;\n"
"}")
        self.right_label.setText("")
        self.right_label.setObjectName("right_label")
        self.horizontalLayout_3.addWidget(self.right_label)
        self.button_next = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.button_next.sizePolicy().hasHeightForWidth())
        self.button_next.setSizePolicy(sizePolicy)
        self.button_next.setStyleSheet("QPushButton{\n"
"    color: rgb(255, 255, 255);\n"
"    font: 40pt \"Californian FB\";\n"
"}\n"
"QPushButton:hover{\n"
"    color: rgb(100, 100, 100);\n"
"}")
        self.button_next.setObjectName("button_next")
        self.horizontalLayout_3.addWidget(self.button_next)
        self.verticalLayout_2.addLayout(self.horizontalLayout_3)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        spacerItem1 = QtWidgets.QSpacerItem(390, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem1)
        self.score_label = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.score_label.sizePolicy().hasHeightForWidth())
        self.score_label.setSizePolicy(sizePolicy)
        self.score_label.setStyleSheet("font: 24pt \"Tw Cen MT Condensed\";\n"
"color: rgb(255, 112, 112);")
        self.score_label.setObjectName("score_label")
        self.horizontalLayout_2.addWidget(self.score_label)
        spacerItem2 = QtWidgets.QSpacerItem(250, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem2)
        self.btn_send = QtWidgets.QPushButton(self.centralwidget)
        self.btn_send.setMinimumSize(QtCore.QSize(130, 40))
        self.btn_send.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self.btn_send.setMouseTracking(False)
        self.btn_send.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
        self.btn_send.setStyleSheet("QPushButton {\n"
"    background-color: rgb(255, 255, 255);    \n"
"    font: 75 12pt \"Bahnschrift\";\n"
"    padding-bottom: 5px;\n"
"    border-radius: 7px;\n"
"}\n"
"QPushButton:hover{\n"
"    background-color: rgb(255, 128, 55);\n"
"}")
        self.btn_send.setObjectName("btn_send")
        self.horizontalLayout_2.addWidget(self.btn_send)
        spacerItem3 = QtWidgets.QSpacerItem(90, 20, QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem3)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.text_label = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.text_label.sizePolicy().hasHeightForWidth())
        self.text_label.setSizePolicy(sizePolicy)
        self.text_label.setMinimumSize(QtCore.QSize(300, 50))
        self.text_label.setMaximumSize(QtCore.QSize(16777215, 300))
        self.text_label.setStyleSheet("QLabel{\n"
"    text-align: center;\n"
"    color: rgb(255, 112, 112);\n"
"    \n"
"    font: 75 8pt \"System\";\n"
"}")
        self.text_label.setObjectName("text_label")
        self.verticalLayout.addWidget(self.text_label)
        self.verticalLayout_2.addLayout(self.verticalLayout)
        spacerItem4 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        self.verticalLayout_2.addItem(spacerItem4)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setContentsMargins(-1, 50, -1, -1)
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.compare2images = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.compare2images.sizePolicy().hasHeightForWidth())
        self.compare2images.setSizePolicy(sizePolicy)
        self.compare2images.setMaximumSize(QtCore.QSize(16777215, 60))
        self.compare2images.setStyleSheet("QPushButton {\n"
"    background-color: rgb(255, 255, 255);    \n"
"    font: 75 12pt \"Bahnschrift\";\n"
"    padding-bottom: 5px;\n"
"    border-radius: 7px;\n"
"}\n"
"QPushButton:hover{\n"
"    background-color: rgb(255, 128, 55);\n"
"}")
        self.compare2images.setObjectName("compare2images")
        self.horizontalLayout.addWidget(self.compare2images)
        self.find_similar = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.find_similar.sizePolicy().hasHeightForWidth())
        self.find_similar.setSizePolicy(sizePolicy)
        self.find_similar.setMaximumSize(QtCore.QSize(16777215, 60))
        self.find_similar.setStyleSheet("QPushButton {\n"
"    background-color: rgb(255, 255, 255);    \n"
"    font: 75 12pt \"Bahnschrift\";\n"
"    padding-bottom: 5px;\n"
"    border-radius: 7px;\n"
"}\n"
"QPushButton:hover{\n"
"    background-color: rgb(255, 128, 55);\n"
"}")
        self.find_similar.setObjectName("find_similar")
        self.horizontalLayout.addWidget(self.find_similar)
        self.who_am_i = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.who_am_i.sizePolicy().hasHeightForWidth())
        self.who_am_i.setSizePolicy(sizePolicy)
        self.who_am_i.setMaximumSize(QtCore.QSize(16777215, 60))
        self.who_am_i.setStyleSheet("QPushButton {\n"
"    background-color: rgb(255, 255, 255);    \n"
"    font: 75 12pt \"Bahnschrift\";\n"
"    padding-bottom: 5px;\n"
"    border-radius: 7px;\n"
"}\n"
"QPushButton:hover{\n"
"    background-color: rgb(255, 128, 55);\n"
"}")
        self.who_am_i.setObjectName("who_am_i")
        self.horizontalLayout.addWidget(self.who_am_i)
        self.verticalLayout_2.addLayout(self.horizontalLayout)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.button_next.setText(_translate("MainWindow", ">"))  # NEXT
        self.score_label.setText(_translate("MainWindow", "96.8"))# SCORE
        self.btn_send.setText(_translate("MainWindow", "Отправить"))# SEND
        self.text_label.setText(_translate("MainWindow", "                                                                 Ваш результат: 96.8%, вы однозначно похожи!"))# SEND
        self.compare2images.setText(_translate("MainWindow", "Сравнить изображения"))
        self.find_similar.setText(_translate("MainWindow", "Найти похожие"))
        self.who_am_i.setText(_translate("MainWindow", "На кого я похож?"))