# _*_ coding:utf-8 _*_
# 开发人员: LingJiabin
# 开发工具: PyCharm
# 开发日期: 2020/3/21 13:16
# 文件名  : demo1.py


import os
from PySide2.QtWidgets import *
from PySide2.QtCharts import *
from PySide2.QtGui import *


app = QApplication()

Pid = QtCharts.QPieSeries()
Pid.append("Jane",1)
Pid.append("Joe",2)
Pid.append("Andy",3)
Pid.append("Barbara",4)
Pid.append("Axel",5)

Pid.setLabelsVisible()


'''
ChartS = QGraphicsScene()
ChartS.addItem(Chart)

ChartS.setSceneRect(0, 0, 280, 280)
ui.graphicsView.(ChartS)
ui.graphicsView.setRenderHint(QPainter.Antialiasing)


ui.show()

'''
Bar = QtCharts.QBarSeries()
set = QtCharts.QBarSet('dddd')
set1 = QtCharts.QBarSet('eeee')
QX = QtCharts.QBarCategoryAxis()
QY = QtCharts.QValueAxis()
set.append(5)
set1.append(6)
QX.append('1')
QY.setRange(0, 10)

Chart = QtCharts.QChart()
Chart.addSeries(Bar)
Chart.setTitle('饼图')

Chart.addAxis(QX, Qt.AlignBottom)
Bar.attachAxis(QX)
Chart.addAxis(QY, Qt.AlignLeft)
Bar.attachAxis(QY)

Bar.append(set)


set = QtCharts.QBarSet('kkkk')
set.append(9)
Bar.append(set)
Bar.append(set1)



ChartView = QtCharts.QChartView(Chart)
ChartView.setRenderHint(QPainter.Antialiasing)

ChartView.resize(500,500)
ChartView.setWindowTitle('Sf')
ChartView.show()
app.exec_()


