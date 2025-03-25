import sys
import pandas as pd
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QComboBox, QFileDialog, QMainWindow
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import transform as tf
# from function import *


# class function(QMainWindow,Ui_xunlian_MainWindow):
#     def __init__(self,parent=None):
#         super(function, self).__init__(parent)
#         self.setupUi(self)
#     def show_function(self):
#         self.show()  # 显示function界面
#         # self.init()  # 重新初始化按钮信号与槽的连接
#         return None

class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4):
        fig = Figure(figsize=(width, height))
        self.axes = fig.add_subplot(111)
        super(PlotCanvas, self).__init__(fig)

    def plot(self, data):
        self.axes.clear()
        self.axes.plot(data)
        self.draw()

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
    def initUI(self):
        self.setGeometry(300, 300, 600, 400)
        self.setWindowTitle('绘制')

        layout = QVBoxLayout()

        self.fileButton = QPushButton('打开文件', self)
        self.fileButton.clicked.connect(self.openFile)
        layout.addWidget(self.fileButton)

        self.columnComboBox = QComboBox(self)
        self.columnComboBox.setEditable(True)
        layout.addWidget(self.columnComboBox)

        self.plotButton = QPushButton('绘图', self)
        self.plotButton.clicked.connect(self.plotGraph)
        layout.addWidget(self.plotButton)

        self.returnButton = QPushButton('返回', self)
        # self.returnButton.clicked.connect(self.fanhui)
        layout.addWidget(self.returnButton)


        self.canvas = PlotCanvas(self, width=5, height=4)
        layout.addWidget(self.canvas)

        self.setLayout(layout)


    def openFile(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "Open File", "", "Excel Files (*.xlsx *.xls);;CSV Files (*.csv)", options=options)
        if fileName:
            self.file = fileName
            self.data = self.readData(fileName)
            self.columnComboBox.clear()
            self.columnComboBox.addItems(self.data.columns.tolist())

    def readData(self, file):
        if file.endswith('.csv'):
            df = pd.read_csv(file, skiprows=7)
            df = df.iloc[1:]
            cols_nulls = df.columns[df.isnull().any()]
            df[cols_nulls] = df[cols_nulls].fillna(method='ffill')
            df['Unnamed: 0'] = pd.to_datetime(df['Unnamed: 0'], format='%H:%M:%S')
            df['DMU INTERNAL FLIGHT PHASE'] = df['DMU INTERNAL FLIGHT PHASE'].astype(int)
            df1 = tf.long_cruise(df)
            return df1
        else:
            return pd.read_excel(file)

    def plotGraph(self):
        if self.data is not None and self.columnComboBox.currentIndex() != -1:
            column = self.columnComboBox.currentText()
            self.canvas.plot(self.data[column].values)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec_())