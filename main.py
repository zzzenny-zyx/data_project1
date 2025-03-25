import sys

from PyQt5.QtWidgets import *
from PyQt5 import QtWidgets
from denglu import Ui_login_MainWindow
from xunlian import *
from function import Ui_xunlian_MainWindow
from linewidget import *
from huizhi import App

class function(QMainWindow,Ui_xunlian_MainWindow):
    def __init__(self,parent=None):
        super(function, self).__init__(parent)
        self.setupUi(self)
        self.init()
        self.moxing = xunlian_window(self)
        self.chakan = App()
    def init(self):
        self.pushButton_xunlian.clicked.connect(self.zhanshi_xunlian)  # 连接槽
        self.pushButton_chakan.clicked.connect(self.zhanshi_chakan)  # 连接槽
    def zhanshi_xunlian(self):
        moxing.show()
        self.hide()
        return None
    def zhanshi_chakan(self):
        chakan.show()
        self.hide()
        return None
    def show_function(self):
        self.show()  # 显示function界面
        self.init()  # 重新初始化按钮信号与槽的连接
        return None

class huizhi(App):
    # def __init__(self, parent=None):
    #     super(huizhi, self).__init__(parent)
    #     self.initUI()
    #     self.init()
    def __init__(self):
        super().__init__()
        self.initUI()
        self.init()
    def init(self):
        self.returnButton.clicked.connect(self.fanhui)
    def fanhui(self):
        gongneng.show()
        gongneng.show_function()
        self.hide()

class xunlian_window(QMainWindow,Ui_MainWindow):
    def __init__(self, parent=None):
        super(xunlian_window, self).__init__(parent)
        self.setupUi(self)
        self.init()
        self.gongneng = function
    def init(self):
        self.returnButton.clicked.connect(self.fanhuift)  # 连接槽
    def fanhuift(self):
        gongneng.show()
        gongneng.show_function()
        self.hide()
        return None

class login(QMainWindow, Ui_login_MainWindow):
    def __init__(self, parent=None):
        super(login, self).__init__(parent)
        self.setupUi(self)
        self.init()
        self.admin = "y"
        self.Password = "666"
    def init(self):
        self.pushButton_denglu.clicked.connect(self.login_button)  # 连接槽
    def login_button(self):
        if self.lineEdit_mima.text() == "":
            QMessageBox.warning(self, '警告', '密码不能为空，请输入！')
            return None
        # if  self.password == self.lineEdit.text():
        if (self.lineEdit_mima.text() == self.Password) and (self.lineEdit_yonghu.text() == self.admin):
            # 1打开新窗口
            # moxing.show()
            gongneng.show()
            # 2关闭本窗口
            self.close()
        else:
            QMessageBox.critical(self, '错误', '密码错误！')
            self.lineEdit.clear()
            return None




if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = login()
    gongneng = function()
    chakan = App()
    moxing = xunlian_window()
    main.show()
    sys.exit(app.exec_())

    # app = QtWidgets.QApplication(sys.argv)
    # # 实例化界面类
    # main_window = LineMainWidget()
    # # 显示界面
    # main_window.show()
    # # 启动事件循环
    # sys.exit(app.exec_())


