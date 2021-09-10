import sys
import os
import subprocess
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import data_processing as dp
import lpc_hmm as new_method
import matplotlib.pyplot as plt
import numpy as np
import librosa as lr

class editwindow(QWidget):
    def __init__(self):

        super(editwindow,self).__init__()

        layout = QVBoxLayout()
        self.setLayout(layout)
        self.setWindowTitle("Berikan Nama File Suara")

        self.editname = QTextEdit(self)
        self.editname.setPlaceholderText("Berikan Nama File Suara")

        self.btnname = QPushButton(self)
        self.btnname.setText("OK")
        self.btnname.clicked.connect(self.record_ok_clicked)

        layout.addWidget(self.editname)
        layout.addWidget(self.btnname)

    @pyqtSlot()
    def record_ok_clicked(self):

        QMessageBox().information(self, 'Information', 'Ready to Record')
        QMessageBox.show(self)

        result, label_predict = new_method.record(self.editname.toPlainText())

        QMessageBox().information(self, 'Information', str(result))
        QMessageBox.show(self)

        #self.result_label.setText(label_predict)

        #label_predict = ""

        if label_predict == 'linkedin':
            os.system(r'cmd /c "start C:\Users\chondroseto\Desktop\LinkedIn"')
        elif label_predict == 'whatsapp':
            os.system(r'cmd /c "start C:\Users\chondroseto\Desktop\Whatsapp"')
        elif label_predict == 'gmail':
            os.system(r'cmd /c "start C:\Users\chondroseto\Desktop\Gmail"')
        elif label_predict == 'tokopedia':
            os.system(r'cmd /c "start C:\Users\chondroseto\Desktop\Tokopedia"')
        elif label_predict == 'powerpoint':
            subprocess.call(r"C:\Program Files\Microsoft Office\root\Office16\POWERPNT.EXE")
        elif label_predict == 'word':
            subprocess.call(r"C:\Program Files\Microsoft Office\root\Office16\WINWORD.EXE")

        #self.close()


class AudioProc(QMainWindow):
    def __init__(self):

        super(AudioProc,self).__init__()
        loadUi('gui.ui',self)
        self.train_btn.clicked.connect(self.train_btn_clicked)
        self.record_btn.clicked.connect(self.record_btn_clicked)
        self.test_btn.clicked.connect(self.test_btn_clicked)
        self.app_btn.clicked.connect(self.app_btn_clicked)
        self.help_btn.clicked.connect(self.help_btn_clicked)

        self.result_label = QLabel(self)
        self.result_label.setStyleSheet("border : 1px solid black;")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.move(230, 150)
        self.result_label.resize(90,30)
        self.result_label.setText("Output")

        self.info_train = QLabel(self)
        self.info_train.move(325, 25)
        self.info_train.resize(200, 30)
        self.info_train.setText("<-- Tombol training data suara")
        self.info_train.setHidden(True)

        self.info_record = QLabel(self)
        self.info_record.move(325, 55)
        self.info_record.resize(500, 30)
        self.info_record.setText("<-- Tombol record dan test suara")
        self.info_record.setHidden(True)

        self.info_test = QLabel(self)
        self.info_test.move(325, 85)
        self.info_test.resize(500, 30)
        self.info_test.setText("<-- Tombol load file suara dan test file suara")
        self.info_test.setHidden(True)

        self.info_tests = QLabel(self)
        self.info_tests.move(325, 120)
        self.info_tests.resize(200, 30)
        self.info_tests.setText("<-- Tombol load semua file suara dan \n       test semua file suara")
        self.info_tests.setHidden(True)

        self.info_output = QLabel(self)
        self.info_output.move(325, 150)
        self.info_output.resize(200, 30)
        self.info_output.setText("<-- Tempat hasil deteksi akan tampil")
        self.info_output.setHidden(True)



    @pyqtSlot()
    def train_btn_clicked(self):
        QMessageBox().information(self, 'Information', 'Training Start')
        QMessageBox.show(self)
        #method.lpc_hmm_train()
        new_method.lpc_hmm_train()
        QMessageBox().information(self,'Information','Training Done')
        QMessageBox.show(self)

    @pyqtSlot()
    def record_btn_clicked(self):
        self.en=editwindow()
        self.en.show()

    @pyqtSlot()
    def test_btn_clicked(self):
        fname, unused = QFileDialog.getOpenFileName(self, 'Open file','C:\\Users\\chondroseto\\PycharmProjects\\assist\\code\\umum',"Audio files (*.wav)")
        if len(fname)>0:
            QMessageBox().information(self, 'Information', 'Load File Success')
            QMessageBox.show(self)

            result,label_actual, label_predict,status =  new_method.lpc_hmm_uji_one(fname)
            dp.LPCode(fname)

            QMessageBox().information(self, 'Information', str(result))
            QMessageBox.show(self)
            #status='undetected'
            if len(label_predict)>1:
                self.result_label.setText(label_predict)
            else:
                self.result_label.setText("undetected")

            #label_predict = ""

            if status=='detected':
                if label_predict == 'linkedin':
                    os.system(r'cmd /c "start C:\Users\chondroseto\Desktop\LinkedIn"')
                elif label_predict == 'whatsapp':
                    os.system(r'cmd /c "start C:\Users\chondroseto\Desktop\Whatsapp"')
                elif label_predict == 'gmail':
                    os.system(r'cmd /c "start C:\Users\chondroseto\Desktop\Gmail"')
                elif label_predict == 'tokopedia':
                    os.system(r'cmd /c "start C:\Users\chondroseto\Desktop\Tokopedia"')
                elif label_predict == 'powerpoint':
                    subprocess.call(r"C:\Program Files\Microsoft Office\root\Office16\POWERPNT.EXE")
                elif label_predict == 'word':
                    subprocess.call(r"C:\Program Files\Microsoft Office\root\Office16\WINWORD.EXE")


    @pyqtSlot()
    def app_btn_clicked(self):
        QMessageBox().information(self, 'Information', 'Test All Start')
        QMessageBox.show(self)

        result=new_method.lpc_hmm_uji_all()

        #result=method.lpc_hmm_uji_all()
        QMessageBox().information(self, 'Information', result)
        QMessageBox.show(self)

    @pyqtSlot()
    def help_btn_clicked(self):
        if self.info_train.isHidden():
            self.info_train.setHidden(False)
            self.info_record.setHidden(False)
            self.info_test.setHidden(False)
            self.info_tests.setHidden(False)
            self.info_output.setHidden(False)
        else:
            self.info_train.setHidden(True)
            self.info_record.setHidden(True)
            self.info_test.setHidden(True)
            self.info_tests.setHidden(True)
            self.info_output.setHidden(True)


if __name__=='__main__':
    app=QApplication(sys.argv)
    window=AudioProc()
    window.setWindowTitle('Virtual Assisten')
    window.show()
    sys.exit(app.exec_())
