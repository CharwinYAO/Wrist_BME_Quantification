# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


from PyQt5.QtWidgets import *
import sys
from MyDialog import Ui_Dialog  # 导入GUI文件
from MyFigure import *  # 嵌入了matplotlib的文件
from pathlib import Path
import nibabel as nib
import cluster
import numpy as np
from onnxruntime import InferenceSession

class MainDialogImgBW(QDialog, Ui_Dialog):
    def __init__(self):
        super(MainDialogImgBW, self).__init__()
        self.setupUi(self)

        self.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        self.setWindowTitle("Show nii Image")
        self.setMinimumSize(0, 0)

        # 创建存放nii文件路径的属性
        self.nii_path = ''
        # 创建存放mask文件路径的属性
        self.mask_path = ''
        # 创建记录nii文件里面图片数量的属性
        self.shape = 1
        # 创建用于检查radio button选择标记的属性，选择'nii图像'，为0，现在‘mask图像’，为1
        self.check = 0
        # 创建用于检查是否画HISTOGRAM
        self.hist = 0
        # 创建存放nii数据的属性。
        self.img = 0
        self.seg = 0

        self.clm = 0

        # 定义MyFigure类的一个实例
        self.F = MyFigure(width=3, height=2, dpi=100)
        # 在GUI的groupBox中创建一个布局，用于添加MyFigure类的实例（即图形）后其他部件。
        self.gridlayout = QGridLayout(self.groupBox)  # 继承容器groupBox
        self.gridlayout.addWidget(self.F, 0, 1)
        self.pushButton.clicked.connect(self.bindButton)
        self.pushButton_2.clicked.connect(self.bindButton2)
        self.pushButton_3.clicked.connect(self.bindButton3)
        self.pushButton_4.clicked.connect(self.bindButton4)

        self.pushButton_5.clicked.connect(self.quantify_BME)
        self.pushButton_6.clicked.connect(self.quantify_the_selected_bones)

        self.horizontalSlider.valueChanged.connect(self.bindSlider)
        self.horizontalSlider_2.valueChanged.connect(self.bindSlider_2)
        self.horizontalSlider_3.valueChanged.connect(self.bindSlider_3)
        self.horizontalSlider_4.valueChanged.connect(self.bindSlider_4)

        self.radioButton.clicked.connect(self.bindradiobutton)
        self.radioButton_2.clicked.connect(self.bindradiobutton)
        self.radioButton_3.clicked.connect(self.his_radiobuttion)




    def showimage(self):
        # data_nii = nib.load(Path(self.nii_path))
        # data1 = data_nii.get_fdata()
        # data1 = data1.swapaxes(1,2)
        slice_idx = self.horizontalSlider.value()
        alpha = self.horizontalSlider_4.value()/10
        data1 = self.img
        self.shape = data1.shape[0]
        self.horizontalSlider.setRange(1, data1.shape[0])

        if not self.mask_path == '':
            data2 = np.ma.masked_where(self.seg == 0, self.seg)

        fig = self.F.figure
        fig.clear()
        ax1 = fig.add_subplot(131)  # 将画布划成1*1的大小并将图像放在1号位置，给画布加上一个坐标轴
        ax1.imshow(data1[slice_idx - 1, :, :].T, cmap='gray', origin='lower')
        ax1.xaxis.set_visible(False)
        ax1.yaxis.set_visible(False)
        # 将mask的矩阵转换，未勾画区为透明的，勾画区为红色
        if self.check == 1:
            ax1.imshow(data2[slice_idx - 1, :, :].T, cmap='tab10', alpha=alpha, origin='lower')

        if self.hist == 1:

            self.horizontalSlider_2.setRange(1,  int(np.max(data1)))
            self.horizontalSlider_3.setRange(1, int(np.max(data1)))
            b_thres = self.horizontalSlider_2.value()
            b_upthres = self.horizontalSlider_3.value()
            b_img = (self.img[slice_idx - 1]>b_thres)&(self.img[slice_idx - 1]<b_upthres)
            ax2 = fig.add_subplot(132)
            ax2.imshow(b_img.T, cmap='gray', origin='lower')
            ax2.imshow(data2[slice_idx - 1, :, :].T, cmap='tab10', alpha=alpha, origin='lower')
            ax2.xaxis.set_visible(False)
            ax2.yaxis.set_visible(False)

            img_hist = cluster.quan_data(self.img, self.seg, quan_bone_num=None)
            ax3 = fig.add_subplot(133)
            ax3.hist(img_hist, bins = 100,density =True, color = 'lightblue',label = 'Intensity histogram')
            ax3.axvline(b_thres, color = 'red' , linestyle= '--')
            ax3.axvline(b_upthres, color='blue', linestyle='--')
            ax3.set_xlim(0,1500)
            ax3.yaxis.set_visible(False)

        fig.tight_layout()
        fig.canvas.draw()

    def bindradiobutton(self):
        if self.radioButton.isChecked():
            self.check = 0
        else:
            self.check = 1
        #slice_idx = self.horizontalSlider.value()
        self.showimage()

    def bindSlider(self):
        self.textBrowser_5.setText(str(self.horizontalSlider.value()))
        self.showimage()

    def bindSlider_2(self):
        self.textBrowser_6.setText(str(self.horizontalSlider_2.value()))
        self.showimage()

    def bindSlider_3(self):
        self.textBrowser_7.setText(str(self.horizontalSlider_3.value()))
        self.showimage()

    def bindSlider_4(self):
        self.textBrowser_10.setText(str(self.horizontalSlider_4.value()/10))
        self.showimage()

    def bindButton(self):
        file_name = QFileDialog.getOpenFileName(None, "Open File", "./", "nii(*.nii.gz;*.nii)")
        self.nii_path = file_name[0]
        self.img = nib.load(Path(self.nii_path))
        self.img = self.img.get_fdata()
        self.img = self.img.swapaxes(0,1) #change from 488,20,488 to 20,488,488.
        self.textBrowser.append('Successfully load the MRI image.')
        #slice_idx = self.horizontalSlider.value()
        self.showimage()

    def bindButton2(self):
        file_name = QFileDialog.getOpenFileName(None, "Open File", "./", "nii(*.nii.gz;*.nii)")
        self.mask_path = file_name[0]
        self.seg = nib.load(Path(self.mask_path))
        self.seg = self.seg.get_fdata()
        self.seg = self.seg.swapaxes(0,1) #change from 488,20,488 to 20,488,488.
        self.textBrowser.append('Successfully load the segmentation image.')


    def bindButton3(self):
        self.textBrowser.append('The datashape is : ' + str(self.img.shape))

    # 此函数用于适用nnUnet模型来处理照片并且保存。
    def bindButton4(self):
        img = nib.load(self.nii_path)

        data = img.get_fdata().astype(np.float32)
        data = data.swapaxes(0, 1)
        data = (data - data.mean()) / (data.std())

        sess = InferenceSession('./torch.onnx')
        output_layers_name = sess.get_outputs()[0].name
        input_layers_name = sess.get_inputs()[0].name
        self.textBrowser.append("loading the model to segment.....")
        labels = []
        for i in range(data.shape[0]):
            label = sess.run([output_layers_name], {input_layers_name:data[i,:,:].reshape(1,1,448,448)})[0]
            # (1,16,448,448)
            tmp = np.zeros((448,448),dtype=np.float32)
            for x in range(448):
                for y in range(448):
                    tmp[x,y] = np.argmax(label[:,:,x,y])

            labels.append(tmp)
        labels = np.array(labels)  #(20,448,448)
        self.textBrowser.append("Successfully predict the label.....")

        save_path = QFileDialog.getSaveFileName(None, "Save File", "./", "nii gz Files (*.nii.gz)")
        affine = img.affine
        new_image = nib.Nifti1Image(labels.astype(np.int8).swapaxes(0, 1), affine)
        nib.save(new_image, save_path[0])

        self.textBrowser.append("Successfully save the segmentation at: " + save_path[0])

    def his_radiobuttion(self):
        if self.radioButton_3.isChecked():
            self.hist = 1
            self.GMM_fit()
            print(int(self.clm.bme_threshold))
            self.horizontalSlider_2.setValue(int(self.clm.bme_threshold))
            #self.textBrowser_6.setText(str(int(self.clm.bme_threshold)))
            self.horizontalSlider_3.setValue(int(self.clm.bme_upper_threshold))
            #self.textBrowser_7.setText(str(int(self.clm.bme_upper_threshold)))
            self.showimage()


    def GMM_fit(self):
        hist_bone_num = None
        quan_bone_num = None
        img_his = cluster.quan_data(self.img, self.seg, hist_bone_num)
        quan_his = cluster.quan_data(self.img, self.seg, quan_bone_num)
        self.clm = cluster.cluster_method(img_his, quan_his, 2)
        self.clm.GMM_fit_2th()


    def quantify_BME(self):

        Quan_info = []
        quan_bone_num = None
        quan_his = (cluster.quan_data(self.img, self.seg, quan_bone_num) - self.clm.normal_mean) / (self.clm.normal_std)
        b_thres = (self.horizontalSlider_2.value() - self.clm.normal_mean) / (self.clm.normal_std)
        b_upthres = (self.horizontalSlider_3.value() - self.clm.normal_mean) / (self.clm.normal_std)
        #overall
        bme_p, mean, std = cluster.bme_information(quan_his, b_thres, b_upthres)
        Quan_info.append([bme_p * 100, mean, std])

        # Carpal bone proportion
        quan_bone_num = [3, 4, 5, 6, 7, 8, 9, 10]
        quan_his = (cluster.quan_data(self.img, self.seg, quan_bone_num) - self.clm.normal_mean) / (self.clm.normal_std)
        bme_p, mean, std = cluster.bme_information(quan_his, b_thres, b_upthres)
        Quan_info.append([bme_p * 100, mean, std])

        # individual bone proportion.
        for quan_bone_num in [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15]]:
            quan_his = (cluster.quan_data(self.img, self.seg, quan_bone_num) - self.clm.normal_mean) / (self.clm.normal_std)
            bme_p, mean, std = cluster.bme_information(quan_his, b_thres, b_upthres)
            Quan_info.append([bme_p * 100, mean, std])


        for i in range (len(Quan_info)):
            self.tableWidget.setItem(i, 0, QTableWidgetItem(str(round(Quan_info[i][0],2))))
            self.tableWidget.setItem(i, 1, QTableWidgetItem(str(round(Quan_info[i][1],2))))
            self.tableWidget.setItem(i, 2, QTableWidgetItem(str(round(Quan_info[i][2],2))))

        self.textBrowser.append('Successfully quantify the BME.')


    def quantify_the_selected_bones(self):
        quan_bone_num = []
        for i in range(1,16):
            #print(getattr(self, 'checkBox_'+str(i)))
            if getattr(self, 'checkBox_'+str(i)).isChecked():
                quan_bone_num.append(i)

        quan_his = (cluster.quan_data(self.img, self.seg, quan_bone_num) - self.clm.normal_mean) / (self.clm.normal_std)
        b_thres = (self.horizontalSlider_2.value() - self.clm.normal_mean) / (self.clm.normal_std)
        b_upthres = (self.horizontalSlider_3.value() - self.clm.normal_mean) / (self.clm.normal_std)
        bme_p, mean, std = cluster.bme_information(quan_his, b_thres, b_upthres)

        self.tableWidget_2.setItem(0, 0, QTableWidgetItem(str(round(bme_p * 100,2))))
        self.tableWidget_2.setItem(1, 0, QTableWidgetItem(str(round(mean, 2))))
        self.tableWidget_2.setItem(2, 0, QTableWidgetItem(str(round(std, 2))))

        self.textBrowser.append('Successfully quantify the selected bones.')

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main = MainDialogImgBW()
    main.show()
    sys.exit(app.exec_())