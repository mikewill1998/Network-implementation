import sys
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, 
                           QHBoxLayout, QWidget, QGroupBox, QFileDialog, QLabel, 
                           QTextEdit, QComboBox, QMessageBox, QDialog)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QStringListModel
import numpy as np
import glob

import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as ds
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torchinfo
from torchinfo import summary
from PIL import Image


# GAN Model Structure
class Generator(nn.Module):
    def __init__(self, ngpu, nc=1, nz=100, ngf=64):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf,nc, kernel_size=1, stride=1, padding=2, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        if self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


class Discriminator(nn.Module):
    def __init__(self, ngpu, nc=1, ndf=64):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, 1, 4, 2, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output.view(-1, 1).squeeze(1)

class ImagePopup(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Show Image")
        self.resize(400, 600)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)

        pixmap = QPixmap("/Users/liuchengwei/cvdl_demo/HW2/GAN_related/GAN_Figure_1.png")
        # self.image_label.setPixmap(pixmap.scaled(400, 400, Qt.KeepAspectRatio))
        self.image_label.setPixmap(pixmap)

    def showEvent(self, event):
        self.adjustSize()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Homework2")
        self.setGeometry(100, 100, 400, 600)
        self.image_paths = []
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
        # Create blocks
        self.create_blocks(main_layout)

    def create_blocks(self, parent_layout):
        group_box = QGroupBox("")
        layout = QVBoxLayout()
        
        buttons = ["1. Show Training Images", "2. Show Model Structure", "3. Show Training Loss",
                    "4. Inference"]

        for text in buttons:
            btn = QPushButton(text)
            if text == "1. Show Training Images":
                btn.clicked.connect(self.show_image)
            elif text == "2. Show Model Structure":
                btn.clicked.connect(self.show_model_structure)        
            elif text == "3. Show Training Loss":
                btn.clicked.connect(self.show_training_loss)
            elif text == "4. Inference":
                btn.clicked.connect(self.inference)
            layout.addWidget(btn)
        
        
        group_box.setLayout(layout)
        parent_layout.addWidget(group_box)

        
    def show_image(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select a folder")
        
        image_files = os.listdir(folder_path)
        
        fig1, axes1 = plt.subplots(8, 8, figsize=(8, 8))
        for i, ax in enumerate(axes1.flat):
            if i < len(image_files):
                image_path = os.path.join(folder_path, image_files[i])
                img_o = mpimg.imread(image_path)
                ax.imshow(img_o)
                ax.set_title("")
                ax.axis("off")
        plt.tight_layout()
        plt.show()

        tfm = transforms.Compose([
            transforms.RandomRotation(60)
        ])
        fig2, axes2 = plt.subplots(8, 8, figsize=(8, 8))
        for j, bx in enumerate(axes2.flat):
            if j < len(image_files):
                image_path = os.path.join(folder_path, image_files[j])
                img = Image.open(image_path)
                img_t = tfm(img)
                bx.imshow(img_t)
                bx.set_title("")
                bx.axis("off")
        
        plt.tight_layout()
        plt.show()


        
           
    def show_model_structure(self):
        netG = Generator(1)
        netD = Discriminator(1)
        print(netG)
        print(netD)
    
    def show_training_loss(self):
        training_loss_img = ImagePopup(self)
        training_loss_img.exec_()

    def inference(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select a folder")
        
        image_files = os.listdir(folder_path)
        
        fig1, axes1 = plt.subplots(8, 8, figsize=(8, 8))
        for i, ax in enumerate(axes1.flat):
            if i < len(image_files):
                image_path = os.path.join(folder_path, image_files[i])
                img_o = mpimg.imread(image_path)
                ax.imshow(img_o)
                ax.set_title("")
                ax.axis("off")
        plt.tight_layout()
        plt.show()

        netD = Discriminator(1).to("cpu")
        netG = Generator(1).to("cpu")
        netD.load_state_dict(torch.load("/Users/liuchengwei/cvdl_demo/HW2/GAN_related/netD_10.pth", weights_only=True, map_location="cpu"))
        netG.load_state_dict(torch.load("/Users/liuchengwei/cvdl_demo/HW2/GAN_related/netG_10.pth", weights_only=True, map_location="cpu"))

        netG.eval()
        with torch.no_grad():
            noise = torch.randn(64, 100, 1, 1, device="cpu")
            fake_images = netG(noise)

            fake_images = np.transpose(fake_images, (0, 2, 3, 1))
            fake_images = (fake_images + 1) / 2.0
            
            fig, axes = plt.subplots(8, 8, figsize=(8, 8))
            for i in range(8):
                for j in range(8):
                    axes[i, j].imshow(fake_images[i*8+j])
                    axes[i, j].axis("off")
            
            plt.tight_layout()
            plt.show()
   

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
