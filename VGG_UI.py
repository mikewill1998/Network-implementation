import os
import sys
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, 
                           QHBoxLayout, QWidget, QGroupBox, QFileDialog, QLabel, 
                           QTextEdit, QComboBox, QMessageBox, QDialog, QSizePolicy)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QStringListModel
import numpy as np
import glob
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import torchinfo
from torchinfo import summary


class ImagePopup(QDialog):
    def __init__(self, image_paths):
        super().__init__()
        self.setWindowTitle("Image Viewer")

        layout = QVBoxLayout()

        # Create two horizontal layouts for each image and label
        hbox1 = QHBoxLayout()
        label1 = QLabel()
        pixmap1 = QPixmap(image_paths[0])
        label1.setPixmap(pixmap1.scaled(400, 300, aspectRatioMode=Qt.KeepAspectRatio))
        # label1.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        hbox1.addWidget(label1)

        hbox2 = QHBoxLayout()
        label2 = QLabel()
        pixmap2 = QPixmap(image_paths[1])
        label2.setPixmap(pixmap2.scaled(400, 300, aspectRatioMode=Qt.KeepAspectRatio))
        # label2.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        hbox2.addWidget(label2)

        layout.addLayout(hbox1)
        layout.addLayout(hbox2)

        self.setLayout(layout)

    def showEvent(self, event):
        self.adjustSize()


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        # Create a layout for the panel
        layout = QHBoxLayout()
        sub_layout = QVBoxLayout()
        self.image_paths = []
        self.file_name = ""
        self.image_label = QLabel()
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.varying_label = QLabel()
        self.fixed_label = QLabel("Predicted: ")
        # Create a button to trigger the file dialog
        buttons = ["Load Image", "1. Show Augmented Images", "2. Show Model Structure", 
                   "3. Show Accuracy and Loss", "4. Inference"]
        
        for text in buttons:
            btn = QPushButton(text)
            if text == "Load Image":
                btn.clicked.connect(self.load_image)
            elif text == "1. Show Augmented Images":
                btn.clicked.connect(self.show_augmented_images)        
            elif text == "2. Show Model Structure":
                btn.clicked.connect(self.show_model_structure)
            elif text == "3. Show Accuracy and Loss":
                btn.clicked.connect(self.show_accuracy_and_loss)
            elif text == "4. Inference":
                btn.clicked.connect(self.inference)
            sub_layout.addWidget(btn)
        
        sub_layout.addWidget(self.fixed_label)
        sub_layout.addWidget(self.varying_label)
        # Create a label to display the image
        
        layout.addLayout(sub_layout)
        layout.addWidget(self.image_label)
        self.setLayout(layout)

    def load_image(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg)")
        if filename:
            self.file_name = filename
            pixmap = QPixmap(self.file_name)
            self.image_label.setPixmap(pixmap.scaled(128, 128, aspectRatioMode=Qt.KeepAspectRatio))
            return self.file_name
        else:
            print("No file loaded")
    def show_augmented_images(self):
        train_tfm = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30)
        ])
        
        folder_path = QFileDialog.getExistingDirectory(self, "Select a folder")
        
        image_files = os.listdir(folder_path)
        
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        for i, ax in enumerate(axes.flat):
            if i < len(image_files):
                image_path = os.path.join(folder_path, image_files[i])
                img = Image.open(image_path)
                img = train_tfm(img)
                ax.imshow(img)
                cls_name = image_files[i].split(".")[0]
                ax.set_title(cls_name)
                ax.axis(True)
        
        plt.tight_layout()
        plt.show()
            

           

    def show_model_structure(self):
        model_1 = models.vgg19_bn()
        summary(model_1)
    
    def show_accuracy_and_loss(self):
        image_paths = ["/Users/liuchengwei/cvdl_demo/HW2/CIFAR10_related/Accuracy_CIFAR_Figure_1.png", "/Users/liuchengwei/cvdl_demo/HW2/CIFAR10_related/Loss_CIFAR_Figure_1.png"]
        popup = ImagePopup(image_paths)
        popup.exec_()

    def inference(self):
        model_0 = models.vgg19_bn(num_classes=10)
        model_0.load_state_dict(torch.load("/Users/liuchengwei/cvdl_demo/HW2/CIFAR10_related/CIFAR10_model.pth", weights_only=True, map_location="cpu"))
        model_0.eval()
        tfm = transforms.Compose([
            transforms.ToTensor()
        ])
        img = Image.open(self.file_name)
        img_tensor = tfm(img).unsqueeze(0)
        with torch.no_grad():
            output = model_0(img_tensor)
            probabilities = torch.softmax(output, dim=1).squeeze().numpy()

        plt.bar(range(10), probabilities)
        plt.xticks(range(10), ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])
        plt.xlabel('Class')
        plt.ylabel('Probability')
        plt.title('Class Probabilities')
        plt.show()
        
        predicted_class_index = np.argmax(probabilities)
        predicted_class = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'][predicted_class_index]

        self.varying_label.setText(predicted_class)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())