application_name = 'Sport Classification'

# pyqt packages
from PyQt5 import uic
from PyQt5.QtGui import QPainter, QPixmap, QImage
from PyQt5.QtWidgets import QMainWindow, QApplication, QMessageBox, QFileDialog

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cv2
import sys
import numpy as np
import pickle
import torch
from PIL import Image

from efficientnet import EfficientNet_b0
from main import SaveFeatures, BidirectionalMap, compute_device, get_transform


def dark_JET_cmap():
    jet = plt.colormaps['jet']
    colors = jet(np.linspace(0.15, 0.9, 256))
    colors = np.vstack((np.array([0, 0, 0, 1]), colors))
    return LinearSegmentedColormap.from_list('modified_jet', colors)



def show_message(parent, title, message, icon=QMessageBox.Warning):
        msg_box = QMessageBox(icon=icon, text=message)
        msg_box.setWindowIcon(parent.windowIcon())
        msg_box.setWindowTitle(title)
        msg_box.setStyleSheet(parent.styleSheet() + 'color:white} QPushButton{min-width: 80px; min-height: 20px; color:white; \
                              background-color: rgb(91, 99, 120); border: 2px solid black; border-radius: 6px;}')
        msg_box.exec()
        
        
        
class QT_Action(QMainWindow):
    def __init__(self):
        # system variable
        super(QT_Action, self).__init__()
        uic.loadUi('qt_main.ui', self)
        self.setWindowTitle(application_name) # set the title
        
        # runtime variable
        self.image = None
        self.model = None
        self.weight = None
        self.activated_features = None
        self.transform = None
        with open('class_ind_pair.pkl', 'rb') as f:
            self.class_ind_pair = pickle.load(f)
        
        # load the model
        self.load_model_action()
        
        
    # linking all button/textbox with actions    
    def link_commands(self,):
        self.toolButton_import.clicked.connect(self.import_action)
        self.comboBox_model.activated.connect(self.load_model_action)
        self.toolButton_process.clicked.connect(self.process_action)
        
    
    # choosing between models
    def load_model_action(self,):
        self.model_name = self.comboBox_model.currentText()
        
        if self.model_name == 'EfficientNet_b0':
            # load the model architechture
            self.model = EfficientNet_b0(len(self.class_ind_pair))
            
            # loading the training model weights
            self.model.load_state_dict(torch.load(f'{self.model_name}.pth', weights_only=True))
            
            # extract the final conv and fully connected layers (for GradCAM)
            final_conv = self.model.model.features[-1]
            fc_params = list(self.model.model.classifier.parameters())
            
            # input image transform
            self.transform = get_transform()
            
        # weights and conv layer features for GradCAM
        self.weight = np.squeeze(fc_params[0].cpu().data.numpy())
        self.activated_features = SaveFeatures(final_conv)
        
        # move model to GPU
        self.model = self.model.to(compute_device())
        
        self.model.eval() # Set model to evaluation mode
        
        
        
    
    # clicking the import button action
    def import_action(self,):
        # show an "Open" dialog box and return the path to the selected file
        filename, _ = QFileDialog.getOpenFileName(None, "Select file", options=QFileDialog.Options())
        self.lineEdit_import.setText(filename)
        
        # didn't select any files
        if filename is None or filename == '': 
            return
    
        # selected .oct or .octa files
        if filename.endswith('.jpg'):
            self.image = Image.open(filename) 
            self.lineEdit_import.setText(filename)
            #X = [transform(img)]
            self.update_display()
        
        # selected the wrong file format
        else:
            show_message(self, title='Load Error', message='Available file format: .jpg')
            self.import_action()
        
        
    def update_display(self):
        if not self.image is None:
            image = self.image.convert('RGBA')  # Ensure the image is in RGBA format
            data = image.tobytes('raw', 'RGBA')  # Get the raw RGBA data
            q_image = QImage(data, image.size[0], image.size[1], QImage.Format_RGBA8888)  # Create QImage
            qpixmap = QPixmap.fromImage(q_image)  # Convert QImage to QPixmap
            self.label_image.setPixmap(qpixmap)
            
            
    def process_action(self):
        if self.image is None:
            show_message(self, title='Process Error', message='Please load an image first')
            return
        
        # apply the transform
        data = self.transform(self.image)
        
        # move data to GPU
        data = data.to(compute_device())
        
        # add the batch dimension
        data = data.unsqueeze(0)
        
        # model inference
        with torch.no_grad():  # Disable gradient calculation
            output = self.model(data).cpu()
        _, predicted = torch.max(output.data, 1)
        predicted = predicted.item() # convert to int
        predicted_label = self.class_ind_pair.get_value(predicted)
 
        # generate GradCAM Heatmap
        conv_fs = self.activated_features.features
        batch, chs, h, w = conv_fs.shape
        cam = self.weight[predicted].dot(conv_fs[0,:, :, ].reshape((chs, h * w)))
        cam = cam.reshape(h, w)
        heatmap = (cam - np.min(cam)) / np.max(cam)
        heatmap = cv2.resize(heatmap, (data.shape[2], data.shape[3]), interpolation=cv2.INTER_LINEAR)
        
        # add result to UI
        self.lineEdit_prediction.setText(predicted_label)
        
        # overlay heatmap on gray image
        img = data.squeeze().cpu().numpy().transpose(1,2,0) # convert to width,height,channel
        gray_img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140]) # convert to gray image
        # min max normalization for RGB image
        minVal, maxVal = np.min(gray_img), np.max(gray_img)
        gray_img = ((gray_img-minVal) / (maxVal - minVal) * 255).astype(np.uint8)
        
        # Create QImage from the grayscale image
        q_image = QImage(gray_img.data, gray_img.shape[1], gray_img.shape[0], gray_img.shape[1], QImage.Format_Grayscale8)

        qpixmap = QPixmap.fromImage(q_image)  # Convert QImage to QPixmap
        
        # Overlay the heatmap
        qp = QPainter(qpixmap)
        qp.setOpacity(0.5)
        
        # Assuming `dark_JET_cmap()` returns a colormap with appropriate dimensions
        heatmap_normalized = heatmap / np.max(heatmap)
        heatmap_colored = dark_JET_cmap()(heatmap_normalized) * 255
        
        # Create QImage from the heatmap
        heatmap_image = QImage(heatmap_colored.astype(np.uint8), heatmap_colored.shape[1], heatmap_colored.shape[0], QImage.Format_RGBA8888)
        
        # Draw the heatmap onto the QPixmap
        qp.drawImage(0, 0, heatmap_image)
        qp.end()
        
        # Display the result in label_heatmap
        self.label_heatmap.setPixmap(qpixmap)
    
    
def main():
    app = QApplication(sys.argv)
    action = QT_Action()
    action.link_commands()
    action.show()
    sys.exit(app.exec_())
    
    
if __name__ == '__main__':
    main()