Introduction:
	This project aims to preform 100 class sports image classification using transfer learning from pretrained Resnet, EfficientNet ... This project includes: image data processing, training deep learning models for classification, GradCAM for visualization, and a GUI for your own data.



Dataset: 
	https://www.kaggle.com/datasets/gpiosenka/sports-classification/



Build: 
	M1 Macbook Pro
	Miniforge 3 (Python 3.9)
	PyTorch version: 2.2.1

* Alternative Build:
	Windows (NIVIDA GPU)
	Anaconda 3
	PyTorch



Generate ".py" file from ".ui" file:
	1) open Terminal. Navigate to directory
	2) Type "pyuic5 -x qt_main.ui -o qt_main.py"



Core Project Structure:
	GUI.py (Run to generate a GUI)
	main.py (Run to train model)
	model.py
	qt_main.py
	training.py
	visualization.py
	data (place the data folder here after downloading from gaggle)
	../pytorch_model_weights (download and place model .pth here)
	