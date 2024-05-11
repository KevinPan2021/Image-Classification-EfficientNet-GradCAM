Introduction:
	This project aims to preform 100 class sports image classification using transfer learning from pretrained EfficientNet. This project includes: image data processing, training deep learning models for classification, GradCAM for visualization, and a GUI for your own data.



Dataset: 
	https://www.kaggle.com/datasets/gpiosenka/sports-classification/



Build: 
	System:
		CPU: Intel i9-13900H (14 cores)
		GPU: NIVIDIA RTX 4060 (VRAM 8 GB)
		RAM: 32 GB

	Configuration:
		CUDA 12.1
		Anaconda 3
		Python = 3.10.9
		Spyder = 5.4.1
		
	Core Python Package:
		pytorch = 2.1.2
		numpy = 1.23.5
		OpenCV = 4.9.0.80
		matplotlib = 3.7.0
		pandas = 1.5.3
		tqdm = 4.64.1



Generate ".py" file from ".ui" file:
	1) open Terminal. Navigate to directory
	2) Type "pyuic5 -x qt_main.ui -o qt_main.py"



Core Project Structure:
	GUI.py (Run to generate a GUI)
	main.py (Run to train model)
	efficientnet.py
	qt_main.py
	training.py
	visualization.py
	data (place the data folder here after downloading from gaggle)
	../pytorch_model_weights (download and place model .pth here)
	