# YoloV3 custom training video
En este proyecto se presenta el entrenamiento con tu propio dataset para YOLOv3

## Artículo original:
https://pjreddie.com/media/files/papers/YOLOv3.pdf

# Implementación:

### Crear entorno en conda

  	$ conda create -n YoloCustom anaconda python=3.10
	
	$ conda activate YoloCustom
  
  	$ pip install opencv-python numpy matplotlib tensorboard terminaltables pillow tqdm
  
  	$ conda install torch==2.1.1 torchvision==0.16.1 cudatoolkit=10.0 -c pytorch

### Preparar Dataset
  Ejecutar labelImg.py (En el mismo entorno conda)
  Ingresamos a la ruuta donde tenemos la caprta en local del entrenador
  E ingesamos esa ruta en la siguiente linea 
  
        $conda install pyqt=5

        $conda install -c anaconda lxml
	
	$cd D:\Proyectos PCA\YoloCustom_Syringe_Only\labelImg
 
  	$ cd labelImg
        
##Crear el script resources.py
        
	$pyrcc5 -o resources.py resources.qrc

#Mover el archivo resources.py  y resources.qrc rearchivo a la cartpeta libs en donde se descargó el repositorio
#Probar 
   
  	$ python labelImg.py
  
  Se deben obtener los archivos labels (.txt) de las imagenes de tu dataset
  
  Luego ejecutar jupyter notebook desde anaconda y abrir Yolov3Files.ipyb para generar los archivos train.txt y valid.txt
  
  $ jupyter notebook
  
  Finalmente se obtienen los siguientes archivos
  
  	En data/custom/images las imagenes del dataset
  
  	En data/custom/labels los labels .txt del dataset
  
  	En data/ los archivos train.txt y valid.txt
  
  	En data/ el archivo classes.names con los nombres de las clases del dataset
  
  	En la carpeta config/ el archivo .cfg que corresponda con el número de clases de nuestro dataset
  
  	En la carpeta config/ modificar el archivo custom.data con el número de clases del dataset
  
  Para el archivo .cfg abrir el archivo bash .sh y modificar el parámetro NUM_CLASSES=3
  
  Ejecutarlo usango Git para windows
  
  Si no tienes instalado Git para windows lo puedes descargar en el siguieinte enlace
  
  https://git-scm.com/download/win
  
  
### Entrenamiento en google colab
  	$!pip install torch==2.1.1 torchvision==0.16.1
  
  	$!pip install opencv-python numpy matplotlib tensorboard terminaltables pillow tqdm
  
  	$!git clone https://github.com/SeUcz/YoloCustom_Syringe_Only
  
  	$cd Yolov3Custom
  

  Entrar al directorio /usr/local/lib/libpackages/torchvision/transforms/functional.py
  
  Cambiar esta línea
  
	from PIL import Image, ImageOps, ImageEnhance, PILLOW_VERSION
	
  Por esta
  
    from PIL import Image, ImageOps, ImageEnhance, __version__ as PILLOW_VERSION
    
  Entrenamiento
	
  	$!python train.py --model_def config/yolov3-custom3C.cfg --data_config config/custom.data --epochs 200 --batch_size 4 --pretrained_weights weights/darknet53.conv.74
  
### Prueba local en imágen y video

   Descargar el archivo .pth y copiarlo en el directorio checkpoints local
   
   	python detectC.py --image_folder data/samplesC/ --model_def config/yolov3-custom3C.cfg --weights_path checkpoints/yolov3_ckpt_252.pth --class_path data/custom/classes.names
   
   	python detect_cam.py --model_def config/yolov3-custom3C.cfg --weights_path checkpoints/yolov3_ckpt_252.pth --class_path data/custom/classes.names --conf_thres 0.6
   

### Basado en el trabajo de Erik Linder
   https://github.com/eriklindernoren

### Dataset original
   https://www.tooploox.com/blog/card-detection-using-yolo-on-android

# **Canal de Youtube**
[Click aquì pare ver mi canal de YOUTUBE](https://www.youtube.com/channel/UCr_dJOULDvSXMHA1PSHy2rg)
