from Detect import Detectcls


#there is meny models in "https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md"
# or "https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md"
modelURL="http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz"

classFile = 'coco.names'
imagePath = "data/1.jpg"
#videoPath = "data/video.mp4" # si en test real time avec webcam juste on remplace par 0
videoPath = 0 # si en test real time avec webcam juste on remplace par 0
threshould=0.5

Detect = Detectcls()
Detect.readClasses(classFile)
Detect.downloadModel(modelURL)
Detect.loadModel()
#Detect.predictImage(imagePath,threshould)

Detect.predictVideo(videoPath,threshould)

detector.createBoundingBox(videoPath)
