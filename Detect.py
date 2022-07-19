import cv2
import numpy as np
import os
import tensorflow as tf
import time
from tensorflow.python.keras.utils.data_utils import get_file
import bidi.algorithm, arabic_reshaper


np.random.seed(123)

#engine = pyttsx3.init()


class Detectcls:

 #todo define : for reading classes file
    def readClasses(self,classesFilePath):
        #with open(classesFilePath, 'r' )as f:
        with open(classesFilePath,encoding ='utf-8')as f:


            self.classesList = f.read().splitlines()

            #color list
            self.colorlist = np.random.uniform(low=0,high=255,size=(len(self.classesList), 3))

            print(len(self.classesList), len(self.colorlist))



    #todo define : for downloadModel in pretrained_models file befor get this file
    def downloadModel(self,modelURL):

        fileName = os.path.basename(modelURL)
        self.modelName = fileName[:fileName.index('.')]

        self.cacheDir = "./pretrained_models"
        os.makedirs(self.cacheDir,exist_ok=True)

        get_file(fname=fileName,origin=modelURL, cache_dir=self.cacheDir, cache_subdir="checkpoint", extract=True)
        print(fileName)
        print(self.modelName)

    #todo define : for loading the model and save it in checkpoint file for using after
    def loadModel(self):
        print('downloading model '+self.modelName)
        tf.keras.backend.clear_session()
        self.model = tf.saved_model.load(os.path.join(self.cacheDir, "checkpoint",self.modelName, "saved_model"))
        print("Model "+self.modelName + "loaded successfuly . . .!")



    #todo define : method that for detect and frame the object object in a image or a video

    def createBoundingBox(self, image,threshold=0.5,):
        inputTensor = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        inputTensor = tf.convert_to_tensor(inputTensor, dtype=tf.uint8)
        inputTensor = inputTensor[tf.newaxis,...]

        detections = self.model(inputTensor)



        bboxes = detections['detection_boxes'][0].numpy()
        classIndexes = detections['detection_classes'][0].numpy().astype(np.int32)
        classScore = detections['detection_scores'][0].numpy()



        imH,imW,imC = image.shape

        #todo :pour encadré seul les objet | for framed objects only
        bboxIdx = tf.image.non_max_suppression(bboxes,classScore ,max_output_size=50,iou_threshold=threshold,score_threshold=threshold)
        print(bboxIdx)

        #
        # engine.say(" detect " + bboxIdx)
        # # os.system("say " + classLabelText)
        # engine.runAndWait()

        if len(bboxIdx) != 0:
            for i in bboxIdx:
                bbox = tuple(bboxes[i].tolist())
                classConfidence = round(100*classScore[i])
                classIndex = classIndexes[i]

                #classe text centent nom dobjet .upper pour le text soit maj
                #classLabelText = self.classesList[classIndex].upper()



                classLabelText = self.classesList[classIndex]#.upper()
                #print(classLabelText)
                classColor = self.colorlist[classIndex]

                # # To get arabic outputs in terminal or kivy or even pyGame etc.
                # reshaper = arabic_reshaper.reshape(classLabelText)
                # bidi_text = bidi.algorithm.get_display(reshaper)

                #displayText = '{} : {}%'.format(bidi_text,classConfidence)

                # #test du voix
                # tts = gTTS(text=classLabelText, lang="en")
                # filename = "voice.mp3"




                displayText = '{} : {}%'.format(classLabelText,classConfidence)




                ymin,xmin,ymax,xmax = bbox

                xmin,xmax,ymin,ymax = (xmin * imW, xmax * imW, ymin * imH , ymax * imH)
                xmin, xmax, ymin, ymax = int(xmin), int(xmax), int(ymin), int(ymax)
                cv2.rectangle(image, (xmin,ymin), (xmax,ymax), color=classColor,thickness=1)

                #pour afficher le classe label et le confidenc %
                cv2.putText(image, displayText,(xmin,ymin -10),cv2.FONT_HERSHEY_PLAIN,1,classColor,2)

                # pour modifier les bordure et les coins du cadre
                lineWidth=min(int((xmax-xmin)*0.2),int((ymax-ymin)*0.2))

                #############

                cv2.line(image,(xmin,ymin), (xmin + lineWidth,ymin),classColor,thickness=5)
                cv2.line(image,(xmin,ymin), (xmin ,ymin + lineWidth),classColor,thickness=5)

                cv2.line(image,(xmax,ymin), (xmax - lineWidth,ymin),classColor,thickness=5)
                cv2.line(image,(xmax,ymin), (xmax ,ymin + lineWidth),classColor,thickness=5)

                #####################

                cv2.line(image, (xmin, ymax), (xmin + lineWidth, ymax), classColor, thickness=5)
                cv2.line(image, (xmin, ymax), (xmin, ymax - lineWidth), classColor, thickness=5)

                cv2.line(image, (xmax, ymax), (xmax - lineWidth, ymax), classColor, thickness=5)
                cv2.line(image, (xmax, ymax), (xmax, ymax - lineWidth), classColor, thickness=5)
        # #test
        # tts.save(filename)
        # playsound.playsound(filename)
        return image

        #todo define : for predict image and detect all the object available from the model used
    def predictImage(self,imagePath,threshold=0.5):
        image = cv2.imread(imagePath)
        bboxImage = self.createBoundingBox(image,threshold)

        cv2.imwrite(self.modelName +".jpg",bboxImage)
        cv2.imshow("result",bboxImage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    #todo define : for predict real time or video and detect all the object available from the model used
    def predictVideo(self,videoPath , threshold = 0.5):
        cap = cv2.VideoCapture(videoPath)
        if(cap.isOpened()== False):
            print("Error video ")
            return
        (success,image) = cap.read()

        starttime =0

        while success:
            currentTime = time.time()
            fps = 1/(currentTime - starttime)
            starttime = currentTime
            bboxImage = self.createBoundingBox(image,threshold)
            cv2.putText(bboxImage , "FPS: " +str(int(fps)),(20,70),cv2.FONT_HERSHEY_PLAIN , 2 ,(0,255,0),2)
            cv2.imshow("RESULT ",bboxImage)
            key=cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            (success,image)=cap.read()
        cv2.destroyAllWindows()