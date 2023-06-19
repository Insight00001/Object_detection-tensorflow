import cv2
import numpy as np
import time
np.random.seed(20)
class Detector:
    def __init__(self,videopath,configpath,modelpath,classpath):
        self.videopath = videopath
        self.configpath = configpath
        self.modelpath = modelpath
        self.classpath = classpath

        self.net = cv2.dnn_DetectionModel(self.modelpath,self.configpath)
        self.net.setInputSize(320,320) # the size of the image on which the model was trained
        self.net.setInputScale(1.0/127.5)
        self.net.setInputMean((127.5,127.5,127.5))
        self.net.setInputSwapRB(True)
    # read class list from coco.names
        self.readclass()
    def readclass(self):
        with open(self.classpath,'r') as f:
            self.classlist = f.read().splitlines()
        self.colorlist = np.random.uniform(low=0,high = 255,size= (len(self.classlist),3))
      

    def onvideo(self):
        cap = cv2.VideoCapture(self.videopath)
        if (cap.isOpened()==False):
            print("error im opening")
            return
        success,image = cap.read()
        while success:
            classlabelids,confidence,bboxes = self.net.detect(image,confThreshold=0.4)
            bboxes=list(bboxes)
            confidence = list(np.array(confidence).reshape(1,-1)[0])
            confidence = list(map(float,confidence))
            bboxidx = cv2.dnn.NMSBoxes(bboxes,confidence,score_threshold=0.5,nms_threshold=0.2)
            
            if len(bboxidx)!=0:
                for i in range(0,len(bboxidx)):
                    bbox = bboxes[np.squeeze(bboxidx[i])]
                    classconfidence = confidence[np.squeeze(bboxidx[i])]
                    classlabelid = np.squeeze(classlabelids[np.squeeze(bboxidx[i])])
                    classlabel = self.classlist[classlabelid]
                    classcolor = [int(c) for c in self.colorlist[classlabelid]]
                    displaytext = "{}:{:.4f}".format(classlabel,classconfidence)
                    x,y,w,h = bbox
                    cv2.rectangle(image,(x,y),(x+w,y+h),color=(255,255,255),thickness=  2)
                    
                    cv2.putText(image,displaytext,(x, y-10),cv2.FONT_HERSHEY_PLAIN,1,classcolor,2)

            cv2.imshow("Result",image)
            key = cv2.waitKey(1) & 0XFF
            if key == ord('q'):
                break
            (success,image) = cap.read()
        cv2.destroyAllWindows()





        