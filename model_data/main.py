from Detector import*
import os
def main():
    videopath = "test_videos/vids1.mp4"
    configpath = os.path.join("model_data","sdd_mobilenet_v3.pbtxt")
    modelpath = os.path.join("model_data","frozen_inference_graph.pb")
    classpath = os.path.join("model_data","coco.names")

    detector=Detector(videopath,configpath,modelpath,classpath)
    detector.onvideo()
if __name__ == '__main__':
    main()
