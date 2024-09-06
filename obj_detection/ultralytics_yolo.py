import os 
import ultralytics
from ultralytics import YOLOv10

wget -p {HOME}/weights -q https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10s.pt

def main():
  conf_thresh: float = 0.55
  IMAGE_NAME: str = 'test_image.png'  
  model = YOLOv10('{HOME}/weights/yolov10s.pt')
  results = model(IMAGE_NAME, conf=conf_thresh)
  print(results[0].boxes.cls)

if __name__ == '__main__':
  main()

