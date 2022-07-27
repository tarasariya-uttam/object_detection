from wsgiref.simple_server import make_server
from pyramid.config import Configurator
from pyramid.response import Response
import os

#sherly's libs
import cv2
import numpy as np
from object_detection import ObjectDetection
from deep_sort.deep_sort import Deep

def detect(model, frame):
  '''Given results from yolov5, extract classs ids, scores and boxes as numpy array'''

  results = model(frame)
  result_pandas = results.pandas().xyxy[0]
  print(result_pandas)
  result_list = results.xyxy[0].cpu().detach().tolist()
  class_ids = []
  scores = []
  boxes = []

  for i in range(len(result_list)):
    class_ids.append(result_list[i][5])
    scores.append(result_list[i][4])
    boxes.append(result_list[i][0:4])

  class_ids = np.asarray(class_ids, dtype=np.int32)
  scores = np.asarray(scores, dtype=np.float64)
  boxes = np.asarray(boxes, dtype=np.float64)

  return class_ids, scores, boxes

yolov4 = ObjectDetection("yolov4.weights", "dnn_model/yolov4.cfg")
yolov4.load_class_names("dnn_model/classes.txt")
yolov4.load_detection_model(image_size=832, # 416 - 1280
                        nmsThreshold=0.4,
                        confThreshold=0.3)

def hello_world(request):
    name = os.environ.get('NAME')
    if name == None or len(name) == 0:
        name = "world"
    message = "Hello, The Uttamkumar " + name + "!\n"
    return Response(message)

if __name__ == '__main__':
    port = int(os.environ.get("PORT"))
    with Configurator() as config:
        config.add_route('hello', '/')
        config.add_view(hello_world, route_name='hello')
        app = config.make_wsgi_app()
    server = make_server('0.0.0.0', port, app)
    server.serve_forever()
