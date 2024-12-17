from ultralytics import YOLO

#Create a new YOLO model from scratch
#model = YOLO("yolov8n.yaml")

#Load a pretrained YOLO model (recommended for training)
#model = YOLO("yolov8n.pt")

model = YOLO('runs/detect/train2/weights/best.pt')

#Train the model using the 'coco8.yaml' dataset for 3 epochs
#results = model.train(data="data.yaml", imgsz=640, batch=1, epochs=50, patience=25)

# model = YOLO("yolov8n.pt")
# model = YOLO("model/best.pt")

result = model.predict('1.jpg', iou=0.3)
result[0].show() 

result = model.predict('2.jpg', iou=0.3)
result[0].show() 

result = model.predict('3.jpg', iou=0.3)
result[0].show() 

result = model.predict('4.jpg', iou=0.3)
result[0].show() 

result = model.predict('5.jpg', iou=0.3)
result[0].show() 

result = model.predict('6.jpg', iou=0.3)
result[0].show() 

result = model.predict('7.jpg', iou=0.3)
result[0].show() 

result = model.predict('8.jpg', iou=0.3)
result[0].show() 

result = model.predict('9.jpg', iou=0.3)
result[0].show() 

#result[0].save('result/resultMine.jpg')