from ultralytics import YOLO


model = YOLO("yolov5s.pt")
data_path = 'fairy_fbS.yaml'

def train():
    model.train(
        data=data_path,
        epochs=350,
        batch=16,
        name="WarspV2",
        workers=3,

    )

if __name__ == '__main__':
    train()
