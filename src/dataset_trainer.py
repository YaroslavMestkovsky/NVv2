from ultralytics import YOLO


model = YOLO("yolov8n.pt")
data_path = 'WarspTestV1.v1i.yolov8/data.yaml'


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
