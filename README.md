# Facer

Its a face recognition system to detect faces using trained model.

## Prerequisites

1. Poetry
2. Python >= 3.10
3. [Dataset](https://www.kaggle.com/datasets/adg1822/7-celebrity-images)

## QuickStart

1. Create virtual env

```bash
poetry shell
```

2. Install dependencies

```bash
poetry install
```

3. Run `facer`

```bash
python main.py --encode --model [MODEL_NAME='hog|cnn'] --image [IMAGE_PATH]
```

_eg:_

```bash
python main.py --encode --model hog --image 'dataset/val/elon_musk/161856.jpg'
```

## Refrences

- [face-recognition-with-python](https://realpython.com/face-recognition-with-python/#demo)
- [real-time-object-detection-with-yolo](https://dipankarmedh1.medium.com/real-time-object-detection-with-yolo-and-webcam-enhancing-your-computer-vision-skills-861b97c78993)
