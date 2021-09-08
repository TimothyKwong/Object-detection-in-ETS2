import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms.functional
from mss.windows import MSS as mss
from PIL import Image
import win32gui
import argparse
import time

# CUDA & CUDNN for PyTorch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cudnn.enabled = True

# Initializing FasterRCNN
faster_rcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, progress=True, num_classes=91, pretrained_backbone=True, trainable_backbone_layers=None)
model = faster_rcnn.to(device)
model.eval()

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def parser():
    parser = argparse.ArgumentParser(description='Faster RCNN project')
    parser.add_argument('screen',  nargs='?', const=0, default=0, type=int, 
                        help="screen to detect: "
                        "Steam, Euro Truck Simulator 2")
    return parser.parse_args()

def check_arguments_errors(args):
    return None


def initialization(screen):
    # Get rect of Window
    game = ['Steam', 'Euro Truck Simulator 2']
    hwnd = win32gui.FindWindow(None, game[screen])
    rect = win32gui.GetWindowRect(hwnd)
    region = rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1]
    bounding_box = {'top': region[1], 'left': region[0], 'width': region[2], 'height': region[3]}
    return bounding_box


def get_prediction(img_path, threshold):
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    img = transform(img_path)
    img = img.unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred = model(img)
        scores, labels, boxes = pred[0]['scores'].cpu().numpy(), pred[0]['labels'].cpu().numpy(), pred[0]['boxes'].cpu().numpy()
        pred_score = list(scores)

        try:
            pred_t = [pred_score.index(x) for x in pred_score if x>threshold][-1]
        except IndexError:
            print('Nothing detected.')
            pred_boxes = None
            pred_class = None
            state = False
            return pred_boxes, pred_class, state

        pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(labels)]
        pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(boxes)]

        pred_boxes = pred_boxes[:pred_t+1]
        pred_class = pred_class[:pred_t+1]
        state = True

        return pred_boxes, pred_class, state


def main():
    args = parser()
    font = cv2.FONT_HERSHEY_SIMPLEX

    with mss() as sct:
        while True:
            loop_time = time.time()
            
            bounding_box = initialization(args.screen)
            sct_img = sct.grab(bounding_box)
            sct_img = np.array(Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX"))

            img = cv2.resize(sct_img, (sct_img.shape[1]//2, sct_img.shape[0]//2))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            boxes, pred_class, state = get_prediction(img, 0.7)            
            
            if state == True:
                for i in range(len(boxes)):
                    (x, y) = (boxes[i][0][0], boxes[i][0][1])
                    (w, h) = (boxes[i][1][0], boxes[i][1][1])
                    (x, y) = (int(x), int(y))
                    (w, h) = (int(w), int(h))
                    img = cv2.rectangle(img, (x,y), (w,h), (0, 0, 255), 2)
                    cv2.putText(img, pred_class[i], (x+3, y+12), font, 0.3, (50, 200, 50), 1)

            # Display FPS(Frames per loop)
            FPS = str(round((1 / (time.time() - loop_time)), 1))
            cv2.putText(img, FPS, (15, 35), font, 1, (255, 0, 0), 3)

            cv2.imshow('screen', img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

if __name__ == "__main__":
    main()