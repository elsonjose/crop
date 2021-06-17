import flask
import urllib.request
from utils.torch_utils import select_device
from models.experimental import attempt_load
import torch
from utils.general import check_img_size,non_max_suppression,scale_coords
import cv2 as cv
import numpy as np
from torch import Tensor
from classes import Plate,Char
from sort import sortCharacters,sortPlatePossibilities
from itertools import product

app=flask.Flask(__name__)

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv.resize(img, new_unpad, interpolation=cv.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv.copyMakeBorder(img, top, bottom, left, right, cv.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def detectChars(org_img):
    device = 'cpu'
    imgsize_char=320
    half = False

    char_weights = 'detect_char_weight.pt'    
    model_char = attempt_load(char_weights, map_location=device)

    stride = int(model_char.stride.max())  # model_char stride
    imgsize_char = check_img_size(imgsize_char, s=stride)  # check img_size
    if half:
        model_char.half()

    img = letterbox(org_img, imgsize_char)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float() 
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    # Inference
    pred = model_char(img, augment=False)[0]
    pred = non_max_suppression(pred, 0.4, 0.5, classes=0, agnostic=False)
    for i, det in enumerate(pred):
        gn = torch.tensor(img.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if det is not None and len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], org_img.shape).round()
            chars=[]
            for *xyxy, conf, cls in reversed(det):
                if(conf.item()>0.59):
                    x1, y1,x2,y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                    crop = org_img[y1:y2, x1:x2]
                    chars.append(Char(x1,x2,y1,y2,crop))

    return chars

def detechPlate(source_img):
    device = 'cpu'
    imgsize_plate=640
    half = False
    org_img = source_img

    plate_weights = 'detect_plate_weight.pt'
    model_plate = attempt_load(plate_weights, map_location=device)

    stride = int(model_plate.stride.max())  # model_char stride
    imgsize_plate = check_img_size(imgsize_plate, s=stride)  # check img_size
    if half:
        model_plate.half()


    img = letterbox(org_img, imgsize_plate)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    # Inference
    pred = model_plate(img, augment=False)[0]
    pred = non_max_suppression(pred, 0.4, 0.5, classes=0, agnostic=False)
    for i, det in enumerate(pred):
        gn = torch.tensor(img.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if det is not None and len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], org_img.shape).round()
            plates=[]
            for *xyxy, conf, cls in reversed(det):
                if(conf.item()>0.59):
                    x1, y1,x2,y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                    crop = org_img[y1:y2, x1:x2]
                    plates.append(Plate(crop))
    return plates



@app.route('/',methods=['POST','GET'])
def recognizePlate():
    if flask.request.method=='POST':
        count = int(flask.request.form.get('count')) if int(flask.request.form.get('count')) is not None else 10
        print(count)
        url = flask.request.form.get('url')
        url_response = urllib.request.urlopen(url)
        img_array = np.array(bytearray(url_response.read()), dtype=np.uint8)
        img = cv.imdecode(img_array, -1)

        device = 'cpu'
        char_recog_weights = 'recog_char_weight.pt'    
        classifier = torch.load(char_recog_weights,map_location=device).eval()

        keyMap={
            0:'0',1:'1',2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',
            10:'A',11:'B',12:'C',13:'D',14:'E',15:'F',16:'G',17:'H',18:'I',
            19:'J',20:'K',21:'L',22:'M',23:'N',24:'O',25:'P',26:'Q',27:'R',
            28:'S',29:'T',30:'U',31:'V',32:'W',33:'X',34:'Y',35:'Z'
        }

        plateList = detechPlate(img)
        plate_count = 0;
        if len(plateList)==0:
            return flask.jsonify(
                {
                    'result':'No plates found'
                }
            )
        else:
            plates=[]
            for plate in plateList:
                plate_count+=1    
                charList = detectChars(plate.plate_image)
                charList = sortCharacters(charList)
                group_char_score=[]
                if len(charList) ==0:
                    plates.append({
                            plate_count:'No characters found'
                        })
                else:
                    for ch in charList:
                        img_gray = cv.cvtColor(cv.resize(ch.char_image,(28,28)), cv.COLOR_BGR2GRAY)
                        ret, thresh1 = cv.threshold(img_gray, 120, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
                        binary_img = ~thresh1
                        binary_img_np = np.array(binary_img)
                        img = binary_img_np / 255.0
                        img_tensor = Tensor(img).view(1, 28, 28).float()
                        img_tensor=img_tensor.unsqueeze(0)
                        img_tensor = img_tensor.to(device)
                        pred = classifier.forward(img_tensor)
                        
                        sorted_indices = torch.argsort(pred.data,descending=True).tolist()[0][:2]
                        score = pred.data.tolist()[0];

                        min_val_in_score = min(score)
                        if(min_val_in_score<0):
                            abs_min_val_in_score = abs(min_val_in_score)
                            for i in range(len(score)):
                                score[i]+=abs_min_val_in_score
                        # normalize score to a max of 100
                        max_score = max(score)
                        factor = 100/max_score
                        for i in range(len(score)):
                                score[i]=(score[i]*factor)

                        char_score=[]
                        for i in sorted_indices:
                            char=keyMap[i]
                            val=score[i]
                            char_score.append((char,val))  
                        
                        # print(pred.data)
                        # y_hat = torch.argmax(pred.data)
                        # print('\n\n')
                        # print(torch.argsort(pred.data,descending=True))
                        # print(pred.data.tolist())
                        # index = y_hat.cpu().numpy().tolist()
                        # print(keyMap[index])    
                        
                        group_char_score.append(char_score)

                    possible_plates=sortPlatePossibilities(list(product(*group_char_score)))
                    
                    possibilities=[]
                    for subl in possible_plates:
                        plate = ''.join(t[0] for t in subl)
                        score = str((sum(t[1] for t in subl)/len(subl)))
                        possibilities.append(
                            {
                                'plate':plate,
                                'score':score
                            }
                        )
                        if count<0:
                            break
                        count-=1
                    plates.append({
                            plate_count:possibilities
                        })
            return flask.jsonify(
                        {
                            'result':'Plates found',
                            'plates':plates,
                        }
                    )

    else:
        return flask.jsonify(
            {
                'result':'Bad request'
            }
        )

if __name__=="__main__":
    app.run()