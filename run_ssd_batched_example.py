# Detectnet SSD300 metVGG16 basenet via Torch en OpenCV voor .pth modellen als toetsing voor TRT detectnet na ONNX conversie.

from urllib.request import proxy_bypass
from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenet_v2_ssd_lite import (
    create_mobilenetv2_ssd_lite,
    create_mobilenetv2_ssd_lite_predictor,
)
from vision.utils.misc import Timer
from torchvision.models.resnet import resnet50
import cv2
import sys
import os

import torch
from torchvision import transforms
import cv2
import numpy as np
import types
from numpy import random

print("Started OpenCV Torch detectnet version 0.1")

if len(sys.argv) < 5:
    print(
        "Usage: python run_ssd_batched_example.py  <model .pth path> <label.txt path> <images path> <output path>"
    )
    sys.exit(0)

mdlPath = sys.argv[1]
labelstxtPath = sys.argv[2]
imagesFolder = sys.argv[3]
outputFolder = sys.argv[4]

os.chdir(outputFolder)

print("Outdir set: " + outputFolder)

classes = [name.strip() for name in open(labelstxtPath).readlines()]

detectionNetwork = create_vgg_ssd(
    len(classes), is_test=True
)  # create SSD netw with base typeVG16
# detectionNetwork = create_mobilenetv2_ssd_lite(len(classes), is_test=True) #create SSD netw with base typeVG16
detectionNetwork.load(mdlPath)

predictor = create_vgg_ssd_predictor(detectionNetwork, candidate_size=200)  # predictor

# loop through images in folder and feed them to the predictor with openCV
queue = []

for file in os.listdir(imagesFolder):
    objectPath = os.path.join(imagesFolder, file)
    if os.path.isfile(objectPath):
        queue.append(objectPath)
        print("Enqueued image located at: " + objectPath)
import time

batch_queue = []

# read paths in queue and run ssd300 algo
iFrame = 0
iter = 0

for path in queue:
    if iFrame == 1:
        torch.cuda.synchronize()
        start_time = time.time()
    iFrame = iFrame + 1
    batch_queue.append(path) #enqueue path to batch job
    iter = iter + 1
    if iter == 8: #run batched job
        iter = 0
        # print("Processing images batch: " + str(batch_queue))
        img0 = cv2.imread(batch_queue[0])  # load
        img1 = cv2.imread(batch_queue[1]) 
        img2 = cv2.imread(batch_queue[2]) 
        img3 = cv2.imread(batch_queue[3]) 
        img4 = cv2.imread(batch_queue[4]) 
        img5 = cv2.imread(batch_queue[5]) 
        img6 = cv2.imread(batch_queue[6]) 
        img7 = cv2.imread(batch_queue[7]) 

        pi0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)  # preprocessing
        pi1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB) 
        pi2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB) 
        pi3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB) 
        pi4 = cv2.cvtColor(img4, cv2.COLOR_BGR2RGB) 
        pi5 = cv2.cvtColor(img5, cv2.COLOR_BGR2RGB) 
        pi6 = cv2.cvtColor(img6, cv2.COLOR_BGR2RGB)
        pi7 = cv2.cvtColor(img7, cv2.COLOR_BGR2RGB)

        batcharr = [
            pi0, pi1,pi2, pi3, pi4, pi5, pi6, pi7
        ]
        batch_queue = []
        result = predictor.predict(
            batcharr, 1, 100, 0.1
        )  # run prediction on processed image

        print("BACK: " + str(result))

        for batch_idx, res_tpl in enumerate(result):
            print("TPL: " + str(res_tpl))
            if type(res_tpl) is tuple:
                (bounding_box, labels, probability) = res_tpl
                print("BATCH_IDX: " + str(batch_idx))
                # postprocess batched results:
                for i in range(bounding_box.size(0)):
                    box = bounding_box[i, :]
                    # fetch original from batch
                    oImg = batcharr[batch_idx]
                    cv2.rectangle(oImg, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 4)
                    label = f"{classes[labels[i]]} p[{probability[i]:.2f}]"
                    cv2.putText(oImg, label, (int(box[0]), int(box[1]) - 20), cv2.FONT_HERSHEY_DUPLEX,1,(0, 0, 255),2)
                outName = str(iFrame + batch_idx) + ".jpg"
                print("Write result: " + outName)
                cv2.imwrite(outName, oImg)
                print(
                    "Detected "
                    + str(len(probability))
                    + " objects in "
                    + str(batch_idx)
                )

torch.cuda.synchronize()
end_time = time.time()
print("Loc: time={}".format(end_time - start_time))
ft = end_time - start_time
frame_time = ft / iFrame
print("frame: time={}".format(frame_time))