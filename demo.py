from __future__ import print_function, division
import sys
import torch
import torchvision.models as models
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import resnet
import jsonlines

# usage: python demo.py <input video file> <output jsonl file name>
# output: *.jsonl

model = resnet.resnet50(num_classes=365, num_new_classes=26)
checkpoint = torch.load('lwf_best.pth.tar')
state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
model.load_state_dict(state_dict)
model = model.cuda().eval()

class_places = list()
with open('categories_places365.txt') as class_file:
    for line in class_file:
        class_places.append(line.strip().split(' ')[0][3:])

class_friends = ['none', 'cafe', 'home-livingroom-Monica', 'home-doorway-Monica', 'home-kitchen-Monica', 'home-livingroom-Ross', 'home-none-Ross', 'home-none-Monica', 'restaurant', 'cafe-doorway', 'home-none-none', 'home-kitchen-none', 'hospital', 'museum', 'museum-none-Ross', 'restaurant-none-Monica', 'home-livingroom-Chandler', 'road-none-none', 'office-none-none', 'home-livingroom-none', 'cafe-kitchen-none', 'home-none-Chandler', 'home-kitchen-Chandler', 'home-doorway-Chandler', 'office-none-Chandler', ' ']

transform = transforms.Compose([
		transforms.Resize((256,256)),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

cap = cv2.VideoCapture(sys.argv[1])
fps = 2

n = 0
frame_counter = 1e10
num_frames = 0
out_dict = []
with torch.no_grad():
	while cap.isOpened():
		ret, image = cap.read()
		if not ret:
			break
		
		data = transform(Image.fromarray(image[:,:,::-1])).unsqueeze(0).pin_memory().cuda(non_blocking=True)
		
		width = int(round(480/image.shape[0]*image.shape[1]))
		image_resized = np.array(transforms.Resize((480, width))(Image.fromarray(image)))
		
		frame_counter += 1
		if frame_counter >= (cap.get(cv2.CAP_PROP_FPS)/fps):
			output = model(data)
			y_places = F.softmax(output[0].view(-1), 0)
			y_friends = F.softmax(output[1].view(-1), 0)
			
			top5_value_places, top5_index_places = y_places.topk(5)
			top5_value_places, top5_label_places = top5_value_places.tolist(), [class_places[i] for i in top5_index_places]

			top5_value_friends, top5_index_friends = y_friends.topk(5)
			top5_value_friends, top5_label_friends = top5_value_friends.tolist(), [class_friends[i] for i in top5_index_friends]

			if num_frames!=0: out_dict.append({"type": "location", "class": top5_label_friends[0], "seconds": float(num_frames) * 1.0 / float(fps)})		
			

			frame_counter = 0
                        num_frames += 1
		n += 1
		if n%100 == 0:
			print('Processed {}/{} frames'.format(n, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))
			sys.stdout.flush()
		if n == 1e10:
			break


with jsonlines.open(str(sys.argv[2])+'.jsonl', mode='w') as writer:
	writer.write_all(out_dict)

cap.release()
print('Done')
