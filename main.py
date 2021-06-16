import torch
import sys
from PIL import UnidentifiedImageError

# Load Model
# model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/model_20210428.pt')    # from remote repo
model = torch.hub.load('models/ultralytics_yolov5_master', 'custom', path='models/model_20210428.pt', source='local')     # local repo

# Model attributes, currently set as:
model.conf = 0.08  # confidence threshold (0-1)
model.iou = 0.3  # NMS IoU threshold (0-1)

# Load Image
img_path = 'input/IMG_9959.jpg'
# a variety of input format supported:
#   filename:   img = 'data/zidane.jpg'
#   URI:             = 'https://github.com/ultralytics/yolov5/releases/download/v1.0/zidane.jpg'
#   OpenCV:          = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(720,1280,3)
#   PIL:             = Image.open('image.jpg')  # HWC x(720,1280,3)
#   numpy:           = np.zeros((720,1280,3))  # HWC
#   torch:           = torch.zeros(16,3,720,1280)  # BCHW
#   multiple:        = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images

# Inference
try:
    results = model(img_path, size=640)
    confidence = float(torch.mean(results.pred[0].T[4])) if results.pred[0].shape[0] > 0 else None
except OSError as err:
    print("OS error: {0}".format(err))
    exit()
except:
    print("Unexpected error:", sys.exc_info()[0])
    exit()

# Post processing
results.print()  # or .show(), .save()
if confidence is not None:
    print("confidence:{:.2f}".format(confidence))
results.save()  # save processed images to directory /results/

num_guide_wire = results.pred[0].shape[0]
guide_wire_exist = True if num_guide_wire>0 else False

# print("Guide wire exists!" if guide_wire_exist else "No wire!")
if guide_wire_exist:
    print("%d Guide Wire(s)"%num_guide_wire)
else:
    print("No wire!")


