import torch

# Load Model
model = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model='models/model_20210428.pt')

# Model attributes, currently set as:
model.conf = 0.08  # confidence threshold (0-1)
model.iou = 0.3  # NMS IoU threshold (0-1)

# Load Image
img_path = 'input/IMG_0078.jpg'
# a variety of input format supported:
#   filename:   img = 'data/zidane.jpg'
#   URI:             = 'https://github.com/ultralytics/yolov5/releases/download/v1.0/zidane.jpg'
#   OpenCV:          = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(720,1280,3)
#   PIL:             = Image.open('image.jpg')  # HWC x(720,1280,3)
#   numpy:           = np.zeros((720,1280,3))  # HWC
#   torch:           = torch.zeros(16,3,720,1280)  # BCHW
#   multiple:        = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images

# Inference
results = model(img_path, size=640)
confidence = results.pred[0].T[4].item()

# Post processing
results.print()  # or .show(), .save()
print("confidence:{:.2f}".format(confidence))
results.save()  # save processed images to directory /results/

items_detected = results.pred[0].shape[0]
guide_wire = True if items_detected>0 else False
print("Guide wire exists!" if guide_wire else "No wire!")