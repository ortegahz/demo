# modules
import sys
import cv2
import torch

from torchvision.transforms import ToTensor
from PIL import Image

import classes

sys.path.append('/home/manu/mnt/kingstop/workspace/BackgroundMattingV2')
from dataset import VideoDataset
from model import MattingBase, MattingRefine

# arguments
model_type = 'mattingrefine'
model_backbone = 'resnet50'
model_backbone_scale = 0.25
model_refine_mode = 'full'
model_refine_sample_pixels = 80000
model_refine_threshold = 0.7
model_checkpoint = '/home/manu/tmp/epoch-3-iter-207261-vloss-0.013960338487448369-vlossl-0.013960338487448369.pth'

path_video = 'rtsp://192.168.1.31:554/live/av0'
# path_video = '/media/manu/samsung/videos/modnet/正常.mp4'
window_width, window_height = 1920, 1080

# Load model
if model_type == 'mattingbase':
    model = MattingBase(model_backbone)
if model_type == 'mattingrefine':
    model = MattingRefine(
        model_backbone,
        model_backbone_scale,
        model_refine_mode,
        model_refine_sample_pixels,
        model_refine_threshold)

model = model.cuda().eval()
model.load_state_dict(torch.load(model_checkpoint), strict=False)

decoder = classes.Decoder(path_video)
displayer = classes.Displayer('demo_bgrm', window_width, window_height)


def cv2_frame_to_cuda(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return ToTensor()(Image.fromarray(frame)).unsqueeze_(0).cuda()


with torch.no_grad():
    while True:
        bgr = None
        while True:  # grab bgr
            frame = decoder.read()
            key = displayer.step(frame)
            if key == ord('b'):
                bgr = cv2_frame_to_cuda(decoder.read())
                break
            elif key == ord('q'):
                exit()
        while True:  # matting
            frame = decoder.read()
            src = cv2_frame_to_cuda(frame)
            pha, fgr = model(src, bgr)[:2]
            res = pha * fgr + (1 - pha) * torch.ones_like(fgr)
            res = res.mul(255).byte().cpu().permute(0, 2, 3, 1).numpy()[0]
            res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
            key = displayer.step(res)
            if key == ord('b'):
                break
            elif key == ord('q'):
                exit()
