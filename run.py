import os

# torch
import torch
from torch.utils.data import DataLoader
from safetensors.torch import load_file

# other models
import segmentation_models_pytorch as smp
from ultralytics import YOLO
from torchvision.models import resnet50, ResNet50_Weights

# violation detection
from models import ViolationDetectionModel
from models.utils import Dataset

# other
import numpy as np
import pandas as pd
from tqdm import tqdm

# torch device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# base paths
ROOT_DIR = os.path.join(os.getcwd())
DATA_DIR = os.path.join(ROOT_DIR, 'data')
VAL_VIDEOS_DIR = os.path.join(DATA_DIR, 'val_videos')
SUBMISSION_PATH = os.path.join(ROOT_DIR, 'submission.csv')

# checkpoints
UNET_CHECKPOINT_PATH = os.path.join(ROOT_DIR, 'checkpoints', 'unet_segment_resnet50.safetensors')
YOLO_SIGNS_CHECKPOINT_PATH = os.path.join(ROOT_DIR, 'checkpoints', 'road-signs-yolov8n.pt')
YOLO_TRAFFIC_LIGHTS_CHECKPOINT_PATH = os.path.join(ROOT_DIR, 'checkpoints', 'traffic-lights-yolov8n.pt')
MODEL_CHECKPOINT_PATH = os.path.join(ROOT_DIR, 'checkpoints', 'violation_model.safetensors')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LEARN_FROM_ZERO = False


# load models
frame_backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
model = ViolationDetectionModel(frame_backbone).to(DEVICE)
state_dict = load_file(MODEL_CHECKPOINT_PATH)
model.load_state_dict(state_dict)

unet_state = load_file(UNET_CHECKPOINT_PATH)
unet = smp.Unet(
    encoder_name='resnet50',
    in_channels=3,
    classes=1
).to(DEVICE)
unet.load_state_dict(unet_state)

yolo_signs = YOLO(YOLO_SIGNS_CHECKPOINT_PATH)
yolo_traffic_lighs = YOLO(YOLO_TRAFFIC_LIGHTS_CHECKPOINT_PATH)

# dataset
dataset = Dataset(
    vid_dir=VAL_VIDEOS_DIR,
    width=736,
    height=416,
    mask_encoder=unet,
    disc_freq=1,
    device=DEVICE
)

dataloader = DataLoader(
    dataset=dataset,
    batch_size=8
)

predictions = []
# model.eval()
with torch.no_grad():
    for batch in tqdm(dataloader):
        img, mask = batch
        pred = model(img, mask)
        predictions.extend(torch.argmax(pred, dim=-1).tolist())

# make up submition
video_paths = []

# get video paths
for filename in os.listdir(VAL_VIDEOS_DIR):
    # ignore txt
    if filename.endswith('.txt'):
        continue

    video_path = os.path.join(VAL_VIDEOS_DIR, filename)
    video_paths.append(video_path)

label_encoder = dataset.encoder
videos_len = (np.array(dataset.videos_len) / np.array(dataset.fps)).round().astype(int).tolist()

submit_df = pd.DataFrame(columns=['номер видео', 'наименование нарушения', 'время нарушения (в секундах)'])
submit_df['наименование нарушения'] = label_encoder.inverse_transform(predictions)

window_start = 0
for idx, duration in enumerate(videos_len):
    submit_df.iloc[window_start:window_start+duration, submit_df.columns.get_loc('время нарушения (в секундах)')] = list(range(1, duration + 1))
    submit_df.iloc[window_start:window_start+duration, submit_df.columns.get_loc('номер видео')] = [video_paths[idx].split(os.path.sep)[-1].split('.')[0]] * duration
    window_start = window_start + duration

label_encoder = dataset.encoder
videos_len = (np.array(dataset.videos_len) / np.array(dataset.fps)).round().astype(int).tolist()

submit_df = pd.DataFrame(columns=['номер видео', 'наименование нарушения', 'время нарушения (в секундах)'])
submit_df['наименование нарушения'] = label_encoder.inverse_transform(predictions)

window_start = 0
for idx, duration in enumerate(videos_len):
    submit_df.iloc[window_start:window_start+duration, submit_df.columns.get_loc('время нарушения (в секундах)')] = list(range(1, duration + 1))
    submit_df.iloc[window_start:window_start+duration, submit_df.columns.get_loc('номер видео')] = [video_paths[idx].split(os.path.sep)[-1].split('.')[0]] * duration
    window_start = window_start + duration

submit_df[submit_df['наименование нарушения'] != 'nothing'].to_csv(SUBMISSION_PATH, index=False)
