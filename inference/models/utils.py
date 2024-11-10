import os
import torch
import cv2
import joblib

from torch.utils.data import Dataset
from torchvision.transforms import transforms
from decord import VideoReader, cpu
from PIL import Image

# base paths
ROOT_DIR = os.path.join(os.getcwd())
DATA_DIR = os.path.join(ROOT_DIR, 'data')
ENCODER_PATH = os.path.join(ROOT_DIR, 'models', 'label_encoder.sk')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LEARN_FROM_ZERO = False


class Dataset(Dataset):
    def __init__(self, vid_dir, width, height, mask_encoder, 
                 disc_freq=1, device='cpu'):
        self.vid_dir = vid_dir
        self.videos_names = []
        self.videos_len = []
        self.fps = []
        self.videos_samples_len = []
        self.width = width
        self.height = height
        self.device=device
        self.mask_encoder = mask_encoder
        self.disc_freq = disc_freq
        self.encoder = joblib.load(ENCODER_PATH)
        self.video = [None, None]
        
        # Preload video metadata and prepare decoders
        for filename in os.listdir(vid_dir):
            if not filename.endswith('.txt'):
                filepath = os.path.join(vid_dir, filename)
                self.videos_names.append(filename)

                with open(filepath, 'rb') as f:
                    video_reader = VideoReader(f, ctx=cpu(0))

                self.videos_len.append(len(video_reader))
                self.fps.append(video_reader.get_avg_fps())

                assert disc_freq <= self.fps[-1]
                self.videos_samples_len.append(int(self.videos_len[-1] / self.fps[-1] * disc_freq))

    def __len__(self):
        return sum(self.videos_samples_len)

    def __getitem__(self, idx):
        # Determine which video this idx falls into
        for vid_idx in range(len(self.videos_names)):
            if idx >= self.videos_samples_len[vid_idx]:
                idx -= self.videos_samples_len[vid_idx]
            else:
                break

        if vid_idx != self.video[0]:
            if self.video[1]:
                self.video[1].seek(0)
            with open(os.path.join(self.vid_dir, self.videos_names[vid_idx]), 'rb') as f:
                self.video = [vid_idx, VideoReader(f, ctx=cpu(0))]
        
        # Calculate the frame index based on disc_freq
        frame_sec = int(idx / self.disc_freq - 1e-8)
        frame_idx = int(idx / self.disc_freq * self.fps[vid_idx])
        
        # Use Decord to fetch the frame efficiently
        video_reader = self.video[1]
        frame = video_reader[frame_idx].asnumpy()
        
        # Preprocess image
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_CUBIC)   
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])

        frame = transform(frame)
        frame = frame.to(self.device)
        
        with torch.no_grad():
            mask_transform = transforms.Resize((164, 164), interpolation=transforms.InterpolationMode.BICUBIC)
            mask = self.mask_encoder(frame.unsqueeze(0))
            mask = mask.squeeze(1)
            mask = mask_transform(mask)
            mask = mask.to(torch.uint8)
            mask = mask.squeeze(0)
            mask = mask.to(torch.float32)
        
        return frame, mask