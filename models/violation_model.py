import torch
import torch.nn as nn
import torch.nn.functional as F


class ViolationDetectionModel(nn.Module):
    def __init__(self, frame_backbone):
        super().__init__()
        self.frame_backbone = frame_backbone
        self.seq_frame_in = self._make_sequential(1000, 512)
        self.seq_mask_in = self._make_sequential(164 * 164, 4096)
        self.seq_combine = self._make_sequential(4096 + 512, 2048)
        self.fc1 = self._make_sequential(2048, 512)
        self.fc2 = self._make_sequential(512, 256)
        self.fc3 = self._make_sequential(256, 128)
        self.fc4 = self._make_sequential(128, 32)
        self.fc_out = self._make_sequential(32, 6)
        
    def forward(self, frame, mask):
        # flatten mask
        mask = torch.flatten(mask, start_dim=1)

        # pass frame and its mask through backbones
        frame_backbone_out = self.frame_backbone(frame)
        mask_out = self.seq_mask_in(mask)

        # combine outputs
        frame_out = self.seq_frame_in(frame_backbone_out)
        combined = self.seq_combine(torch.hstack([frame_out, mask_out]))

        # get class probs
        out = self.fc1(combined)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        out = self.fc_out(out)
        out = F.softmax(out, dim=-1)

        return out

    def _make_sequential(self, input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Dropout(p=0.3),
            nn.ReLU()
        )