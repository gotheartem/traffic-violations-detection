import streamlit as st
import pandas as pd
import ffmpeg
import tempfile
import os
import io


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
ROOT_DIR = os.path.join(os.getcwd(), os.pardir)
DATA_DIR = os.path.join(ROOT_DIR, 'data')
TMP_VIDEOS_DIR = os.path.join(ROOT_DIR, 'inference', 'tmp')
SUBMISSION_PATH = os.path.join(ROOT_DIR, 'submission.csv')

# checkpoints
UNET_CHECKPOINT_PATH = os.path.join(ROOT_DIR, 'checkpoints', 'unet_segment_resnet50.safetensors')
YOLO_SIGNS_CHECKPOINT_PATH = os.path.join(ROOT_DIR, 'checkpoints', 'road-signs-yolov8n.pt')
YOLO_TRAFFIC_LIGHTS_CHECKPOINT_PATH = os.path.join(ROOT_DIR, 'checkpoints', 'traffic-lights-yolov8n.pt')
MODEL_CHECKPOINT_PATH = os.path.join(ROOT_DIR, 'checkpoints', 'violation_model.safetensors')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LEARN_FROM_ZERO = False


@st.cache_resource
def convert(input_video_bytes, input_format):
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{input_format}") as temp_input_file:
        temp_input_file.write(input_video_bytes)
        temp_input_path = temp_input_file.name

    temp_output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

    try:
        (ffmpeg
        .input(temp_input_path)
        .output(
            temp_output_path,
            vcodec='libx264',
            acodec='aac')
        .run(
            quiet=True,
            overwrite_output=True
        )
        )
    except ffmpeg.Error as e:
        st.error(f"Error converting video: {e}")
        return None

    with open(temp_output_path, 'rb') as output_file:
        output_video_bytes = output_file.read()

    os.remove(temp_input_path)
    os.remove(temp_output_path)

    return output_video_bytes


@st.cache_resource
def load_video(uploaded_file):
    if uploaded_file is not None:
        input_format = uploaded_file.name.split('.')[-1]
        video_bytes = uploaded_file.read()

        if input_format != 'mp4':
            video_bytes = convert(video_bytes, input_format)

        return video_bytes
    return None


@st.cache_resource
def analyze_video(video_data):
    with open('tmp/tmp.mp4', "wb") as f:
        f.write(video_data)

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

    # dataset
    dataset = Dataset(
        vid_dir='tmp',
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
    for filename in os.listdir(TMP_VIDEOS_DIR):
        # ignore txt
        if filename.endswith('.txt'):
            continue

        video_path = os.path.join(TMP_VIDEOS_DIR, filename)
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

    s = submit_df[submit_df['наименование нарушения'] != 'nothing']
    violations = []
    for row in s[['наименование нарушения', 'время нарушения (в секундах)']].iterrows():
        violation = row[1][0]
        time = row[1][1]
        violations.append({violation: time})

    return {'violations': violations}


def main():
    st.title("Анализ видео с помощью ИИ")

    uploaded_file = st.file_uploader('Выберите видео для детекции нарушений',
                                     type=['mp4', 'mov', 'wmv', 'avi', 'flv', 'mkv', 'webm', 'mpg', 'mts', 'swf'])

    if uploaded_file is not None:
        with st.spinner(text="Обрабатываем видео..."):
            video_data = load_video(uploaded_file)

        if video_data:
            if "current_time" not in st.session_state or st.session_state.current_time == -1:
                st.session_state.current_time = -1
                autoplay = False
                st.video(video_data)
            else:
                autoplay = True
                start_time = max(0, st.session_state.current_time - 5)
                end_time = st.session_state.current_time + 5
                st.video(video_data, start_time=start_time, end_time=end_time, autoplay=autoplay, loop=autoplay)

            if autoplay and st.button("Перезагрузить видео"):
                st.session_state.current_time = -1
                st.rerun()

            with st.spinner(text="Анализируем видео..."):
                results = analyze_video(video_data)

            violations = results.get('violations', [])
            if violations:
                df = pd.DataFrame(violations)
                buffer = io.BytesIO()
                df.to_csv(buffer, index=False)
                st.write("Общее количество нарушений:", len(violations))
                st.download_button("Скачать результаты", buffer, file_name='violations.csv', use_container_width=True)
                st.header("Нарушения")

                cols = st.columns([2, 2, 2])
                with cols[0]:
                    st.write("<b>Тип</b>", unsafe_allow_html=True)
                with cols[1]:
                    st.write("<b>Время</b>", unsafe_allow_html=True)
                with cols[2]:
                    pass

                for idx, row in df.iterrows():
                    cols = st.columns([2, 2, 2])
                    with cols[0]:
                        st.write(row['type'])
                    with cols[1]:
                        st.write(row['timestamp'])
                    with cols[2]:
                        if st.button(f"Перейти к {row['timestamp']}", key=f"button_{idx}", use_container_width=True):
                            st.session_state.current_time = row['timestamp']
                            st.rerun()

            else:
                st.write("Нарушения не обнаружены")


main()