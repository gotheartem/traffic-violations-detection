import streamlit as st
import pandas as pd
import ffmpeg
import tempfile
import os
import io


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

        if input_format not in ['mp4', 'mov']:
            video_bytes = convert(video_bytes, input_format)

        return video_bytes
    return None


@st.cache_resource
def analyze_video(video_data):
    return {'violations': [{'type': 'Заезд за стоп-линию', 'timestamp': 10},
                           {'type': 'Выезд на встречную полосу', 'timestamp': 20},
                           {'type': 'Движение по полосам для автобусов/троллейбусов/трамваев и остановка на них', 'timestamp': 30},
                           {'type': 'Нарушение знаков и разметки', 'timestamp': 40},
                           {'type': 'Движение по полосам для автобусов/троллейбусов/трамваев и остановка на них', 'timestamp': 60},
                           {'type': 'Поворот налево и разворот через сплошную или при нарушении знаков', 'timestamp': 80}]}


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
                    st.write("Тип")
                with cols[1]:
                    st.write("Время")
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