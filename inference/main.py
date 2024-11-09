import streamlit as st
import pandas as pd
from io import StringIO
import requests


def model(data):
    pass


# Функция для загрузки видео
def load_video():
    uploaded_file = st.file_uploader('Выберите видео для детекции нарушений', type=['mp4', 'mov'])
    if uploaded_file is not None:

        video_bytes = uploaded_file.read()

        start_time = 10
        end_time = 20

        st.video(video_bytes, start_time=start_time, end_time=end_time)

        return uploaded_file.read()
    return None


# Функция для обращения к модели ИИ
def analyze_video(video_data):
    model(data=load_video)


# Основная функция приложения
def main():
    st.title("Анализ видео с помощью ИИ")

    video_data = load_video()

    if video_data:
        if st.button("Анализировать видео"):
            results = analyze_video(video_data)

            # Результат содержит список нарушений
            violations = results.get('violations', [])
            if violations:
                # Таблица
                df = pd.DataFrame(violations)
                st.write("Общее количество нарушений:", len(violations))
                st.write("Типы нарушений:")
                st.dataframe(df)
            else:
                st.write("Нарушения не обнаружены")


main()
