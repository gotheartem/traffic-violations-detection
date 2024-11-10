import streamlit as st
import pandas as pd
import ffmpeg
import tempfile
import os

# Function to convert video bytes to MP4
def convert_to_mp4(input_video_bytes, input_format):
    # Write input video bytes to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{input_format}") as temp_input_file:
        temp_input_file.write(input_video_bytes)
        temp_input_path = temp_input_file.name

    # Define the output path for the converted mp4 file
    temp_output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

    # Use ffmpeg to convert the video to mp4
    try:
        ffmpeg.input(temp_input_path).output(temp_output_path, vcodec='libx264', acodec='aac').run(quiet=True, overwrite_output=True)
    except ffmpeg.Error as e:
        st.error(f"Error converting video: {e}")
        return None

    # Read the converted video bytes
    with open(temp_output_path, 'rb') as output_file:
        output_video_bytes = output_file.read()

    # Clean up temporary files
    os.remove(temp_input_path)
    os.remove(temp_output_path)

    return output_video_bytes

# Function to load and display video
@st.cache_resource
def load_video(uploaded_file):
    if uploaded_file is not None:
        input_format = uploaded_file.name.split('.')[-1]  # Get the file extension to determine the format
        video_bytes = uploaded_file.read()

        if input_format not in ['mp4', 'mov']:
            # Convert the video to mp4 format
            video_bytes = convert_to_mp4(video_bytes, input_format)

        return video_bytes
    return None

# Function to analyze the video (stub function)
def analyze_video(video_data):
    # Placeholder for model analysis
    st.write("Анализируем видео...")
    return {'violations': [{'type': 'Speeding', 'timestamp': '00:01:23'}, 
                           {'type': 'Lane Change', 'timestamp': '00:02:45'}, 
                           {'type': 'Lane Change', 'timestamp': '00:02:45'}, 
                           {'type': 'Lane Change', 'timestamp': '00:02:45'}, 
                           {'type': 'Lane Change', 'timestamp': '00:02:45'}, 
                           {'type': 'Lane Change', 'timestamp': '00:02:45'}, 
                           {'type': 'Lane Change', 'timestamp': '00:02:45'}, 
                           {'type': 'Lane Change', 'timestamp': '00:02:45'}, 
                           {'type': 'Lane Change', 'timestamp': '00:02:45'}, 
                           {'type': 'Lane Change', 'timestamp': '00:02:45'}]}  # Example result

# Main function of the app
def main():
    st.title("Анализ видео с помощью ИИ")

    uploaded_file = st.file_uploader('Выберите видео для детекции нарушений', type=['mp4', 'mov', 'wmv', 'avi', 'flv', 'mkv', 'webm', 'mpg', 'mts', 'swf'])
    video_data = load_video(uploaded_file)

    if video_data:
        st.video(video_data)

        if st.button("Анализировать видео"):
            results = analyze_video(video_data)

            # Result contains a list of violations
            violations = results.get('violations', [])
            if violations:
                # Display results in a table with buttons in each row
                df = pd.DataFrame(violations)
                st.write("Общее количество нарушений:", len(violations))
                st.write("Типы нарушений:")
                
                # Display DataFrame with buttons inline
                for idx, row in df.iterrows():
                    cols = st.columns([2, 2, 2])
                    with cols[0]:
                        st.write(row['type'])
                    with cols[1]:
                        st.write(row['timestamp'])
                    with cols[2]:
                        if st.button(f"Перейти к {row['timestamp']}", key=f"button_{idx}"):
                            st.write(f"Переход к времени: {row['timestamp']} (здесь должна быть логика перехода в видео)")
            else:
                st.write("Нарушения не обнаружены")

if __name__ == "__main__":
    main()
