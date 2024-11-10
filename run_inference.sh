source .venv/bin/activate
echo "Current working directory: $(pwd)"
cd inference/
streamlit run main.py
cd ../