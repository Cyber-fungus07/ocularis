# Image Caption Generator (College Project)

A clean, simple Image Captioning application using PyTorch (ResNet-50 + LSTM) and Streamlit.

## Project Files
- `app.py`: Streamlit frontend. Upload an image and get the caption.
- `model.py`: Model architecture (Encoder, Decoder, Seq2Seq) and beam search inference.
- `checkpoint.pth`: Model weights and vocabulary (placed here after training).
- `requirements.txt`: Required dependencies.

## How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
```
Open `http://localhost:8501` in your browser.
