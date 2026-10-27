import torch
import streamlit as st
from PIL import Image
from model import (
    Encoder, Decoder, Seq2SeqCaptioner,
    get_resnet_extractor, extract_feature, generate_caption, DEVICE,
)

st.title("Image Caption Generator")


@st.cache_resource
def load_models():
    extractor = get_resnet_extractor()
    ckpt = torch.load("checkpoint.pth", map_location=DEVICE)
    idx2word = {int(k): v for k, v in ckpt["vocab"]["idx2word"].items()}
    model = Seq2SeqCaptioner(Encoder(), Decoder(vocab_size=len(idx2word)))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE).eval()
    return extractor, model, idx2word


extractor, model, idx2word = load_models()

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image)

    if st.button("Generate Caption"):
        feature = extract_feature(image, extractor)
        caption = generate_caption(model, feature, idx2word)
        st.write(caption)
