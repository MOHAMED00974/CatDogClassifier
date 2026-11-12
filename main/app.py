import streamlit as st
from PIL import Image
from logic import predict

st.set_page_config(page_title="Cat or Dog?", page_icon="🐾")
st.title("Cat or Dog Image Classifier")
st.write("Upload an image to get started")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)
        st.success("Image uploaded successfully.")
        if st.button('Classify'): 
            label= predict(image)
            st.write(f'Your image is {label}')
    except Exception as error:
        st.error(f"Sorry, an error happened: {error}")
        print(error)

