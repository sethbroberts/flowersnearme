import streamlit as st 
import numpy as np
from PIL import Image
from classify import predict

def main():
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.title("Flowers Around Here")

    uploaded_file = st.file_uploader("Choose an image of a flower...")
    if uploaded_file is not None:
        image = np.asarray(Image.open(uploaded_file))
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        labels, probs = predict(image)
        for l, p in list(zip(labels, probs)):
            st.write(f"{l}, prob={p*100}")

if __name__ == '__main__':
    main()