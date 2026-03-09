import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="AI Cat vs Dog Classification",
    page_icon="🐾",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>

.stApp {
background-color:#f4f6f9;
}

.title{
text-align:center;
font-size:80px;
font-weight:bold;
color:#4A90E2;
}

.subtitle{
text-align:center;
color:gray;
margin-bottom:20px;
}

.center-box{
background:white;
padding:30px;
border-radius:12px;
box-shadow:0px 5px 20px rgba(0,0,0,0.1);
text-align:center;
}

.side-img{
height:90vh;
object-fit:cover;
border-radius:10px;
}

</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("C:/Users/Irfan Haider Attash/Documents/Cat_Vs_Dog_classification_Project/models/cat_dog_model.h5")

model = load_model()

IMG_SIZE = 150

def predict_image(img):

    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        label = "Dog 🐶"
        confidence = prediction * 100
    else:
        label = "Cat 🐱"
        confidence = (1 - prediction) * 100

    return label, confidence


# Layout with 3 columns
col1, col2, col3 = st.columns([1,2,1])

# LEFT DOG IMAGE
with col1:
    st.image(
        "C:/Users/Irfan Haider Attash/Documents/Cat_Vs_Dog_classification_Project/image/dog.jfif",
        use_container_width=True
    )

# CENTER AI SYSTEM
with col2:

    st.markdown('<p class="title">🐶🐱 AI Cat vs Dog Classifier</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Upload an image and AI will classify it instantly</p>', unsafe_allow_html=True)

    st.markdown('<div class="center-box">', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "📂 Drag & Drop or Upload Image",
        type=["jpg","jpeg","png"]
    )

    if uploaded_file:

        image = Image.open(uploaded_file)

        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("🔍 Predict Image"):

            label, confidence = predict_image(image)

            st.success(f"Prediction: {label}")
            st.info(f"Confidence: {confidence:.2f}%")

            st.progress(int(confidence))

    st.markdown('</div>', unsafe_allow_html=True)

# RIGHT CAT IMAGE
with col3:
    st.image(
        "C:/Users/Irfan Haider Attash/Documents/Cat_Vs_Dog_classification_Project/image/cat.jfif",
        use_container_width=True
    )

# Footer
st.write("")
st.markdown("---")
st.caption("This Project Built By Engr Irfan Haider with ❤️ using Streamlit & TensorFlow")