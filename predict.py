import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

IMG_SIZE = 150

model = tf.keras.models.load_model("C:/Users/Irfan Haider Attash/Documents/Cat_Vs_Dog_classification_Project/models/cat_dog_model.h5")

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    if prediction[0][0] > 0.5:
        return "Dog"
    else:
        return "Cat"

if __name__ == "__main__":
    result = predict_image("C:/Users/Irfan Haider Attash/Documents/Cat_Vs_Dog_classification_Project/2.jpg")
    print("Prediction:", result)