from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from PIL import Image
import os
import cv2
from mtcnn import MTCNN
import numpy as np

# Initialize the MTCNN detector and the VGGFace model
detector = MTCNN()
model = VGGFace(model='senet50', include_top=False, input_shape=(224, 224, 3))

# Load the features and filenames
feature_list = np.array(pickle.load(open("my_features", 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Function to save the uploaded image
def save_uploaded_image(uploaded_image):
    try:
        os.makedirs('uploads', exist_ok=True)
        file_path = os.path.join('uploads', uploaded_image.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_image.getbuffer())
        return file_path
    except Exception as e:
        print(f"An error occurred while saving the image: {e}")
        return None

# Function to extract features from the image
def extract_features(img_path, model, detector):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to read image from path: {img_path}")
        return None

    results = detector.detect_faces(img)
    if len(results) == 0:
        print(f"No faces detected in image: {img_path}")
        return None

    x, y, width, height = results[0]['box']
    face = img[y:y + height, x:x + width]

    image = Image.fromarray(face)
    image = image.resize((224, 224))

    face_array = np.asarray(image)
    face_array = face_array.astype('float32')

    expanded_img = np.expand_dims(face_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)
    result = model.predict(preprocessed_img).flatten()
    return result

# Function to recommend the most similar celebrity
def recommend(features, feature_list):
    similarity = []
    for i in range(len(feature_list)):
        similarity.append(cosine_similarity(features.reshape(1, -1), feature_list[i].reshape(1, -1))[0][0])

    index_pos = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][0]
    return index_pos

# Streamlit app
st.title('Which Bollywood celebrity are you?')

uploaded_image = st.file_uploader('Choose an image')

if uploaded_image is not None:
    file_path = save_uploaded_image(uploaded_image)
    if file_path:
        display_image = Image.open(uploaded_image)

        with st.spinner('Processing...'):
            extracted_features = extract_features(file_path, model, detector)
            if extracted_features is not None:
                index_pos = recommend(extracted_features, feature_list)
                predicted_actor = " ".join(filenames[index_pos].split('\\')[1].split('_'))

                col1, col2 = st.columns(2)

                with col1:
                    st.header('Your uploaded image')
                    st.image(display_image, width=300)

                with col2:
                    st.header(f"Seems like {predicted_actor}")
                    st.image(filenames[index_pos], width=300)

                # Remove the uploaded image after prediction
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"An error occurred while deleting the image: {e}")
            else:
                st.error('No faces detected. Please upload a different image.')
    else:
        st.error('Failed to save the uploaded image. Please try again.')
