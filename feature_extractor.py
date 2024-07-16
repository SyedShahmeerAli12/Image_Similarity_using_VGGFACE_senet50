import os
import pickle
from tensorflow.keras.preprocessing import image
actors = os.listdir("Data")
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import numpy as np
from tqdm import tqdm


# FileNames = []
# for actor in actors:
#     for file in os.listdir(os.path.join("Data" , actor)):
#         FileNames.append(os.path.join("Data" , actor , file))

# pickle.dump(FileNames , open("filenames.pkl" , 'wb'))
filenames  = pickle.load(open("filenames.pkl" , 'rb'))
print(filenames)
# model = VGGFace( model='senet50' ,include_top=False, input_shape=(224, 224, 3))


# def Feature_Extractor(file, model):
#     img = image.load_img(file, target_size=(224, 224))
#     img_array = image.img_to_array(img)
#     expanded_img = np.expand_dims(img_array, axis=0)
#     processed_img = preprocess_input(expanded_img , version=1) 
#     result = model.predict(processed_img)
#     return result

# featues = []

# # Example usage
# for file in tqdm(filenames):
#     result = Feature_Extractor(file, model)
#     result = result.reshape((-1,))
#     featues.append(result)

# pickle.dump(featues , open("my_features" , 'wb'))
# features = pickle.load(open("my_features" , 'rb'))
print(len(features))
