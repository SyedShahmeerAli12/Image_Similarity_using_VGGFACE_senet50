import pickle
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import cv2
from mtcnn import MTCNN
from PIL import Image

features = np.array(pickle.load(open("my_features" , 'rb')))
filenames = pickle.load(open('filenames.pkl','rb'))
model = VGGFace( model='senet50' ,include_top=False, input_shape=(224, 224, 3) )    
detector = MTCNN()
sample_img = cv2.imread(r'C:\Users\Musa Computer\Downloads\neha.jpg')
results = detector.detect_faces(sample_img)

x,y,width,height = results[0]['box']

face = sample_img[y:y+height,x:x+width]



image = Image.fromarray(face)
image = np.array(image.resize((224,224))).astype('float32')
expanded_img = np.expand_dims(image ,axis=0)
preprocessed_img = preprocess_input(expanded_img)
result = model.predict(preprocessed_img).flatten()



similarity_score = []

for i in range(len(features)):
    similarity_score.append(cosine_similarity(result.reshape(1,-1) , features[i].reshape(1,-1))[0][0])


index_pos = sorted(list(enumerate(similarity_score)),reverse=True,key=lambda x:x[1])[0][0]
temp_img = cv2.imread(filenames[index_pos])
cv2.imshow("output" , temp_img)
cv2.waitKey(0)



