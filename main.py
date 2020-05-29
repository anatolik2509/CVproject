import functions
from tensorflow import keras

model = keras.models.load_model('model.hS')
text = functions.img_to_str(model, 'test.png')
print(text)
