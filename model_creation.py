import functions

model = functions.emnist_model()
functions.emnist_train(model, 'C:\\emnist\\')
model.save('model.hS')
