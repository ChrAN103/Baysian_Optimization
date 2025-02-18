from model import CNN, load_data, train, test

model = CNN()
train(model, 6)
accuracy = test(model)

print("model", accuracy)