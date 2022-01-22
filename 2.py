from utils import *
visualizer = Visualizer(2, 4)
x_train, y_train, x_test, y_test = helpers.dataset(200, 400)
visualizer.add(x_test, y_test, title='Real Classes')

# Iterate and predict over some different hidden layer sizes
PARAMETERS_COUNT = 2
CLASSES_COUNT = 2
EPSILON = .01
R_LAMBDA = .01
for HIDDEN_LAYER_COUNT in [1, 2, 3, 4, 5, 20, 40]:
    print(HIDDEN_LAYER_COUNT)
    nn = NeuralNetwork([PARAMETERS_COUNT, HIDDEN_LAYER_COUNT, CLASSES_COUNT], EPSILON, R_LAMBDA)
    nn.train(x_train, y_train)
    y_pred = nn.predict(x_test)
    visualizer.add(
        x_test, y_pred,
        title=f'{HIDDEN_LAYER_COUNT} Hidden Neurons\nLoss {helpers.loss(y_test, nn.outputs_history)}'
    )

visualizer.show('2.png')
