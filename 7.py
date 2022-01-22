from utils import *
visualizer = Visualizer(2, 3)
x_train, y_train, x_test, y_test = helpers.dataset(200, 400)
visualizer.add(x_test, y_test, title='Real Classes')

# Iterate and predict over some different hidden layers
PARAMETERS_COUNT = 2
CLASSES_COUNT = 2
EPSILON = .01
R_LAMBDA = .01
for HIDDEN_LAYERS in [[5, 5, 5], [4, 5, 4], [3, 4, 5], [3, 3], [3]]:
    print(HIDDEN_LAYERS)
    nn = NeuralNetwork([PARAMETERS_COUNT, *HIDDEN_LAYERS, CLASSES_COUNT], EPSILON, R_LAMBDA)
    nn.train(x_train, y_train)
    y_pred = nn.predict(x_test)
    visualizer.add(
        x_test, y_pred,
        title=f'Hidden Layers {", ".join(map(str, HIDDEN_LAYERS))} \nLoss {helpers.loss(y_test, nn.outputs_history)}'
    )

visualizer.show('7.png')
