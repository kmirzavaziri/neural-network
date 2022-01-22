from utils import *
visualizer = Visualizer(1, 3)
x_train, y_train, x_test, y_test = helpers.dataset(200, 400)
visualizer.add(x_test, y_test, title='Real Classes')

# Iterate and predict over different activation functions
PARAMETERS_COUNT = 2
HIDDEN_LAYER_COUNT = 3
CLASSES_COUNT = 2
EPSILON = .01
R_LAMBDA = .01
for ACTIVATION_FUNCTION in [ACTIVATION.SIGMOID, ACTIVATION.TANH]:
    print(ACTIVATION_FUNCTION)
    nn = NeuralNetwork(
        [PARAMETERS_COUNT, HIDDEN_LAYER_COUNT, CLASSES_COUNT], EPSILON, R_LAMBDA,
        activation_function=ACTIVATION_FUNCTION
    )
    nn.train(x_train, y_train)
    y_pred = nn.predict(x_test)
    visualizer.add(
        x_test, y_pred,
        title=f'Activation Function {ACTIVATION_FUNCTION} \nLoss {helpers.loss(y_test, nn.outputs_history)}'
    )

visualizer.show('5.png')
