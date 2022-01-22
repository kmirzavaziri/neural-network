from utils import *
visualizer = Visualizer(2, 3)
x_train, y_train, x_test, y_test = helpers.dataset(200, 400)
visualizer.add(x_test, y_test, title='Real Classes')

# Iterate and predict over some different annealing degrees
PARAMETERS_COUNT = 2
HIDDEN_LAYER_COUNT = 3
CLASSES_COUNT = 2
EPSILON = .01
R_LAMBDA = .01
for ANNEALING_DEGREE in [.2, .1, .01, .001, 0]:
    print(ANNEALING_DEGREE)
    nn = NeuralNetwork(
        [PARAMETERS_COUNT, HIDDEN_LAYER_COUNT, CLASSES_COUNT], EPSILON, R_LAMBDA,
        annealing_degree=ANNEALING_DEGREE
    )
    nn.train(x_train, y_train)
    y_pred = nn.predict(x_test)
    visualizer.add(
        x_test, y_pred,
        title=f'Annealing degree {ANNEALING_DEGREE} \nLoss {helpers.loss(y_test, nn.outputs_history)}'
    )

visualizer.show('4.png')
