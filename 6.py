from utils import *
visualizer = Visualizer(3, 3)
x_train, y_train, x_test, y_test = helpers.dataset_3(200, 400)
visualizer.add(x_train, y_train, title='Real Classes (Train)')

# Initiate
PARAMETERS_COUNT = 2
HIDDEN_LAYER_COUNT = 3
CLASSES_COUNT = 3
EPSILON = .01
R_LAMBDA = .01

nn = NeuralNetwork([PARAMETERS_COUNT, HIDDEN_LAYER_COUNT, CLASSES_COUNT], EPSILON, R_LAMBDA)

# Train
nn.train(x_train, y_train, max_iterations=1500, visualizer=visualizer)

visualizer.add(x_test, y_test, title='Real Classes (Test)')

# Predict
y_pred = nn.predict(x_test)
visualizer.add(
    x_test, y_pred,
    title=f'Prediction\nLoss {helpers.loss(y_test, nn.outputs_history)}'
)

visualizer.show('6.png')
