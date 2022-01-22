from utils import *
visualizer = Visualizer(2, 3)
x, y, _, _ = helpers.dataset(200, 0)
visualizer.add(x, y, title='Real Classes')

# Initiate
PARAMETERS_COUNT = 2
HIDDEN_LAYER_COUNT = 3
CLASSES_COUNT = 2
EPSILON = .01
R_LAMBDA = .01
nn = NeuralNetwork([PARAMETERS_COUNT, HIDDEN_LAYER_COUNT, CLASSES_COUNT], EPSILON, R_LAMBDA)

# Train
nn.train(x, y, visualizer=visualizer)

visualizer.show('1.png')
