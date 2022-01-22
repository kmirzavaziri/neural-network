from utils import *
visualizer = Visualizer(2, 3)
x_train, y_train, x_test, y_test = helpers.dataset(200, 400)
visualizer.add(x_test, y_test, title='Real Classes')

# Iterate and predict over some different batch sizes
PARAMETERS_COUNT = 2
HIDDEN_LAYER_COUNT = 3
CLASSES_COUNT = 2
EPSILON = .01
R_LAMBDA = .01
for BATCH_SIZE in [10, 20, 50, 100, 200]:
    print(BATCH_SIZE)
    nn = NeuralNetwork(
        [PARAMETERS_COUNT, HIDDEN_LAYER_COUNT, CLASSES_COUNT], EPSILON, R_LAMBDA,
        batch_size=BATCH_SIZE
    )
    nn.train(x_train, y_train)
    y_pred = nn.predict(x_test)
    visualizer.add(
        x_test, y_pred,
        title=f'Batches of size {BATCH_SIZE} \nLoss {helpers.loss(y_test, nn.outputs_history)}'
    )

visualizer.show('3.png')
