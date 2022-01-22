import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plot


class helpers:
    def sigmoid(x):
        return 1 / (1 + np.exp(-np.array(x)))

    def softmax(x):
        return np.exp(x) / np.sum(np.exp(x))

    def dataset(n, m):
        np.random.seed(0)
        x, y = datasets.make_moons(n + m, noise=0.20)
        return x[:n], y[:n], x[n:], y[n:]

    def dataset_3(n, m):
        np.random.seed(0)
        x, y = datasets.make_classification(n + m, 2, n_classes=3, n_redundant=0, n_clusters_per_class=1)
        return x[:n], y[:n], x[n:], y[n:]

    def loss(y, nn_outputs):
        return -sum([np.log(nn_outputs[n][y[n]]) for n in range(len(y))]) / len(y)


class Visualizer:
    def __init__(self, rows, cols):
        self.fig, self.axs = plot.subplots(rows, cols)
        self.axs = self.axs.flatten()
        self.counter = 0

        self.fig.set_size_inches(20, 10)
        for ax in self.axs:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
        plot.subplots_adjust(wspace=0, hspace=0.3)

    def add(self, x, y, *, title=''):
        self.axs[self.counter].scatter(x[:, 0], x[:, 1], s=40, c=y, cmap=plot.cm.Spectral)
        self.axs[self.counter].set_title(title)
        self.counter += 1

    def show(self, filename=None):
        if filename:
            self.fig.savefig(filename, dpi=300)
        plot.show()


class Neuron:
    def __init__(self, input_size):
        self.input_size = input_size
        self.w = list(np.random.randn(self.input_size))
        self.b = [0.0] * self.input_size

    def summation(self, signal):
        return sum([signal[i] * self.w[i] + self.b[i] for i in range(self.input_size)])


class ACTIVATION:
    TANH = 'tanh'
    SIGMOID = 'sigmoid'


class NeuralNetwork:
    def __init__(self, layer_sizes, epsilon, r_lambda, batch_size=None, annealing_degree=0, activation_function=ACTIVATION.TANH):
        self.layer_sizes = layer_sizes
        self.layers = []
        for i in range(1, len(self.layer_sizes)):
            self.layers.append([Neuron(self.layer_sizes[i - 1]) for j in range(self.layer_sizes[i])])

        self.epsilon = epsilon
        self.r_lambda = r_lambda
        self.batch_size = batch_size
        self.annealing_degree = annealing_degree
        self.activation_function = activation_function

    def compute(self, signal):
        self.signals_history = [signal]
        activation_functions = self.activation_functions()
        for i in range(len(self.layers)):
            signal = list(activation_functions[i]([neuron.summation(signal) for neuron in self.layers[i]]))
            self.signals_history.append(signal)
        return signal

    def predict(self, x):
        y_pred = []
        self.outputs_history = []
        for node in x:
            output = self.compute(node)
            self.outputs_history.append(output)
            y_pred.append(output.index(max(output)))
        return y_pred

    def train(self, x, y, max_iterations=1200, visualizer=None):
        if len(x) != len(y):
            raise Exception('x and y must have the same length.')
        batch_size = self.batch_size if self.batch_size is not None else len(x)
        for i in range(max_iterations + 1):
            outputs = []
            signals_histories = []
            batch_counter = 0
            for node in x:
                outputs.append(self.compute(node))
                signals_histories.append(self.signals_history)
                batch_counter += 1
                if batch_counter % batch_size == 0:
                    self.backpropagate(signals_histories, y[batch_counter-batch_size:batch_counter])
                    signals_histories = []
                    self.epsilon *= 1 - self.annealing_degree

            if visualizer and i % 300 == 0:
                y_pred = [output.index(max(output)) for output in outputs]
                visualizer.add(x, y_pred, title=f'Iteration {i}\nLoss {helpers.loss(y, outputs)}')

    def backpropagate(self, raw_sh, y):
        w = []
        b = []
        for j in range(len(self.layers)):
            w.append(np.array([neuron.w for neuron in self.layers[j]]).T)
            b.append(np.array([neuron.b for neuron in self.layers[j]]).T)

        signals_histories = [
            np.array([raw_sh[n][l] for n in range(len(raw_sh))])
            for l in range(len(self.layer_sizes))
        ]

        delta = self.deltas(signals_histories, w, y)
        for j in range(len(self.layers) - 1, -1, -1):
            w[j] += -self.epsilon * (signals_histories[j].T.dot(delta[j]) + self.r_lambda * w[j])
            b[j] += -self.epsilon * np.sum(delta[j], axis=0)
            for k in range(len(self.layers[j])):
                self.layers[j][k].w = w[j].T[k]
                self.layers[j][k].b = b[j].T[k]

    def activation_functions(self):
        if self.activation_function == ACTIVATION.TANH:
            af = np.tanh
        elif self.activation_function == ACTIVATION.SIGMOID:
            af = helpers.sigmoid

        return [af] * (len(self.layers) - 1) + [helpers.softmax]

    def deltas(self, signals_histories, w, y):
        deltas = [0] * len(self.layers)

        deltas[-1] = np.array(signals_histories[-1])
        deltas[-1][range(len(y)), y] -= 1

        for i in range(len(self.layers) - 2, -1, -1):
            if self.activation_function == ACTIVATION.TANH:
                deltas[i] = deltas[i + 1].dot(w[i + 1].T) * (1 - np.power(signals_histories[i + 1], 2))
            elif self.activation_function == ACTIVATION.SIGMOID:
                deltas[i] = deltas[i + 1].dot(w[i + 1].T) * (signals_histories[i + 1] * (1 - signals_histories[i + 1]))

        return deltas
