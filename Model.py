import numpy as np


class Model():
    """Encapsulate Lasagne model
    Note on concept
    ===============
    Variables are dicts which contain symbolic theano variables or lasagne
    layers. Method arguments are typically actual data such as arrays.
    """

    def __init__(self, net=None):
        self.net = net
        self._predict = None
        self._predict_stoch = None

    def predict(self, *inputs):
        """Forward pass"""
        if self._predict is None:
            self._predict = self.net.predict
        return self._predict(*inputs)

    def mc_samples(self, *inputs, **kwargs):
        """Stochastic forward passes to generate T MC samples"""
        T = kwargs.pop('T', 100)
        if kwargs:
            raise TypeError('Unexpected **kwargs: %r' % kwargs)
        # TODO: implement stochastic prediction
        if self._predict_stoch is None:
            self._predict_stoch = None
        n_samples = len(inputs[0])
        n_out = self.net.values()[-1].output_shape[1]
        mc_samples = np.zeros((n_samples, n_out, T))
        for t in range(T):
            mc_samples[:, :, t] = self._predict_stoch(*inputs)
        return mc_samples

    def get_output_layer(self):
        return self.net.values()[-1]
