import numpy as np


class DescentResult:
    def __init__(self,
                 rescaled_scalars, scaled_scalars,
                 func, method_name,
                 batch_size=None, scaler_name=None,
                 time=None, memory=None, grad_calls=None):
        self.rescaled_scalars = rescaled_scalars
        self.scaled_scalars = scaled_scalars
        self.func = func
        self.method_name = method_name

        self.batch_size = batch_size
        self.scaler_name = scaler_name

        self.time = time
        self.memory = memory
        self.grad_calls = grad_calls

    def __str__(self):
        np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

        return '%s(\n\t%s)' % (
            type(self).__name__,
            ',\n\t'.join('%s=%s' % item for item in vars(self).items())
        )
