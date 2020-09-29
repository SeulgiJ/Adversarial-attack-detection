import operator
import functools
import numpy as np
from keras.models import Model

from squeeze import get_squeezer_by_name
#
# Model = tf.keras.applications.MobileNetV2(include_top=True,
#                                                        weights='imagenet')

def reshape_2d(x):
    if len(x.shape) > 2:
        # Reshape to [#num_examples, ?]
        batch_size = x.shape[0]
        num_dim = functools.reduce(operator.mul, x.shape, 1)
        x = x.reshape((batch_size, num_dim/batch_size))
    return x

l1_dist = lambda x1,x2: np.sum(np.abs(x1 - x2), axis=tuple(range(len(x1.shape))[1:]))

class FeatureSqueezingDetector:
    def __init__(self, model, squeezers):
        self.model = model

        layer_id = len(model.layers)-1
        # squeezers_name = ['bit_depth_5', 'median_filter_2_2', 'non_local_means_color_11_3_4']
        squeezers_name = squeezers
        self.set_config(layer_id, squeezers_name)
        # todo: change threshold
        self.threshold = 1.2128
        self.train_fpr = 0.05

    def get_squeezer_by_name(self, name):
        return get_squeezer_by_name(name, 'python')

    def set_config(self, layer_id, squeezers_name):
        self.layer_id = layer_id
        self.squeezers_name = squeezers_name

    def get_config(self):
        return self.layer_id, self.squeezers_name

    def calculate_distance_max(self, val_orig, vals_squeezed):
        distance_func = l1_dist

        dist_array = []
        for val_squeezed in vals_squeezed:
            dist = distance_func(val_orig, val_squeezed)
            dist_array.append(dist)

        dist_array = np.array(dist_array)
        return np.max(dist_array, axis=0)

    def eval_layer_output(self, X, layer_id):
        layer_output = Model(inputs=self.model.layers[0].input, outputs=self.model.layers[layer_id].output)
        return layer_output.predict(X)


    def get_distance(self, X1, X2=None):
        layer_id, squeezers_name = self.get_config()

        input_to_normalized_output = lambda x: reshape_2d(self.eval_layer_output(x, layer_id))

        val_orig_norm = input_to_normalized_output(X1)

        if X2 is None:
            vals_squeezed = []
            for squeezer_name in squeezers_name:
                squeeze_func = self.get_squeezer_by_name(squeezer_name)
                val_squeezed_norm = input_to_normalized_output(squeeze_func(X1))
                vals_squeezed.append(val_squeezed_norm)
            distance = self.calculate_distance_max(val_orig_norm, vals_squeezed)
        else:
            val_1_norm = val_orig_norm
            val_2_norm = input_to_normalized_output(X2)
            distance_func = l1_dist
            distance = distance_func(val_1_norm, val_2_norm)

        return distance

    # Only examine the legitimate examples to get the threshold, ensure low False Positive rate.
    def train(self, X, Y):
        """
        Calculating distance depends on:
            layer_id
            normalizer
            distance metric
            feature squeezer(s)
        """

        if self.threshold is not None:
            print ("Loaded a pre-defined threshold value %f" % self.threshold)
        else:
            neg_idx = np.where(Y == 0)[0]
            X_neg = X[neg_idx]
            distances = self.get_distance(X_neg)

            selected_distance_idx = int(np.ceil(len(X_neg) * (1-self.train_fpr)))
            threshold = sorted(distances)[selected_distance_idx-1]
            self.threshold = threshold
            print ("Selected %f as the threshold value." % self.threshold)
        return self.threshold

    def test(self, X):
        distances = self.get_distance(X)
        threshold = self.threshold
        Y_pred = distances > threshold

        return Y_pred, distances