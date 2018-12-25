import numpy as np
import tensorflow as tf
from layers import conv, dropout, fc, lrn, max_pool
from base_model import BasePretrainedModel


class AlexNet(BasePretrainedModel):
    """Implementation of the AlexNet."""
    def __init__(self, x_tensor, y_tensor, keep_prob, num_classes, train_layers,
                 weights_path='/pretrained/bvlc_alexnet.npy'):
        """Create the graph of the AlexNet model.

        Args:
            x_tensor: Placeholder for the input tensor.
            keep_prob: Dropout probability.
            num_classes: Number of classes in the dataset.
            train_layers: List of names of the layer, that get trained from scratch
            weights_path: Complete path to the pretrained weight file
        """
        # Parse input arguments into class variables
        super(AlexNet, self).__init__()
        self.X = x_tensor
        self.y = y_tensor
        # self.X = tf.placeholder(tf.float32, [None, 227, 227, 3])
        self.NUM_CLASSES = num_classes
        self.KEEP_PROB = keep_prob
        self.TRAIN_LAYERS = train_layers
        self.WEIGHTS_PATH = weights_path

        # Call the create function to build the computational graph of AlexNet
        # with tf.variable_scope('') as scope:

        self._build_network()

        # define metrics
        correct_pred = tf.equal(tf.argmax(self.fc8, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

        self._create_loss()

    def _build_network(self):
        """Create the network graph. returns tensors of fc7 and fc8"""
        # 1st Layer: Conv (w ReLu) -> Lrn -> Pool
        conv1 = conv(self.X, 11, 11, 96, 4, 4, padding='VALID', name='conv1')
        norm1 = lrn(conv1, 2, 1e-05, 0.75, name='norm1')
        pool1 = max_pool(norm1, 3, 3, 2, 2, padding='VALID', name='pool1')
        self.conv1, self.norm1, self.pool1 = conv1, norm1, pool1

        # 2nd Layer: Conv (w ReLu)  -> Lrn -> Pool with 2 groups
        conv2 = conv(pool1, 5, 5, 256, 1, 1, groups=2, name='conv2')
        norm2 = lrn(conv2, 2, 1e-05, 0.75, name='norm2')
        pool2 = max_pool(norm2, 3, 3, 2, 2, padding='VALID', name='pool2')
        self.conv2, self.norm2, self.pool1 = conv2, norm2, pool2

        # 3rd Layer: Conv (w ReLu)
        conv3 = conv(pool2, 3, 3, 384, 1, 1, name='conv3')
        self.conv3 = conv3

        # 4th Layer: Conv (w ReLu) split into two groups
        conv4 = conv(conv3, 3, 3, 384, 1, 1, groups=2, name='conv4')
        self.conv4 = conv4

        # 5th Layer: Conv (w ReLu) -> Pool split into two groups
        conv5 = conv(conv4, 3, 3, 256, 1, 1, groups=2, name='conv5')
        pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID', name='pool5')
        self.conv5, self.pool5 = conv5, pool5

        # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
        flattened = tf.reshape(pool5, [-1, 6 * 6 * 256])
        fc6 = fc(flattened, 6 * 6 * 256, 4096, name='fc6')
        dropout6 = dropout(fc6, self.KEEP_PROB, name='dropout6')
        self.flattened, self.fc6, self.dropout6 = flattened, fc6, dropout6

        # 7th Layer: FC (w ReLu) -> Dropout
        fc7 = fc(dropout6, 4096, 4096, name='fc7')
        dropout7 = dropout(fc7, self.KEEP_PROB, name='dropout7')
        self.fc7, self.dropout7 = fc7, dropout7

        # 8th Layer: FC and return unscaled activations
        fc8 = fc(dropout7, 4096, self.NUM_CLASSES, relu=False, name='fc8')
        self.fc8 = fc8

    def _create_loss(self):
        with tf.name_scope("cross_ent"):
            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.fc8, labels=self.y),
                name="loss",
            )

    def load_model_pretrained(self, session):
        """Load weights from file into network.

        As the weights from http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
        come as a dict of lists (e.g. weights['conv1'] is a list) and not as
        dict of dicts (e.g. weights['conv1'] is a dict with keys 'weights' &
        'biases') we need a special load function
        """
        # Load the weights into memory
        variable_dict = np.load(self.WEIGHTS_PATH, encoding='bytes').item()  # type: dict
        # Loop over all layer names stored in the weights dict
        for op_name in variable_dict:  # type: str
            # Check if layer should be trained from scratch
            if op_name not in self.TRAIN_LAYERS:
                with tf.variable_scope(op_name, reuse=True):
                    # Assign weights/biases to their corresponding tf variable
                    for data in variable_dict[op_name]:
                        var_name = "biases" if len(data.shape) == 1 else "weights"
                        var = tf.get_variable(var_name, trainable=False)
                        try:
                            session.run(var.assign(data))
                        except Exception as e:
                            print(e)
                            print("Failed to assign value to", var.name)

    def get_model_vars(self, session):
        """returns a dict of variables in the model, with keys being layer names and values being list of np.arrays"""
        layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8']
        variable_dict = {layer: [] for layer in layers}
        for layer in variable_dict:
            with tf.variable_scope(layer, reuse=True):
                for var_name in ["weights", "biases"]:
                    var = tf.get_variable(var_name)
                    variable_dict[layer].append(session.run(var))
        return variable_dict

    def set_model_vars(self, variable_dict, session):
        """assign model variables with values from a dict passed"""
        for op_name in variable_dict:
            with tf.variable_scope(op_name, reuse=True):
                for data in variable_dict[op_name]:
                    var_name = 'biases' if len(data.shape) == 1 else "weights"
                    # in case set_model_vars() is called before load_model_pretrained(), set trainable
                    var = tf.get_variable(var_name, trainable=op_name in self.TRAIN_LAYERS)
                    session.run(var.assign(data))

    def save_model_vars(self, path: str, session):
        """save model var-value dict under passed path"""
        np.save(path, self.get_model_vars(session))

    def load_model_vars(self, path: str, session):
        """load model var-value from passed path"""
        variable_dict = np.load(path, encoding="bytes").item()  # type: dict
        self.set_model_vars(variable_dict, session)
