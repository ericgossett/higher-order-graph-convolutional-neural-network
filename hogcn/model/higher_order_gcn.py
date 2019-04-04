import tensorflow as tf
import numpy as np
from tqdm import trange
from ..core import Model
from ..core import Trainer
from ..core import Logger
from ..layers import higher_order_gcl
from ..operations import get_data, get_normalized_adjacency_matrix


class HOGCN(Model):
    def __init__(self, config, sess):
        super(HOGCN, self).__init__(config)
        self.sess = sess
        self.model_constructor()
        self.saver = tf.train.Saver(
            max_to_keep=self.config['saver_max_to_keep']
        )
    
    def model_constructor(self):

        graph, feats, labels = get_data(self.config['h5_file'])
        A_norm = get_normalized_adjacency_matrix(graph)

        layers = self.config['layers']
        order = len(layers)

        gcl = higher_order_gcl(feats, A_norm, layers[0], 1)
        print('start', gcl.shape)
        for i in range(2, order + 1):
            '''
            layer_n = higher_order_GCL(feats, A_norm, layers[i-1], i)
            print('layer' + str(i) + ' :', layer_n.shape, layer_n.dtype)
            '''
            gcl = tf.concat(
                [gcl, higher_order_gcl(feats, A_norm, layers[i-1], i)],
                axis=1
            )
        self.logits = tf.layers.dense(gcl, labels.shape[1])
        
        self.cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=labels,
                logits=self.logits
            )
        )

        self.optimizer = tf.train.AdamOptimizer(
            self.config['learning_rate']
        ).minimize(
            self.cross_entropy,
            global_step=self.global_step
        )
        correct_prediction = tf.equal(
            tf.argmax(self.logits, 1),
            tf.argmax(labels, 1)
        )

        self.accuracy = tf.reduce_mean(
            tf.cast(correct_prediction, tf.float32)
        )

        self.error = tf.reduce_mean(
            tf.square(
                self.logits - labels
            )
        )


class HOGCNTrainer(Trainer):
    def __init__(self, sess, model, logger):
        super(HOGCNTrainer, self).__init__(sess, model, logger)

    def epoch(self):
        """The logic for a single epoch of training. Loops though the batches 
        and logs the average loss and accuracy.
        """
        batch_loop = trange(self.model.config['num_epochs'], desc="accuracy")
        losses = []
        accuracies = []
        for _ in batch_loop:
            loss, accuracy = self.batch_step()
            losses.append(loss)
            accuracies.append(accuracy)
            batch_loop.set_description('Test Accuracy: ' + str(round(accuracy, 3)))
            current_iteration = self.model.global_step.eval(self.sess)

            self.logger.log(
                'cost',
                np.mean(losses),
                current_iteration
            )
            self.logger.log(
                'accuracy',
                np.mean(accuracies),
                current_iteration
            )
            self.model.save(self.sess)

    def batch_step(self):
        """Calculates the loss and accuracy per batch."""
        _, loss, accuracy = self.sess.run([
                self.model.optimizer,
                self.model.cross_entropy,
                self.model.accuracy
            ]
        )
        return loss, accuracy

    def predict(self):
        prediction = self.sess.run(
            self.model.logits,
        )
        return prediction