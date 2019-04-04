import tensorflow as tf
from .validators import config_validator

class Trainer:
    """Base class for training a model.

    Args:
        sess (tf.Session): The current session.
        model (Model): The model to train.
        logger (Logger): The logger to write the summary to.
    """
    def __init__(self, sess, model, logger):
        self.sess = sess
        self.model = model
        self.logger = logger
        self.sess.run(
            tf.group(
                tf.local_variables_initializer(),
                tf.global_variables_initializer()
            )
        )

    def epoch(self):
        """Abstract method to define the operations for each epoch."""
        raise NotImplementedError

    def train(self):
        """Iterates over epochs preforming the training_step for each epoch.
        
        Args:
            features (np.array): The feature vectors of the training set.
            labels (np.array): The labels of the training set.
        """
        with tf.name_scope('train'):
            self.epoch()
            

    def test(self, features, labels):
        """Determines the loss and accuracy of the trained model on the 
        test set.
        
        Args:
            features (np.array): The feature vectors of the test set.
            labels (np.array): The labels of the test set.
        """
        self.sess.run(
            self.model.data_iter.initializer, 
            feed_dict={
                self.model.x: features,
                self.model.y: labels
            }
        )
        with tf.name_scope('test'):
            _, loss, accuracy = self.sess.run([
                    self.model.optimizer,
                    self.model.cross_entropy,
                    self.model.accuracy
                ]
            )

            print('test cost: ', loss)
            print('test accuracy: ', accuracy)
            current_iteration = self.model.global_step.eval(self.sess)
            self.logger.log(
                'cost',
                loss,
                current_iteration,
                training=False
            )
            self.logger.log(
                'accuracy',
                accuracy,
                current_iteration,
                training=False
            )
            self.model.save(self.sess)

    def predict(self, feature):
        """Abstract method to fetch a prediction.

        Args:
            feature (np.array): The feature vector to predict a label of.
        """
        raise NotImplementedError

        


