import os
import tensorflow as tf
from .validators import config_validator

class Logger:
    """Logger for the model for Tensorboard visualization

    Args:
        sess (tf.Session): The current session.
        config (dict): The configuration.
    """
    def __init__(self, sess, config):
        self.summary_dir = config['summary_dir'] + '/' + config['name']
        self.sess = sess
        self.logged_items = {}
        self.train_summary_writer = tf.summary.FileWriter(
            os.path.join(self.summary_dir, 'training')
        )
        self.test_summary_writer = tf.summary.FileWriter(
            os.path.join(self.summary_dir, 'testing')
        )

    def log(self, name, value, step, is_image=False, training=True):
        """Creates a summary for each value defined in items_to_log

        Args:
            name (string): The name of the item to log.
            value (tensor): The value of the item to log.
            step (int): The current training step.
            is_image (bool): Set to true if the item being logged is an image.
            training (bool): Set false if logging for a test set.
        """
        summary_writer = (
            self.train_summary_writer if training else 
            self.test_summary_writer
        )

        logged_item_key = ('train' if training else 'test') + '-' + name
        if logged_item_key not in self.logged_items:
            placeholder = tf.placeholder(
                'float32',
                value.shape,
                name=logged_item_key
            )
            summary_op = tf.summary.scalar(
                name,
                placeholder
            )

            if is_image:
                placeholder = tf.placeholder(
                    'float32',
                    [None] + list(value.shape[1:]),
                    name=logged_item_key
                )
                summary_op = tf.summary.image(
                    name,
                    placeholder
                ) 

            self.logged_items[logged_item_key] = (
                summary_op,
                placeholder
            )

        summary = self.sess.run(
            self.logged_items[logged_item_key][0],
            { self.logged_items[logged_item_key][1]: value }
        )
        summary_writer.add_summary(summary, step)
        summary_writer.flush()

    def add_graph(self, graph):
        """Adds graph to the summary writer

        Args:
            graph (tf.Graph): The graph to add.  
        """
        self.train_summary_writer.add_graph(graph)
