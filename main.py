import tensorflow as tf
import numpy as np
import h5py

from hogcn.core import Logger
from hogcn.model import HOGCN, HOGCNTrainer
from hogcn.viz import plot_confusion_matrix


def model_runner(config, model, trainer):

    # sets the process name. Useful for nohup & nvidia-smi
    from setproctitle import setproctitle
    setproctitle(config['name'])

    # Prevent TF from allocating the totality of GPU memory
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth=True

    # Instantiate the session, logger, model and trainer
    sess = tf.InteractiveSession(config=tf_config)
    logger = Logger(sess, config)
    _model = model(config, sess)
    _trainer = trainer(sess, _model, logger)

    # Visualize the graph in Tensorboard
    logger.add_graph(sess.graph)

    # Load the model if pre-trained
    _model.load(sess)

    # Train
    _trainer.train()

    # TODO: Clean this up so I don't have to submit the whole graph for predictions.
    with h5py.File(config['h5_file'], 'r') as hf:
        labels = hf['labels']
        
        predictions = _trainer.predict()
        actual = [np.argmax(l) for l in labels]
        pred = [np.argmax(p) for p in predictions]
        
        cora_classes = [
            'Case_Based',
            'Genetic_Algorithms',
            'Neural_Networks',
            'Probabilistic_Methods',
            'Reinforcement_Learning',
            'Rule_Learning',
            'Theory'
        ]

        citeseer_classes = [
            'Agents',
            'AI',
            'DB',
            'IR',
            'ML',
            'HCI',
        ]

        plot_confusion_matrix(actual, pred, citeseer_classes, normalize=False)

if __name__ == '__main__':

    config = {
        'name': 'citeseer',
        'num_epochs': 200,
        'layers': [64, 64, 64],
        'learning_rate': 1.0e-2,
        'summary_dir': 'logs',
        'save_dir': 'snapshots',
        'h5_file': 'citeseerData.h5',
        'saver_max_to_keep': 30,
    }

    model_runner(config, HOGCN, HOGCNTrainer)