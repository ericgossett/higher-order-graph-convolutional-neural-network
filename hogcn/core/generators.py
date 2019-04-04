import h5py
import tensorflow as tf

class H5Generator:
    # Adapted from this answer:
    # https://stackoverflow.com/questions/48309631/tensorflow-tf-data-dataset-reading-large-hdf5-files
    # Also see here for more inspiration:
    # http://adventuresinmachinelearning.com/tensorflow-dataset-tutorial/
    def __init__(self, path, key):
        self.path = path
        self.key = key
    
    def __call__(self):
        with h5py.File(self.path, 'r') as hf:
            for dat in hf[self.key]:
                yield dat