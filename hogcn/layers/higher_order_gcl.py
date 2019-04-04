import numpy as np
import tensorflow as tf

def higher_order_gcl(feats, A_norm, num_filters, iterations):
    xavier_init = tf.glorot_uniform_initializer()
    weights = tf.Variable(
        xavier_init([feats.shape[1], num_filters]),
        name="weights"
    )
    bias = tf.Variable(
        tf.random_normal([num_filters]),
        name='bias'
    )
    feat_sparse_tensor = tf.SparseTensor(
        indices=np.array([feats.row, feats.col]).T,
        values=[float(f) for f in feats.data],
        dense_shape=feats.shape,
    )

    output = tf.nn.relu(
        tf.sparse.sparse_dense_matmul(feat_sparse_tensor, weights) + bias
    )

    prop_mat_tensor = tf.SparseTensor(
        indices=np.array([A_norm.row, A_norm.col]).T,
        values=[float(f) for f in A_norm.data],
        dense_shape=[output.shape[0], output.shape[0]],
    )

    for _ in range(iterations):
        output = tf.sparse.sparse_dense_matmul(
            prop_mat_tensor,
            output
        )

    return output