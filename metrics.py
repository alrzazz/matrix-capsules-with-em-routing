"""
License: Apache 2.0
Author: Ashley Gritzman
E-mail: ashley.gritzman@za.ibm.com
"""

import tensorflow as tf

def accuracy(logits, labels):
  """Compute accuracy
  
  Credit:
    Suofei Zhang's implementation on GitHub, "Matrix-Capsules-EM-
    Tensorflow"
    https://github.com/www0wwwjs1/Matrix-Capsules-EM-Tensorflow
  Args: 
    logits: shape (batch_size, num_classes)
    labels: shape (batch_size,) containing index of correct class 
  Returns:
    accuracy: 
  """
    
  with tf.compat.v1.variable_scope("accuracy") as scope:
    logits = tf.identity(logits, name="logits")
    labels = tf.identity(labels, name="labels")
    batch_size = int(logits.get_shape()[0])
    logits_idx = tf.cast(tf.argmax(input=logits, axis=1), dtype=tf.int32)
    logits_idx = tf.reshape(logits_idx, shape=(batch_size,))
    correct_preds = tf.equal(tf.cast(labels, dtype=tf.int32), logits_idx)
    accuracy = (tf.reduce_sum(input_tensor=tf.cast(correct_preds, tf.float32)) 
          / batch_size)
  return accuracy