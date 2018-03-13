### Saving and restoring models
import tensorflow as tf
import numpy as np

### Saving variables
'''v1 = tf.get_variable("v1", shape=[3], initializer=tf.zeros_initializer)
v2 = tf.get_variable("v2", shape=[5], initializer=tf.zeros_initializer)

inc_v1 = v1.assign(v1+1)
dec_v2 = v2.assign(v2-1)

init_op = tf.global_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init_op)
    inc_v1.op.run()
    dec_v2.op.run()
    save_path = saver.save(sess, "tmp/model.ckpt")
    print("Model saved in path: %s" % save_path) '''

### Restoring variables
tf.reset_default_graph()

# Create some variables
v1 = tf.get_variable("v1", shape=[3])
v2 = tf.get_variable("v2", shape=[5])

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, "tmp/model.ckpt")
    print("Model restored")
    print("v1: %s" % v1.eval())
    print("v2: %s" % v2.eval())

### Choosing which variables to save/restore

tf.reset_default_graph()

v1 = tf.get_variable("v1", [3], initializer=tf.zeros_initializer)
v2 = tf.get_variable("v2", [5], initializer=tf.zeros_initializer)

saver = tf.train.Saver({"v2": v2})

with tf.Session() as sess:
    v1.initializer.run()
    saver.restore(sess, "tmp/model.ckpt")

    print("v1: %s" % v1.eval())
    print("v2: %s" % v2.eval())

### Inspect variables in a checkpoint
from tensorflow.python.tools import inspect_checkpoint as chkp

chkp.print_tensors_in_checkpoint_file("tmp/model.ckpt", tensor_name='', all_tensors=True, all_tensor_names=True)
chkp.print_tensors_in_checkpoint_file("tmp/model.ckpt", tensor_name='v1', all_tensors=False, all_tensor_names=False)
chkp.print_tensors_in_checkpoint_file("tmp/model.ckpt", tensor_name="v2", all_tensors=False, all_tensor_names=False)