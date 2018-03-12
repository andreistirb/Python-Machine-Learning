### Variables

import tensorflow as tf
import numpy as np

my_variable = tf.get_variable("my_variable", [1, 2, 3])

my_int_variable = tf.get_variable("my_int_varible", [1, 2, 3], dtype=tf.int32, initializer=tf.zeros_initializer)

other_variable = tf.get_variable("other_variable", dtype=tf.int32, initializer=tf.constant[23, 42])

my_local = tf.get_variable("my_local", shape=(), collections=[tf.GraphKeys.LOCAL_VARIABLES])

my_non_trainable = tf.get_variable("my_non_trainable", shape=(), trainable=False)

tf.add_to_collection("my_collection_name", my_local)

tf.get_collection("my_collection_name")

### Set variable on other device
#with tf.device("/device:GPU:1"):
#    v = tf.get_variable("v", [1])

session.run(tf.global_variables_initializer())

session.run(my_variable.initializer)

print(session.run(tf.report_uninitialized_variables()))

v = tf.get_variable("v", shape=(), initializer=tf.zeros_initializer())
w = tf.get_variable("w", initializer=v.initialized_value() + 1)
b = v + 1

assignement = v.assign_add(1)
tf.global_variables_initializer.run()
sess.run(assignement) # || assignement.op.run() || assignement.eval()

with tf.control_dependencies([assignement]):
    w = v.read_value()

def conv_relu(input, kernel_shape, bias_shape):
    # Create variable named "weights"
    weights = tf.get_variable("weights", kernel_shape, initializer=tf.random_normal_initializer())
    # Create variable named "biases"
    biases = tf.get_variable("biases", bias_shape, initializer=tf.constant_initializer(0.0))
    conv = tf.nn.conv2d(input, weights, strides[1, 1, 1, 1], padding='SAME')
    return tf.nn.relu(conv + biases)


# This fails: 
input1 = tf.random_normal([1, 10, 10, 32])
input2 = tf.random_normal([1, 20, 20, 32])
x = conv_relu(input1, kernel_shape=[5, 5, 32, 32], bias_shape=[32])
x = conv_relu(x, kernel_shape=[5, 5, 32, 32], bias_shape=[32])

# Change scope of variables
def my_image_filter(input_images):
    with tf.variable_scope("conv1"):
        relu1 = conv_relu(input_images, [5, 5, 32, 32], [32])
    with tf.variable_scope("conv2"):
        return conv_relu(relu1, [5, 5, 32, 32], [32])

with tf.variable_scope("model"):
    output1 = my_image_filter(input1)
with tf.variable_scope("model", reuse=True):
    output2 = my_image_filter(input2)

with tf.variable_scope("model") as scope:
    output1 = my_image_filter(input1)
    scope.reuse_variables()
    output2 = my_image_filter(input2)

with tf.variable_scope("model") as scope:
  output1 = my_image_filter(input1)
with tf.variable_scope(scope, reuse=True):
  output2 = my_image_filter(input2)