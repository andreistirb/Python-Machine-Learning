### Graphs and Sessions

import tensorflow as tf
import numpy as np

c_0 = tf.constant(0, name="c") # operation named "c"
c_1 = tf.constant(2, name="c") # operation named "c_1"

print(c_0)
print(c_1)

with tf.name_scope("outer"):
    c_2 = tf.constant(2, name="c") # operation named "outer/c"
    print(c_2)

    with tf.name_scope("inner"):
        c_3 = tf.constant(3, name="c") # operation named "outer/inner/c"
        print(c_3)

    c_4 = tf.constant(4, name="c") # operation named "outer/c_1"
    print(c_4)

    with tf.name_scope("inner"):
        c_5 = tf.constant(5, name="c") # operation named "outer/inner_1/c"
        print(c_5)

'''weights = tf.random_normal(...)

with tf.device("/device:CPU:0"):
    img = tf.decode_jpeg(tf.read_file("img.jpg"))

with tf.device("/device:GPU:0"):
    result = tf.matmul(weights, img)

with tf.device("/job:ps/task:0"):
  weights_1 = tf.Variable(tf.truncated_normal([784, 100]))
  biases_1 = tf.Variable(tf.zeroes([100]))

with tf.device("/job:ps/task:1"):
  weights_2 = tf.Variable(tf.truncated_normal([100, 10]))
  biases_2 = tf.Variable(tf.zeroes([10]))

with tf.device("/job:worker"):
  layer_1 = tf.matmul(train_batch, weights_1) + biases_1
  layer_2 = tf.matmul(train_batch, weights_2) + biases_2

with tf.device(tf.train.replica_device_setter(ps_tasks=3)):
  # tf.Variable objects are, by default, placed on tasks in "/job:ps" in a
  # round-robin fashion.
  w_0 = tf.Variable(...)  # placed on "/job:ps/task:0"
  b_0 = tf.Variable(...)  # placed on "/job:ps/task:1"
  w_1 = tf.Variable(...)  # placed on "/job:ps/task:2"
  b_1 = tf.Variable(...)  # placed on "/job:ps/task:0"

  input_data = tf.placeholder(tf.float32)     # placed on "/job:worker"
  layer_0 = tf.matmul(input_data, w_0) + b_0  # placed on "/job:worker"
  layer_1 = tf.matmul(layer_0, w_1) + b_1     # placed on "/job:worker" '''


### Executing a graph in a tf.Session

x = tf.constant([[37.0, -23.0], [1.0, 4.0]])
w = tf.Variable(tf.random_uniform([2, 2]))
y = tf.matmul(x, w)
output = tf.nn.softmax(y)
init_op = w.initializer

# Create a default in-process session
with tf.Session() as sess:
    # Run the initializer on `w`
    sess.run(init_op)

    # Evaluate `output`. The return will be a numpy array containing the result of computation
    print("sess.run(output): ")
    print(sess.run(output))

    # Evaluate `y` and `output`. Note that `y` will only be computed once, and its result used both to return `y_val` and as 
    # an input to the `tf.nn.softmax()` op. Both `y_val` and `output_val` will be NumPy arrays.

    y_val, output_val = sess.run([y, output])
    print("y_val: ")
    print(y_val)

### With placeholders and feed_dict

x = tf.placeholder(tf.float32, shape=[3])
y = tf.square(x)

with tf.Session() as sess:
    print(sess.run(y, {x: [1.0, 2.0, 3.0]})) 
    print(sess.run(y, {x: [0.0, 0.0, 5.0]}))

    # sess.run(y)

    # sess.run(y, {x: 37.0})

### With options argument

y = tf.matmul([[37.0, -23.0], [1.0, 4.0]], tf.random_uniform([2, 2]))

with tf.Session() as sess:
    # Define options for the sess.run() call
    options = tf.RunOptions()
    options.output_partition_graphs = True
    options.trace_level = tf.RunOptions.FULL_TRACE

    # Define a container for the returned metadata
    metadata = tf.RunMetadata()

    sess.run(y, options=options, run_metadata=metadata)

    # Print the subgraphs that executed on each device
    # print(metadata.partition_graphs)

    # Print the timings of each operation that executed
    # print(metadata.step_stats)

### Visualizing the graph

# Build your graph
'''x = tf.constant([[37.0, -23.0], [1.0, 4.0]])
w = tf.Variable(tf.random_uniform([2,2]))
y = tf.matmul(x, w)

loss = ...
train_op = tf.train.AdagradOptimizer(0.01).minimize(loss)

with tf.Session() as sess:
    writer = tf.summary.FileWriter("/tmp/log/...", sess.graph)

    for i in range(1000):
        sess.run(train_op)

    writer.close() '''

### Working with more than one graph
g_1 = tf.Graph()
with g_1.as_default():
    c = tf.constant("Node in g_1")
    sess_1 = tf.Session()

g_2 = tf.Graph()
with g_2.as_default():
    d = tf.constant("node in g_2")

sess_2 = tf.Session(graph=g_2)

assert c.graph is g_1
assert sess_1.graph is g_1

assert d.graph is g_2
assert sess_2.graph is g_2

# Print all the operations in the default graph
g = tf.get_default_graph()
print(g.get_operations())

