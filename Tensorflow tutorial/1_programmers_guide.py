### Introduction

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np

a = tf.constant(3.0, dtype=tf.float32)
b = tf.constant(4.0)
total = a+b
print(a)
print(b)
print(total)

writer = tf.summary.FileWriter('C:/Users/Andrei/Desktop/workspace/Python-Machine-Learning')
writer.add_graph(tf.get_default_graph())

sess = tf.Session()
print(sess.run({'ab': (a, b), 'total': total}))

vec = tf.random_uniform(shape=(3,))
out1 = vec + 1
out2 = vec + 2
print(sess.run(vec))
print(sess.run(vec))
print(sess.run((out1, out2)))

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
z = x + y

print(sess.run(z, feed_dict={x: 3, y: 5}))
print(sess.run(z, feed_dict={x: [1,3], y: [2,6]}))

my_data = [ [0, 1,], [2, 3,], [4, 5,], [6,7,],]
slices = tf.data.Dataset.from_tensor_slices(my_data)
next_item = slices.make_one_shot_iterator().get_next()

while True:
    try:
        print(sess.run(next_item))
    except tf.errors.OutOfRangeError:
        break

x = tf.placeholder(tf.float32, shape=[None, 3])
linear_model = tf.layers.Dense(units=1)
y = linear_model(x)
y1 = tf.layers.dense(x, units=1)


init = tf.global_variables_initializer()
sess.run(init)

print(sess.run(y, feed_dict={x: [[1,2,3],[4,5,6]]}))
print(sess.run(y1, feed_dict={x: [[1,2,3],[4,5,6]]}))

features = {
    'sales' : [[5], [10], [8], [9]],
    'department': ['sports', 'sports', 'gardening', 'gardening']
}

department_column = tf.feature_column.categorical_column_with_vocabulary_list(
    'department', ['sports', 'gardening']
)
department_column = tf.feature_column.indicator_column(department_column)

columns = [
    tf.feature_column.numeric_column('sales'),
    department_column
]

inputs = tf.feature_column.input_layer(features, columns)

var_init = tf.global_variables_initializer()
table_init = tf.tables_initializer()
sess = tf.Session()
sess.run((var_init, table_init))

print(sess.run(inputs))

### Training
x = tf.constant([[1], [2], [3], [4]], dtype=tf.float32)
y_true = tf.constant([[0], [-1], [-2], [-3]], dtype=tf.float32)

linear_model = tf.layers.Dense(units=1)
y_pred = linear_model(x)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

print(sess.run(y_pred))

loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)

print(sess.run(loss))

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

for i in range(100):
    _, loss_value = sess.run((train,loss))
    print(loss_value)

print(sess.run(y_pred))