### Tensors

import tensorflow as tf

### Rank 0

mammal = tf.Variable("Elephant", tf.string)
ignition = tf.Variable(451, tf.int16)
floating = tf.Variable(3.14159265359, tf.float64)
its_complicated = tf.Variable(12.3 - 4.85j, tf.complex64)

### Rank 1

mystr = tf.Variable(["Hello"], tf.string)
cool_numbers = tf.Variable([3.14159, 2.71828], tf.float32)
first_primes = tf.Variable([2, 3, 5, 7, 11], tf.int32)
its_very_complicated = tf.Variable([12.3 - 4.85j, 7.5 - 6.23], tf.complex64)

### Rank 2
mymat = tf.Variable([[7], [11]], tf.int16)
myxor = tf.Variable([[False, True], [True, False]], tf.bool)
linear_squares = tf.Variable([[4], [9], [16], [25]], tf.int32)
squarish_squares = tf.Variable([ [4, 9], [16, 25] ], tf.int32)
rank_of_squares = tf.rank(squarish_squares)
mymatC = tf.Variable([[7],[11]], tf.int32)

### Higher ranks
my_image = tf.zeros([10, 299, 299, 4])

### Getting a tf.Tensor object's rank
r = tf.rank(my_image)


my_scalar = my_vector[2]
my_scalar = my_matrix[1, 2]
my_row_vector = my_matrix[2]
my_column_vector = my_matrix[:, 3]

### Reshape tensors

rank_three_tensor = tf.ones([3, 4, 5])
matrix = tf.reshape(rank_three_tensor, [6,10])
matrixB = tf.reshape(matrix, [3, -1])
matrixAlt = tf.reshape(matrixB, [4, 3, -1])

yet_another = tf.reshape(matrixAlt, [13, 2, -1])

### Cast tensor to another datatype
float_tensor = tf.cast(tf.constant([1, 2, 3]), dtype=tf.float32)

### Evaluating tensors
constant = tf.constant([1,2,3])
tensor = constant * constant
print (tensor.eval())

p = tf.placeholder(tf.float32)
t = p + 1.0
t.eval()
t.eval(feed_dict={p:2.0})

