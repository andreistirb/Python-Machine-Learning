import tensorflow as tf

'''with tf.name_scope('hidden') as scope:
    a = tf.constant(4, name='alpha')
    W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0), name='weights')
    b = tf.Variable(tf.zeros([1]), name='biases')

k = tf.placeholder(tf.float32)

# Make a normal distribution with a shifting mean
mean_moving_normal = tf.random_normal(shape=[1000], mean=(5*k), stddev=1)
# Record that distribution into a histogram summary
tf.summary.histogram("normal/moving_mean", mean_moving_normal)

sess = tf.Session()
writer = tf.summary.FileWriter("tmp/histogram_example")

summaries = tf.summary.merge_all()

N = 400
for step in range(N):
    k_val = step/float(N)
    summ = sess.run(summaries, feed_dict={k: k_val})
    writer.add_summary(summ, global_step=step)'''


k = tf.placeholder(tf.float32)
mean_moving_normal = tf.random_normal(shape=[1000], mean=(5*k), stddev=1)
tf.summary.histogram("normal/moving_mean", mean_moving_normal)

variance_shrinking_normal = tf.random_normal(shape=[1000], mean=0, stddev=1-(k))
tf.summary.histogram("normal/shrinking_variance", variance_shrinking_normal)

normal_combined = tf.concat([mean_moving_normal, variance_shrinking_normal], 0)
tf.summary.histogram("normal/bimodal", normal_combined)

summaries = tf.summary.merge_all()

sess = tf.Session()
writer = tf.summary.FileWriter("tmp/histogram_example_2")

N = 400
for step in range(N):
    k_val = step/float(N)
    summ = sess.run(summaries, feed_dict={k: k_val})
    writer.add_summary(summ, global_step=step)

