import tensorflow as tf
import numpy as np

#create data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

# create tensorflow structure statt
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0)) #随机生成数据，作为weights,1维数组，从-1到1
biases = tf.Variable(tf.zeros([1]))

y = Weights * x_data + biases

loss = tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

state = tf.Variable(0)
one = tf.constant(1)
new_value = tf.add(state,one)
update = tf.assign(state,new_value)

init = tf.global_variables_initializer()#tf.initialize_all_variables()
# create tensorflow structure end

sess = tf.Session()
sess.run(init)
for _ in range(3):
    sess.run(update)
    print(sess.run(state))

for steps in range(301):
    sess.run(train)
    if steps % 20 == 0:
        print(steps, sess.run(Weights), sess.run(biases))

'''
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
'''

