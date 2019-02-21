import tensorflow as tf
import numpy as np

x1_data=np.random.rand(100)
#print(x1_data)
x2_data=np.random.rand(100)
y_data=3*x1_data+19*x2_data+24
#
a=tf.Variable(1.0)
b=tf.Variable(2.1)
c=tf.Variable(3.2)
#
y=a*x1_data+b*x2_data+c
#
loss=tf.reduce_mean(tf.square(y-y_data))
optimiser=tf.train.GradientDescentOptimizer(0.5)
train=optimiser.minimize(loss)
# 
init_op=tf.global_variables_initializer()
#
with tf.Session() as sess:
    sess.run(init_op)
    eval=sess.run([train,a,b,c])
    for steps in range(100):
        eval2=sess.run([train])
        if steps%1==0:
#            print(steps,eval)
            print(steps,sess.run(a),sess.run(b),sess.run(c))
#            print("fdgdgsg")