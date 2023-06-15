import tensorflow as tf

x = tf.Variable(1.0)

def f(x):
  y = x**2
  return y


with tf.GradientTape() as tape:
  y = f(x)
dy_dx = tape.gradient(y, x)

print(dy_dx)

x1 = tf.Variable(initial_value=[1.0,2.0,3.0])
w = tf.Variable(initial_value=[1.0,2.0,1.0])
b = tf.Variable(initial_value=[0.0,0.,0.])
y_val = tf.constant([2.,2.,2.])
with tf.GradientTape() as tape:
  loss = tf.reduce_sum((y_val - x1*w-b)**2)

dl_dw = tape.gradient(loss, w)

print(dl_dw)
