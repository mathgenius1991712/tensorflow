import tensorflow as tf

A = tf.constant([1])
print(A)

B = tf.constant([1,2,3])
print(B)

C = tf.constant([[1,2,3],[4,5,6]])
print(C)

D = A + B
print(D)

E = tf.constant([[2,2]])
F = tf.matmul(E, C)
print("F",F)


