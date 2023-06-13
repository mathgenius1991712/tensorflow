import tensorflow as tf
import matplotlib
from matplotlib import pyplot as plt

DATA_SIZE = 201
PLOT_HEIGHT = 9
PLOT_WIDTH = 6

BATCH_SIZE = 32
EPOCHES = 100
LEARNING_RATE = 0.01
losses = []



matplotlib.rcParams['figure.figsize']=[PLOT_HEIGHT,PLOT_WIDTH]

x = tf.linspace(-2,2, DATA_SIZE)
x = tf.cast(x, tf.float32)

def f(x):
  y = x**2 + 2*x - 5
  return y

y = f(x) + tf.random.normal(shape=[DATA_SIZE])

class Model(tf.Module):
  def __init__(self):
    rand_init = tf.random.uniform(shape=[3], minval = -5, maxval=5, seed=22)
    self.w_q = tf.Variable(rand_init[0])
    self.w_l = tf.Variable(rand_init[1])
    self.b = tf.Variable(rand_init[2])
  
  @tf.function
  def __call__(self, x):
    return self.w_q*(x**2) + self.w_l*x + self.b


quad_model = Model()

plt.figure(num=1)
plt.plot(x.numpy(), y.numpy(), '.', label='Data')
plt.plot(x.numpy(), f(x).numpy(), label='Ground Truth')
plt.plot(x.numpy(), quad_model(x).numpy(), label='Initial Model before training' )


data = tf.data.Dataset.from_tensor_slices((x, y))
data = data.shuffle(buffer_size=x.shape[0]).batch(BATCH_SIZE)

def mse_loss(y_pred, y):
  return tf.reduce_mean(tf.square(y_pred-y))

for epoch in range(EPOCHES):
  for x_batch, y_batch in data:
    with tf.GradientTape() as tape:
      loss_batch = mse_loss(quad_model(x_batch), y_batch)
    grads = tape.gradient(loss_batch, quad_model.variables)
    for grad, v in zip(grads, quad_model.variables):
      v.assign_sub(grad*LEARNING_RATE)
  loss = mse_loss(quad_model(x), y)
  losses.append(loss)
  if epoch % 10 == 0:
    print(f'Mean squared error for step {epoch}: {loss.numpy():0.3f}')

print("\n")
plt.figure(num=2)
plt.plot(range(EPOCHES), losses)
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error (MSE)")
plt.title('MSE loss vs training iterations')



plt.figure(num=1)
plt.plot(x.numpy(), quad_model(x).numpy(), label='Model after training' )
plt.legend()
plt.show()

