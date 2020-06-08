import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


tf.compat.v1.disable_eager_execution()
tf = tf.compat.v1

h1_dim = 150
h2_dim = 300
dim = 100
batch_size = 256

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)



w1_g = tf.Variable(tf.truncated_normal([dim, h1_dim], stddev=0.1), name="w1_g", dtype=tf.float32)
b1_g = tf.Variable(tf.zeros([h1_dim]), name="b1_g", dtype=tf.float32)

w2_g = tf.Variable(tf.truncated_normal([h1_dim,h2_dim], stddev=0.1), name="w2_g", dtype=tf.float32)
b2_g = tf.Variable(tf.zeros([h2_dim]), name="b2_g", dtype=tf.float32)

w3_g = tf.Variable(tf.truncated_normal([h2_dim,28*28], stddev=0.1), name="w3_g", dtype=tf.float32)
b3_g = tf.Variable(tf.zeros([28*28]), name="b3_g", dtype=tf.float32)

gen_weights = [w1_g, b1_g, w2_g, b2_g, w3_g, b3_g]

w1_d = tf.Variable(tf.truncated_normal([28*28, h2_dim], stddev=0.1), name="w1_d", dtype=tf.float32)
b1_d = tf.Variable(tf.zeros([h2_dim]), name="b1_d", dtype=tf.float32)

w2_d = tf.Variable(tf.truncated_normal([h2_dim, h1_dim], stddev=0.1), name="w2_d", dtype=tf.float32)
b2_d = tf.Variable(tf.zeros([h1_dim]), name="b2_d", dtype=tf.float32)

w3_d = tf.Variable(tf.truncated_normal([h1_dim, 1], stddev=0.1), name="w3_d", dtype=tf.float32)
b3_d = tf.Variable(tf.zeros([1]), name="b3_d", dtype=tf.float32)

disc_weights = [w1_d, b1_d, w2_d, b2_d, w3_d, b3_d]

x = tf.placeholder(tf.float32, [batch_size, 28*28], name="x_data")
z_noise = tf.placeholder(tf.float32, [batch_size, dim], name="z_prior")


def generator_(z_noise):
    h1 = tf.nn.elu(tf.matmul(z_noise, w1_g) + b1_g)
    h2 = tf.nn.elu(tf.matmul(h1, w2_g) + b2_g)
    h3 = tf.matmul(h2, w3_g) + b3_g
    out_gen_ = tf.nn.tanh(h3)
    return out_gen_


def discriminator_(x, out_gen):
    x_all = tf.concat([x,out_gen], 0)
    h1 = tf.nn.elu(tf.matmul(x_all, w1_d) + b1_d)
    h2 = tf.nn.elu(tf.matmul(h1, w2_d) + b2_d)
    h3 = tf.matmul(h2, w3_d) + b3_d
    y_data = tf.nn.sigmoid(tf.slice(h3, [0, 0], [batch_size, -1], name=None))
    y_fake = tf.nn.sigmoid(tf.slice(h3, [batch_size, 0], [-1, -1], name=None))
    return y_data,y_fake


#generate the output ops for generator and also define the weights.
out_gen = generator_(z_noise)
# Define the ops and weights for Discriminator
y_data, y_fake = discriminator_(x,out_gen)
#Cost function for Discriminator and Generator
gen_loss = -tf.reduce_mean(tf.log(y_fake + 1e-10))
disc_loss = -tf.reduce_mean(tf.log(y_data + 1e-10) + tf.log(1 - y_fake + 1e-10))

alpha = tf.constant(0.0001)
beta1 = tf.constant(0.5)
beta2 = tf.constant(0.9)
eps = tf.constant( 10 ** (-8))
t = tf.placeholder(dtype = tf.float32)


md, vd, yd = [], [], []
tr_md, tr_vd, tr_wd, tr_yd = [], [], [], []

for i in range(len(disc_weights)):
    d = tf.Variable(tf.truncated_normal(shape=disc_weights[i].get_shape(), mean = 0.0, stddev=0.1), dtype=tf.float32)
    md.append(d)
    vd.append(tf.Variable(tf.abs(d)))
    yd.append(tf.Variable(d, dtype=tf.float32))


for i in range(len(disc_weights)):
    g = tf.gradients(disc_loss, [disc_weights[i]])[0]
    g2 = g * g
    tmd = tf.assign(md[i], tf.add(tf.multiply(beta1,md[i]), tf.multiply((1 - beta1),g)))
    tr_md.append(tmd)
    tvd =  tf.assign(vd[i], tf.add(tf.multiply(beta2,vd[i]), tf.multiply((1 - beta2),g2)))
    tr_vd.append(tvd)
    m_ = tf.divide(md[i], (1 - (beta1 ** t)))
    v_ = tf.divide(vd[i], (1 - (beta2 ** t)))
    tyd = tf.assign(yd[i], tf.subtract(yd[i], alpha*tf.divide(m_,tf.sqrt(tf.abs(v_))+eps)))
    tr_yd.append(tyd)
    w = tf.assign(disc_weights[i], tf.subtract(yd[i], alpha*tf.divide(m_,tf.sqrt(tf.abs(v_))+eps)))
    tr_wd.append(w)


mg, vg, yg = [], [], []
tr_mg, tr_vg, tr_wg, tr_yg = [], [], [], []

for i in range(len(gen_weights)):
    g = tf.Variable(tf.truncated_normal(shape=gen_weights[i].get_shape(), mean = 0.0, stddev=0.1), dtype=tf.float32)
    mg.append(g)
    vg.append(tf.Variable(tf.abs(g)))
    yg.append(tf.Variable(g, dtype=tf.float32))


for i in range(len(gen_weights)):
    g = tf.gradients(gen_loss, [gen_weights[i]])[0]
    g2 = tf.abs(tf.multiply(g,g))
    tmg = tf.assign(mg[i], tf.add(tf.multiply(beta1,mg[i]), tf.multiply((1 - beta1),g)))
    tr_mg.append(tmg)
    tvg =  tf.assign(vg[i], tf.add(tf.multiply(beta2,vg[i]), tf.multiply((1 - beta2),g2)))
    tr_vg.append(tvg)
    m_ = tf.divide(mg[i], (1 - (beta1 ** t)))
    v_ = tf.divide(vg[i], (1 - (beta2 ** t)))
    tyg = tf.assign(yg[i], tf.subtract(yg[i], alpha * tf.divide(m_, tf.sqrt(tf.abs(v_)) + eps)))
    tr_yg.append(tyg)
    w = tf.assign(gen_weights[i], tf.subtract(yg[i], alpha*tf.divide(m_, tf.sqrt(tf.abs(v_))+eps)))
    tr_wg.append(w)


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

z_sample = np.random.uniform(-1, 1, size=(batch_size,dim)).astype(np.float32)

glf = []
dlf = []

T = 15000

for i in range(T):
    batch_x, _ = mnist.train.next_batch(batch_size)
    x_value = 2*batch_x.astype(np.float32) - 1
    z_value = np.random.uniform(-1, 1, size=(batch_size,dim)).astype(np.float32)
    for j in range(len(disc_weights)):
        sess.run(tr_md[j], feed_dict={x: x_value, z_noise: z_value, t: i+1})
        sess.run(tr_vd[j], feed_dict={x: x_value, z_noise: z_value, t: i+1})
        sess.run(tr_yd[j], feed_dict={x: x_value, z_noise: z_value, t: i + 1})
        sess.run(tr_wd[j], feed_dict={x: x_value, z_noise: z_value, t: i+1})

    for k in range(len(gen_weights)):
        sess.run(tr_mg[k], feed_dict={x: x_value, z_noise: z_value, t: i + 1})
        sess.run(tr_vg[k], feed_dict={x: x_value, z_noise: z_value, t: i + 1})
        sess.run(tr_yg[k], feed_dict={x: x_value, z_noise: z_value, t: i + 1})
        sess.run(tr_wg[k], feed_dict={x: x_value, z_noise: z_value, t: i + 1})


    [c1, c2] = sess.run([disc_loss, gen_loss], feed_dict={x: x_value, z_noise: z_value})
    dlf.append(c1)
    glf.append(c2)
    print('iter:', i)
    print(c1)
    print(c2)
    if i == 299 or i == 999 or i == 4999 or i == 7999 or i == 11999 or i == T - 1:
        out_val_img = sess.run(out_gen, feed_dict={z_noise: z_sample})
        imgs = 0.5 * (out_val_img + 1)
        for k in range(36):
            plt.subplot(6, 6, k + 1)
            image = np.reshape(imgs[k], (28, 28))
            plt.imshow(image, cmap='gray')
        plt.show()

