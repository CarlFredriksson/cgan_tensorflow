import numpy as np
import tensorflow as tf
import utils

def generator(Z, Y):
    with tf.variable_scope("Generator"):
        x = tf.concat([Z, Y], 1)
        x = tf.layers.dense(x, 128, activation="relu")
        x = tf.layers.dense(x, 784, activation="tanh")
        x = tf.reshape(x, [-1, 28, 28, 1])

    return x

def discriminator(X, Y, reuse=False):
    with tf.variable_scope("Discriminator", reuse=reuse):
        x = tf.layers.flatten(X)
        x = tf.concat([x, Y], 1)
        x = tf.layers.dense(x, 128, activation="relu")
        x = tf.layers.dense(x, 1, activation="sigmoid")

    return x

def create_cgan(X, Y, Z):
    G = generator(Z, Y)
    D_real = discriminator(X, Y)
    D_fake = discriminator(G, Y, reuse=True)

    return G, D_real, D_fake

# Define constants
NUM_EPOCHS = 100
BATCH_SIZE = 128
LEARNING_RATE = 0.0002
BETA1 = 0.5
NOISE_DIM = 100
NUM_DIGITS = 10
SAMPLE_SIZE = NUM_DIGITS**2

# Load mnist data
X_train, Y_train = utils.load_mnist_data()
utils.plot_sample(X_train[:SAMPLE_SIZE], "output/mnist_data.png")
X_train = utils.preprocess_images(X_train)
Y_train = utils.convert_to_one_hot(Y_train)
mini_batches = utils.random_mini_batches(X_train, Y_train, BATCH_SIZE)

# Create DCGAN
X = tf.placeholder(tf.float32, shape=(None, X_train.shape[1], X_train.shape[2], X_train.shape[3]))
Y = tf.placeholder(tf.float32, shape=(None, Y_train.shape[1]))
Z = tf.placeholder(tf.float32, [None, NOISE_DIM])
G, D_real, D_fake = create_cgan(X, Y, Z)

# Create training steps
G_loss_func, D_loss_func = utils.create_loss_funcs(D_real, D_fake)
G_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Generator")
D_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Discriminator")
G_train_step = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE, beta1=BETA1).minimize(G_loss_func, var_list=G_vars)
D_train_step = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE, beta1=BETA1).minimize(D_loss_func, var_list=D_vars)

# Start session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Training loop
    for epoch in range(NUM_EPOCHS):
        for (X_batch, Y_batch) in mini_batches:
            Z_batch = utils.generate_Z_batch((X_batch.shape[0], NOISE_DIM))

            # Compute losses
            G_loss, D_loss = sess.run([G_loss_func, D_loss_func], feed_dict={X: X_batch, Y: Y_batch, Z: Z_batch})
            print("Epoch [{0}/{1}] - G_loss: {2}, D_loss: {3}".format(epoch, NUM_EPOCHS - 1, G_loss, D_loss))

            # Run training steps
            _ = sess.run(G_train_step, feed_dict={Y: Y_batch, Z: Z_batch})
            _ = sess.run(D_train_step, feed_dict={X: X_batch, Y: Y_batch, Z: Z_batch})

        # Plot generated images
        Y_batch = np.ones((NUM_DIGITS, NUM_DIGITS))
        Y_batch = (Y_batch * np.arange(NUM_DIGITS)).T.reshape((SAMPLE_SIZE, )).astype(int)
        Y_batch = utils.convert_to_one_hot(Y_batch)
        Z_batch = utils.generate_Z_batch((SAMPLE_SIZE, NOISE_DIM))
        gen_imgs = sess.run(G, feed_dict={Y: Y_batch, Z: Z_batch})
        gen_imgs = utils.postprocess_images(gen_imgs)
        utils.plot_sample(gen_imgs, "output/cgan/cgan_gen_data_" + str(epoch) + ".png")
