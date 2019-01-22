import numpy as np
import tensorflow as tf
import utils

def generator(Z, Y, is_training):
    with tf.variable_scope("Generator"):
        x = tf.concat([Z, Y], 3)
        # x.shape: (?, 1, 1, 110)

        x = tf.layers.conv2d_transpose(x, 256, 7, strides=1, padding="valid", use_bias=False)
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.leaky_relu(x)
        # x.shape: (?, 7, 7, 256)

        x = tf.layers.conv2d_transpose(x, 128, 5, strides=2, padding="same", use_bias=False)
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.leaky_relu(x)
        # x.shape: (?, 14, 14, 128)

        x = tf.layers.conv2d_transpose(x, 1, 5, strides=2, padding="same")
        x = tf.nn.tanh(x)
        # x.shape: (?, 28, 28, 1)

    return x

def discriminator(X, Y_fill, is_training, reuse=False):
    with tf.variable_scope("Discriminator", reuse=reuse):
        x = tf.concat([X, Y_fill], 3)
        # x.shape: (?, 28, 28, 11)

        x = tf.layers.conv2d(x, 128, 5, strides=2, padding="same", use_bias=False)
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.leaky_relu(x)
        # x.shape: (?, 14, 14, 128)

        x = tf.layers.conv2d(x, 256, 5, strides=2, padding="same", use_bias=False)
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.leaky_relu(x)
        # x.shape: (?, 7, 7, 256)

        x = tf.layers.conv2d(x, 1, 7, strides=1, padding="valid")
        x = tf.nn.sigmoid(x)
        # x.shape: (?, 1, 1, 1)

    return x

def create_cdcgan(X, Y, Y_fill, Z, is_training):
    G = generator(Z, Y, is_training)
    D_real = discriminator(X, Y_fill, is_training)
    D_fake = discriminator(G, Y_fill, is_training, reuse=True)

    return G, D_real, D_fake

def preprocess_labels(Y):
    Y = utils.convert_to_one_hot(Y)
    Y = np.expand_dims(Y, axis=1)
    Y = np.expand_dims(Y, axis=1)
    return Y

# Define constants
NUM_EPOCHS = 20
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
Y_train = preprocess_labels(Y_train)
mini_batches = utils.random_mini_batches(X_train, Y_train, BATCH_SIZE)

# Create DCGAN
X = tf.placeholder(tf.float32, shape=(None, X_train.shape[1], X_train.shape[2], X_train.shape[3]))
Y = tf.placeholder(tf.float32, shape=(None, 1, 1, NUM_DIGITS))
Y_fill = tf.placeholder(tf.float32, shape=(None, X_train.shape[1], X_train.shape[2], NUM_DIGITS))
Z = tf.placeholder(tf.float32, shape=(None, 1, 1, NOISE_DIM))
is_training = tf.placeholder(tf.bool, shape=())
G, D_real, D_fake = create_cdcgan(X, Y, Y_fill, Z, is_training)

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
            Z_batch = utils.generate_Z_batch((X_batch.shape[0], 1, 1, NOISE_DIM))
            Y_fill_batch = Y_batch * np.ones((X_batch.shape[0], Y_fill.shape[1], Y_fill.shape[2], Y_fill.shape[3]))

            # Compute losses
            G_loss, D_loss = sess.run([G_loss_func, D_loss_func], feed_dict={X: X_batch, Y: Y_batch, Y_fill: Y_fill_batch, Z: Z_batch, is_training: True})
            print("Epoch [{0}/{1}] - G_loss: {2}, D_loss: {3}".format(epoch, NUM_EPOCHS - 1, G_loss, D_loss))

            # Run training steps
            _ = sess.run(G_train_step, feed_dict={Y: Y_batch, Y_fill: Y_fill_batch, Z: Z_batch, is_training: True})
            _ = sess.run(D_train_step, feed_dict={X: X_batch, Y: Y_batch, Y_fill: Y_fill_batch, Z: Z_batch, is_training: True})

        # Plot generated images
        Y_batch = np.ones((NUM_DIGITS, NUM_DIGITS))
        Y_batch = (Y_batch * np.arange(NUM_DIGITS)).T.reshape((SAMPLE_SIZE, )).astype(int)
        Y_batch = preprocess_labels(Y_batch)
        Y_fill_batch = Y_batch * np.ones((SAMPLE_SIZE, Y_fill.shape[1], Y_fill.shape[2], Y_fill.shape[3]))
        Z_batch = utils.generate_Z_batch((SAMPLE_SIZE, 1, 1, NOISE_DIM))
        gen_imgs = sess.run(G, feed_dict={Y: Y_batch, Y_fill: Y_fill_batch, Z: Z_batch, is_training: True})
        gen_imgs = utils.postprocess_images(gen_imgs)
        utils.plot_sample(gen_imgs, "output/cdcgan/cdcgan_gen_data_" + str(epoch) + ".png")
