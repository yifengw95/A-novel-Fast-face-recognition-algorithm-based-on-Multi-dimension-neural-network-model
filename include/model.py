import tensorflow as tf


def model(num_classes):
    _IMAGE_SIZE = 32
    _IMAGE_CHANNELS = 3
    _NUM_CLASSES = num_classes

    with tf.name_scope('main_params'):
        x = tf.placeholder(tf.float32, shape=[None, _IMAGE_SIZE * _IMAGE_SIZE * _IMAGE_CHANNELS], name='Input')
        y = tf.placeholder(tf.float32, shape=[None, _NUM_CLASSES], name='Output')
        x_image = tf.reshape(x, [-1, _IMAGE_SIZE, _IMAGE_SIZE, _IMAGE_CHANNELS], name='images')

        global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')
        learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')

    with tf.variable_scope('conv1') as scope:
        conv = tf.layers.conv2d(
            inputs=x_image,
            filters=32,
            kernel_size=[3, 3],
            padding='SAME',
            activation=tf.nn.relu
        )
        conv = tf.layers.conv2d(
            inputs=conv,
            filters=64,
            kernel_size=[3, 3],
            padding='SAME',
            activation=tf.nn.relu
        )
        pool = tf.layers.max_pooling2d(conv, pool_size=[2, 2], strides=2, padding='SAME')
        drop = tf.layers.dropout(pool, rate=0.25, name=scope.name)

    with tf.variable_scope('conv2') as scope:
        conv = tf.layers.conv2d(
            inputs=drop,
            filters=128,
            kernel_size=[3, 3],
            padding='SAME',
            activation=tf.nn.relu
        )
        pool = tf.layers.max_pooling2d(conv, pool_size=[2, 2], strides=2, padding='SAME')
        conv = tf.layers.conv2d(
            inputs=pool,
            filters=128,
            kernel_size=[2, 2],
            padding='SAME',
            activation=tf.nn.relu
        )
        pool = tf.layers.max_pooling2d(conv, pool_size=[2, 2], strides=2, padding='SAME')
        drop = tf.layers.dropout(pool, rate=0.25, name=scope.name)

    with tf.variable_scope('fully_connected') as scope:
        flat = tf.reshape(drop, [-1, 4 * 4 * 128])

        fc = tf.layers.dense(inputs=flat, units=1500, activation=tf.nn.relu)
        drop = tf.layers.dropout(fc, rate=0.5)
        logits = tf.layers.dense(inputs=drop, units=_NUM_CLASSES)
        softmax = tf.nn.softmax(logits)
        # softmax = tf.layers.dense(inputs=drop, units=_NUM_CLASSES, activation=tf.nn.softmax, name=scope.name)

    with tf.variable_scope('temp_scope') as scope:
        var1 = tf.Variable(initial_value=0, trainable=False, name='var1')

    y_pred_cls = tf.argmax(softmax, axis=1)

    return x, y, logits, softmax, y_pred_cls, global_step, learning_rate


def lr(epoch, alpha_0=0.01, mode=0):
    learning_rate = alpha_0
    
    if mode == 0:
        if epoch == 50:
            learning_rate *= 0.5
        elif epoch == 40:
            learning_rate *= 0.5
        elif epoch == 30:
            learning_rate *= 0.5
        elif epoch == 20:
            learning_rate *= 0.5
        elif epoch == 10:
            learning_rate *= 0.75
        elif epoch == 60:
            learning_rate *= 0.1
        elif epoch == 70:
            learning_rate *= 0.1
    
    elif mode == 1:
        if epoch >= 60:
            learning_rate *= 0.5
        elif epoch >= 40:
            learning_rate *= 0.7
        elif epoch >= 30:
            learning_rate *= 0.8
        elif epoch >= 20:
            learning_rate *= 0.9

    elif mode == 2:
        if epoch == 80:
            learning_rate *= 0.1
        elif epoch == 40:
            learning_rate *= 0.5
        elif epoch == 30:
            learning_rate *= 0.1
        elif epoch == 20:
            learning_rate *= 0.5
        elif epoch == 10:
            learning_rate *= 0.5
    

    return learning_rate
