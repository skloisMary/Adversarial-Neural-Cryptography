import tensorflow as tf

def _conv1D(input, filter, stride, kernelSize, name, activation = tf.nn.sigmoid):
    with tf.variable_scope(name):
        return tf.layers.conv1d(inputs=input, filters=filter, strides=stride,
                                   kernel_size=kernelSize, padding='SAME', activation=activation, use_bias=False)

def _ConvNet(input, unitsLength):
    """
    构建网络架构
    """
    input = tf.convert_to_tensor(input, dtype=tf.float32)
    input = tf.reshape(input, shape=[-1, unitsLength, 1])
    # print(input.shape)
    with tf.name_scope('convlayers'):
        conv1 = _conv1D(input, 2, 1, [4], name='conv_1')
        conv2 = _conv1D(conv1, 4, 2, [2], name='conv_2')
        conv3 = _conv1D(conv2, 4, 1, [1], name='conv_3')
        output = _conv1D(conv3, 1, 1, [1], name='conv_4', activation=tf.nn.tanh)
        return output

def _build_Network(plain, key, plainTextLength, keyLength):
    unitsLength = plainTextLength + keyLength
    # 定义Alice网络
    with tf.variable_scope('Alice'):
        Alice_input = tf.concat([plain, key], axis=1)
        A_w = tf.Variable(tf.truncated_normal(shape=[unitsLength, unitsLength], mean=0, stddev=0.1))
        Alice_FC_layer = tf.nn.sigmoid(tf.matmul(Alice_input, A_w))
        Alice_output = _ConvNet(Alice_FC_layer, unitsLength)
        # print(Alice_output.shape)

    reshape_Alice_output = tf.reshape(Alice_output, shape=[-1, plainTextLength])

    # 定义Bob网络
    with tf.variable_scope('Bob'):
        Bob_input = tf.concat([reshape_Alice_output, key], axis=1)
        B_w = tf.Variable(tf.truncated_normal(shape=[unitsLength, unitsLength], mean=0, stddev=0.1))
        Bob_FC_layer = tf.nn.sigmoid(tf.matmul(Bob_input, B_w))
        Bob_output = _ConvNet(Bob_FC_layer, unitsLength)

    # 定义Eve 网络
    with tf.variable_scope('Eve'):
        E_w_1 = tf.Variable(tf.truncated_normal(shape=[plainTextLength, unitsLength], mean=0, stddev=0.1))
        E_FC_layer1 = tf.nn.sigmoid(tf.matmul(reshape_Alice_output, E_w_1))
        E_w_2 = tf.Variable(tf.truncated_normal(shape=[unitsLength, unitsLength], mean=0, stddev=0.1))
        E_FC_layer2 = tf.nn.sigmoid(tf.matmul(E_FC_layer1, E_w_2))
        Eve_output = _ConvNet(E_FC_layer2, unitsLength)
    # print('Alice:', Alice_output.shape, ' Bob:', Bob_output.shape, ' Eve:', Eve_output.shape)
    return Alice_output, Bob_output, Eve_output














