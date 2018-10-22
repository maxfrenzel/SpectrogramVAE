import tensorflow as tf


def create_variable(name, shape):
    '''Create a convolution filter variable with the specified name and shape,
    and initialize it using Xavier initialition.'''
    initializer = tf.contrib.layers.xavier_initializer_conv2d()
    variable = tf.Variable(initializer(shape=shape), name=name)
    return variable


def create_bias_variable(name, shape):
    '''Create a bias variable with the specified name and shape and initialize
    it to zero.'''
    initializer = tf.constant_initializer(value=0.001, dtype=tf.float32)
    return tf.Variable(initializer(shape=shape), name)


# def upsample(net, name, stride, mode='ZEROS'):
#     """
#     Imitate reverse operation of Max-Pooling by either placing original max values
#     into a fixed postion of upsampled cell:
#     [0.9] =>[[.9, 0],   (stride=2)
#            [ 0, 0]]
#     or copying the value into each cell:
#     [0.9] =>[[.9, .9],  (stride=2)
#            [ .9, .9]]
#     :param net: 4D input tensor with [batch_size, width, heights, channels] axis
#     :param stride:
#     :param mode: string 'ZEROS' or 'COPY' indicating which value to use for undefined cells
#     :return:  4D tensor of size [batch_size, width*stride, heights*stride, channels]
#     """
#     assert mode in ['COPY', 'ZEROS']
#     with tf.name_scope('Upsampling'):
#         net = _upsample_along_axis(net, 2, stride[1], mode=mode)
#         net = _upsample_along_axis(net, 1, stride[0], mode=mode)
#         return net


# def _upsample_along_axis(volume, axis, stride, mode='ZEROS'):
#     shape = volume.get_shape().as_list()

#     assert mode in ['COPY', 'ZEROS']
#     assert 0 <= axis < len(shape)

#     target_shape = shape[:]
#     target_shape[axis] *= stride

#     print(volume.dtype)
#     print(shape)

#     padding = tf.zeros(shape, dtype=volume.dtype) if mode == 'ZEROS' else volume
#     parts = [volume] + [padding for _ in range(stride - 1)]
#     volume = tf.concat(parts, min(axis+1, len(shape)-1))

#     volume = tf.reshape(volume, target_shape)
#     return volume

def upsample(value, name, factor=[2, 2]):
    size = [int(value.shape[1] * factor[0]), int(value.shape[2] * factor[1])]
    with tf.name_scope(name):
        out = tf.image.resize_bilinear(value, size=size, align_corners=None, name=None)
        return out


def upsample2(value, name, output_shape):
    size = [int(output_shape[1]), int(output_shape[2])]
    with tf.name_scope(name):
        out = tf.image.resize_bilinear(value, size=size, align_corners=None, name=None)
        return out


def two_d_conv(value, filter_, pool_kernel=[2, 2], name='two_d_conv'):
    out = tf.nn.conv2d(value, filter_, strides=[1, 1, 1, 1], padding='SAME')
    out = tf.contrib.layers.max_pool2d(out, pool_kernel)

    return out


def two_d_deconv(value, filter_, deconv_shape, pool_kernel=[2, 2], name='two_d_conv'):
    out = upsample2(value, 'unpool', deconv_shape)
    # print(out)
    out = tf.nn.conv2d_transpose(out, filter_, output_shape=deconv_shape, strides=[1, 1, 1, 1], padding='SAME')
    # print(out)

    return out


class VAEModel(object):

    def __init__(self,
                 param,
                 batch_size,
                 activation=tf.nn.elu,
                 activation_conv=tf.nn.elu,
                 activation_nf=tf.nn.elu,
                 encode=False):

        self.param = param
        self.batch_size = batch_size
        self.activation = activation
        self.activation_conv = activation_conv
        self.activation_nf = activation_nf
        self.encode = encode
        self.layers_enc = len(param['conv_channels'])
        self.layers_dec = self.layers_enc
        self.conv_out_shape = [7, 7]
        self.conv_out_units = self.conv_out_shape[0] * self.conv_out_shape[1] * param['conv_channels'][-1]
        self.cells_hidden = 512

        self.variables = self._create_variables()

    def _create_variables(self):
        '''This function creates all variables used by the network.
        This allows us to share them between multiple calls to the loss
        function and generation function.'''

        var = dict()

        with tf.variable_scope('VAE'):

            with tf.variable_scope("Encoder"):

                var['encoder_conv'] = list()
                with tf.variable_scope('conv_stack'):

                    for l in range(self.layers_enc):

                        with tf.variable_scope('layer{}'.format(l)):
                            current = dict()

                            if l == 0:
                                channels_in = 1
                            else:
                                channels_in = self.param['conv_channels'][l - 1]
                            channels_out = self.param['conv_channels'][l]

                            current['filter'] = create_variable("filter",
                                                                [3, 3, channels_in, channels_out])
                            #                             current['bias'] = create_bias_variable("bias",
                            #                                               [channels_out])
                            var['encoder_conv'].append(current)

                with tf.variable_scope('fully_connected'):

                    layer = dict()

                    layer['W_z0'] = create_variable("W_z0",
                                                    shape=[self.conv_out_units, self.cells_hidden])
                    layer['b_z0'] = create_bias_variable("b_z0",
                                                         shape=[1, self.cells_hidden])

                    layer['W_mu'] = create_variable("W_mu",
                                                    shape=[self.cells_hidden, self.param['dim_latent']])
                    layer['W_logvar'] = create_variable("W_logvar",
                                                        shape=[self.cells_hidden, self.param['dim_latent']])
                    layer['b_mu'] = create_bias_variable("b_mu",
                                                         shape=[1, self.param['dim_latent']])
                    layer['b_logvar'] = create_bias_variable("b_logvar",
                                                             shape=[1, self.param['dim_latent']])

                    var['encoder_fc'] = layer

            with tf.variable_scope("Decoder"):

                with tf.variable_scope('fully_connected'):
                    layer = dict()

                    layer['W_z'] = create_variable("W_z",
                                                   shape=[self.param['dim_latent'], self.conv_out_units])
                    layer['b_z'] = create_bias_variable("b_z",
                                                        shape=[1, self.conv_out_units])

                    var['decoder_fc'] = layer

                var['decoder_deconv'] = list()
                with tf.variable_scope('deconv_stack'):

                    for l in range(self.layers_enc):
                        with tf.variable_scope('layer{}'.format(l)):
                            current = dict()

                            channels_in = self.param['conv_channels'][-1 - l]
                            if l == self.layers_enc - 1:
                                channels_out = 1
                            else:
                                channels_out = self.param['conv_channels'][-l - 2]

                            current['filter'] = create_variable("filter",
                                                                [3, 3, channels_out, channels_in])
                            #                             current['bias'] = create_bias_variable("bias",
                            #                                                 [channels_out])
                            var['decoder_deconv'].append(current)

        return var

    def _create_network(self, input_batch, keep_prob=1.0, encode=False):

        # Do encoder calculation
        encoder_hidden = input_batch
        for l in range(self.layers_enc):
            # print(encoder_hidden)
            encoder_hidden = two_d_conv(encoder_hidden, self.variables['encoder_conv'][l]['filter'],
                                        self.param['max_pooling'][l])
            encoder_hidden = self.activation_conv(encoder_hidden)

        # print(encoder_hidden)

        encoder_hidden = tf.reshape(encoder_hidden, [-1, self.conv_out_units])

        # print(encoder_hidden)

        # Additional non-linearity between encoder hidden state and prediction of mu_0,sigma_0
        mu_logvar_hidden = tf.nn.dropout(self.activation(tf.matmul(encoder_hidden,
                                                                   self.variables['encoder_fc']['W_z0'])
                                                         + self.variables['encoder_fc']['b_z0']),
                                         keep_prob=keep_prob)

        # print(mu_logvar_hidden)

        encoder_mu = tf.add(tf.matmul(mu_logvar_hidden, self.variables['encoder_fc']['W_mu']),
                            self.variables['encoder_fc']['b_mu'], name='ZMu')
        encoder_logvar = tf.add(tf.matmul(mu_logvar_hidden, self.variables['encoder_fc']['W_logvar']),
                                self.variables['encoder_fc']['b_logvar'], name='ZLogVar')

        # print(encoder_mu)

        # Convert log variance into standard deviation
        encoder_std = tf.exp(0.5 * encoder_logvar)

        # Sample epsilon
        epsilon = tf.random_normal(tf.shape(encoder_std), name='epsilon')

        if encode:
            z0 = tf.identity(encoder_mu, name='LatentZ0')
        else:
            z0 = tf.identity(tf.add(encoder_mu, tf.multiply(encoder_std, epsilon),
                                    name='LatentZ0'))

        # print(z0)

        # Fully connected
        decoder_hidden = tf.nn.dropout(self.activation(tf.matmul(z0, self.variables['decoder_fc']['W_z'])
                                                       + self.variables['decoder_fc']['b_z']),
                                       keep_prob=keep_prob)

        # print(decoder_hidden)

        # Reshape
        decoder_hidden = tf.reshape(decoder_hidden, [-1, self.conv_out_shape[0], self.conv_out_shape[1],
                                                     self.param['conv_channels'][-1]])

        for l in range(self.layers_enc):
            # print(decoder_hidden)

            pool_kernel = self.param['max_pooling'][-1 - l]
            decoder_hidden = two_d_deconv(decoder_hidden, self.variables['decoder_deconv'][l]['filter'],
                                          self.param['deconv_shape'][l], pool_kernel)
            if l < self.layers_enc - 1:
                decoder_hidden = self.activation_conv(decoder_hidden)

        decoder_output = tf.nn.sigmoid(decoder_hidden)

        # print(decoder_output)

        # return decoder_output, encoder_hidden, encoder_logvar, encoder_std
        return decoder_output, encoder_mu, encoder_logvar, encoder_std

    def loss(self,
             input_batch,
             name='vae',
             beta=1.0):

        with tf.name_scope(name):
            output, encoder_mu, encoder_logvar, encoder_std = self._create_network(input_batch)

            # loss=tf.reduce_min(encoder_std)

            loss_latent = tf.identity(-0.5 * tf.reduce_sum(1 + encoder_logvar
                                                           - tf.square(encoder_mu)
                                                           - tf.square(encoder_std), 1), name='LossLatent')

            loss_reconstruction = tf.identity(-tf.reduce_sum(input_batch * tf.log(1e-8 + output)
                                                             + (1 - input_batch) * tf.log(1e-8 + 1 - output),
                                                             [1, 2]), name='LossReconstruction')

            # loss_reconstruction = tf.reduce_mean(tf.pow(input_batch - output, 2))

            loss = tf.reduce_mean(loss_reconstruction + beta*loss_latent, name='Loss')
            # loss = tf.reduce_mean(loss_reconstruction, name='Loss')

            tf.summary.scalar('loss', loss)
            tf.summary.scalar('loss_rec', tf.reduce_mean(loss_reconstruction))
            tf.summary.scalar('loss_kl', tf.reduce_mean(loss_latent))
            tf.summary.scalar('beta', beta)

            return loss

    def encode_and_reconstruct(self,
               input_batch):

        output, encoder_mu, _, _ = self._create_network(input_batch)

        return encoder_mu, output