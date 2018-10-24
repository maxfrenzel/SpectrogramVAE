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

# KL divergence between posterior with autoregressive flow and prior
def kl_divergence(sigma, epsilon, z_K, param, batch_mean=True):
    # logprob of posterior
    log_q_z0 = -0.5 * tf.square(epsilon)

    # logprob of prior
    log_p_zK = 0.5 * tf.square(z_K)

    # Terms from each flow layer
    flow_loss = 0
    for l in range(param['iaf_flow_length'] + 1):
        # Make sure it can't take log(0) or log(neg)
        flow_loss -= tf.log(sigma[l] + 1e-10)

    kl_divs = tf.identity(log_q_z0 + flow_loss + log_p_zK)
    kl_divs_reduced = tf.reduce_sum(kl_divs, axis=1)

    if batch_mean:
        return tf.reduce_mean(kl_divs, axis=0), tf.reduce_mean(kl_divs_reduced)
    else:
        return kl_divs, kl_divs_reduced


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
        self.cells_hidden = param['cells_hidden']

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

            with tf.variable_scope("IAF"):

                var['iaf_flows'] = list()
                for l in range(self.param['iaf_flow_length']):

                    with tf.variable_scope('layer{}'.format(l)):

                        layer = dict()

                        # Hidden state
                        layer['W_flow'] = create_variable("W_flow",
                                                        shape=[self.conv_out_units, self.param['dim_latent']])
                        layer['b_flow'] = create_bias_variable("b_flow",
                                                             shape=[1, self.param['dim_latent']])

                        flow_variables = list()
                        # Flow parameters from hidden state (m and s parameters for IAF)
                        for j in range(self.param['dim_latent']):
                            with tf.variable_scope('flow_layer{}'.format(j)):

                                flow_layer = dict()

                                # Set correct dimensionality
                                units_to_hidden_iaf = self.param['dim_autoregressive_nl']

                                flow_layer['W_flow_params_nl'] = create_variable("W_flow_params_nl",
                                                                  shape=[self.param['dim_latent'] + j, units_to_hidden_iaf])
                                flow_layer['b_flow_params_nl'] = create_bias_variable("b_flow_params_nl",
                                                                       shape=[1, units_to_hidden_iaf])

                                flow_layer['W_flow_params'] = create_variable("W_flow_params",
                                                                                 shape=[units_to_hidden_iaf,
                                                                                        2])
                                flow_layer['b_flow_params'] = create_bias_variable("b_flow_params",
                                                                                      shape=[1, 2])

                                flow_variables.append(flow_layer)

                        layer['flow_vars'] = flow_variables

                        var['iaf_flows'].append(layer)


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

        # -----------------------------------
        # Encoder

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

        # -----------------------------------
        # Latent flow

        # Lists to store the latent variables and the flow parameters
        nf_z = [z0]
        nf_sigma = [encoder_std]

        # Do calculations for each flow layer
        for l in range(self.param['iaf_flow_length']):

            W_flow = self.variables['iaf_flows'][l]['W_flow']
            b_flow = self.variables['iaf_flows'][l]['b_flow']

            nf_hidden = self.activation_nf(tf.matmul(encoder_hidden, W_flow) + b_flow)

            # Autoregressive calculation
            m_list = self.param['dim_latent'] * [None]
            s_list = self.param['dim_latent'] * [None]

            for j, flow_vars in enumerate(self.variables['iaf_flows'][l]['flow_vars']):

                # Go through computation one variable at a time
                if j == 0:
                    hidden_autoregressive = nf_hidden
                else:
                    z_slice = tf.slice(nf_z[-1], [0, 0], [-1, j])
                    hidden_autoregressive = tf.concat(axis=1, values=[nf_hidden, z_slice])

                W_flow_params_nl = flow_vars['W_flow_params_nl']
                b_flow_params_nl = flow_vars['b_flow_params_nl']
                W_flow_params = flow_vars['W_flow_params']
                b_flow_params = flow_vars['b_flow_params']

                # Non-linearity at current autoregressive step
                nf_hidden_nl = self.activation_nf(tf.matmul(hidden_autoregressive,
                                                       W_flow_params_nl) + b_flow_params_nl)

                # Calculate parameters for normalizing flow as linear transform
                ms = tf.matmul(nf_hidden_nl, W_flow_params) + b_flow_params

                # Split into individual components
                # m_list[j], s_list[j] = tf.split_v(value=ms,
                #                    size_splits=[1,1],
                #                    split_dim=1)
                m_list[j], s_list[j] = tf.split(value=ms,
                                                num_or_size_splits=[1, 1],
                                                axis=1)

            # Concatenate autoregressively computed variables
            # Add offset to s to make sure it starts out positive
            # (could have also initialised the bias term to 1)
            # Guarantees that flow initially small
            m = tf.concat(axis=1, values=m_list)
            s = self.param['initial_s_offset'] + tf.concat(axis=1, values=s_list)

            # Calculate sigma ("update gate value") from s
            sigma = tf.nn.sigmoid(s)
            nf_sigma.append(sigma)

            # Perform normalizing flow
            z_current = tf.multiply(sigma, nf_z[-1]) + tf.multiply((1 - sigma), m)

            # Invert order of variables to alternate dependence of autoregressive structure
            z_current = tf.reverse(z_current, axis=[1], name='LatentZ%d' % (l + 1))

            # Add to list of latent variables
            nf_z.append(z_current)

        z = tf.identity(nf_z[-1], name="LatentZ")

        # -----------------------------------
        # Decoder

        # Fully connected
        decoder_hidden = tf.nn.dropout(self.activation(tf.matmul(z, self.variables['decoder_fc']['W_z'])
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
        return decoder_output, encoder_mu, encoder_logvar, encoder_std, epsilon, z, nf_sigma

    def loss(self,
             input_batch,
             name='vae',
             beta=1.0):

        with tf.name_scope(name):
            output, encoder_mu, encoder_logvar, encoder_std, epsilon, z, nf_sigma = self._create_network(input_batch)

            _, div = kl_divergence(nf_sigma, epsilon, z, self.param, batch_mean=False)
            loss_latent = tf.identity(div, name='LossLatent')
            print(loss_latent)

            # loss_latent = tf.identity(-0.5 * tf.reduce_sum(1 + encoder_logvar
            #                                                - tf.square(encoder_mu)
            #                                                - tf.square(encoder_std), 1), name='LossLatent')

            print(input_batch)
            loss_reconstruction = tf.identity(-tf.reduce_sum(input_batch * tf.log(1e-8 + output)
                                                             + (1 - input_batch) * tf.log(1e-8 + 1 - output),
                                                             [1,2]), name='LossReconstruction')

            # loss_reconstruction = tf.reduce_mean(tf.pow(input_batch - output, 2))

            loss = tf.reduce_mean(loss_reconstruction + beta*loss_latent, name='Loss')
            # loss = tf.reduce_mean(loss_reconstruction, name='Loss')

            tf.summary.scalar('loss', loss)
            tf.summary.scalar('loss_rec', tf.reduce_mean(loss_reconstruction))
            tf.summary.scalar('loss_kl', tf.reduce_mean(loss_latent))
            tf.summary.scalar('beta', beta)

            return loss

    def encode_and_reconstruct(self, input_batch):

        output, _, _, _, _, encoder_mu, _ = self._create_network(input_batch, encode=True)

        return encoder_mu, output

    def decode(self, input_batch):

        z = input_batch

        # Fully connected
        decoder_hidden = self.activation(tf.matmul(z, self.variables['decoder_fc']['W_z'])
                                                       + self.variables['decoder_fc']['b_z'])

        # Reshape
        decoder_hidden = tf.reshape(decoder_hidden, [-1, self.conv_out_shape[0], self.conv_out_shape[1],
                                                     self.param['conv_channels'][-1]])

        for l in range(self.layers_enc):

            pool_kernel = self.param['max_pooling'][-1 - l]
            decoder_hidden = two_d_deconv(decoder_hidden, self.variables['decoder_deconv'][l]['filter'],
                                          self.param['deconv_shape'][l], pool_kernel)
            if l < self.layers_enc - 1:
                decoder_hidden = self.activation_conv(decoder_hidden)

        decoder_output = tf.nn.sigmoid(decoder_hidden)

        return decoder_output