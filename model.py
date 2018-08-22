import numpy as np
np.random.seed(42)

from keras.layers import Dense, Input, Flatten, dot, concatenate, Reshape, Lambda, Concatenate, Multiply, Activation, Add
from keras.layers import Conv2D, MaxPooling2D, Embedding, GRU
from keras.layers import TimeDistributed
from keras.models import Model

from keras.activations import softmax
from keras import initializers
from keras import backend as K

import tensorflow as tf
tf.set_random_seed(42)


def softvaxaxis2(x):
    return softmax(x, axis=2)

def expand_tile(units, axis):
    """Expand and tile tensor along given axis
    Args:
        units: tf tensor with dimensions [batch_size, time_steps, n_input_features]
        axis: axis along which expand and tile. Must be 1 or 2
    """
    assert axis in (1, 2)
    n_time_steps = K.int_shape(units)[1]
    repetitions = [1, 1, 1, 1]
    repetitions[axis] = n_time_steps
    if axis == 1:
        expanded = Reshape(target_shape=( (1,) + K.int_shape(units)[1:] ))(units)
    else:
        expanded = Reshape(target_shape=(K.int_shape(units)[1:2] + (1,) + K.int_shape(units)[2:]))(units)
    return K.tile(expanded, repetitions)


def build_DUA_1(max_turn=10, maxlen=50, word_dim=200, sent_dim=200, session_hidden_size=50,
              num_words=50000, embedding_matrix=None):

    def additive_self_attention(units, n_hidden=None, n_output_features=None, activation=None):
        """ Computes additive self attention for time series of vectors (with batch dimension)
                the formula: score(h_i, h_j) = <v, tanh(W_1 h_i + W_2 h_j)>
                v is a learnable vector of n_hidden dimensionality,
                W_1 and W_2 are learnable [n_hidden, n_input_features] matrices
            Args:
                units: tf tensor with dimensionality [batch_size, time_steps, n_input_features]
                n_hidden: number of2784131 units in hidden representation of similarity measure
                n_output_features: number of features in output dense layer
                activation: activation at the output
            Returns:
                output: self attended tensor with dimensionality [batch_size, time_steps, n_output_features]
            """
        n_input_features = K.int_shape(units)[2]
        if n_hidden is None:
            n_hidden = n_input_features
        if n_output_features is None:
            n_output_features = n_input_features
        exp1 = Lambda(lambda x: expand_tile(x, axis=1))(units)
        exp2 = Lambda(lambda x: expand_tile(x, axis=2))(units)
        units_pairs = Concatenate(axis=3)([exp1, exp2])
        query = Dense(n_hidden, activation="tanh")(units_pairs)
        attention = Dense(1, activation=softvaxaxis2)(query)
        attended_units = Lambda(lambda x: K.sum(attention * x, axis=2))(exp1)
        # output = Dense(n_output_features, activation=activation)(attended_units)

        return attended_units

    def additive_attention(args):
        units  = args[0]
        hidden = args[1]
        """ Computes additive self attention for time series of vectors (with batch dimension)
                the formula: score(h_i, h_j) = <v, tanh(W_1 h_i + W_2 h_j)>
                v is a learnable vector of n_hidden dimensionality,
                W_1 and W_2 are learnable [n_hidden, n_input_features] matrices
            Args:
                units: tf tensor with dimensionality [batch_size, time_steps, n_input_features]
                n_hidden: number of2784131 units in hidden representation of similarity measure
                n_output_features: number of features in output dense layer
                activation: activation at the output
            Returns:
                output: self attended tensor with dimensionality [batch_size, time_steps, n_output_features]
            """
        # n_input_features = K.int_shape(units)[2]
        # if n_hidden is None:
        #     n_hidden = n_input_features
        # if n_output_features is None:
        #     n_output_features = n_input_features
        #
        # exp1 = Lambda(lambda x: expand_tile(x, axis=1))(units)
        # exp2 = Lambda(lambda x: expand_tile(x, axis=2))(units)
        # units_pairs = Concatenate(axis=3)([exp1, exp2])
        # query = Dense(n_hidden, activation="tanh")(units_pairs)
        # attention = Dense(1, activation=softvaxaxis2)(query)
        # attended_units = Lambda(lambda x: K.sum(attention * x, axis=2))(exp1)

        ############################


        W1_layer = Dense(50, input_shape=(max_turn, session_hidden_size))(units)
        W2_layer = Dense(50, input_shape=(max_turn, sent_dim))(hidden)
        v = Lambda(lambda x: K.tanh(x))(W1_layer + W2_layer)   # (?, 10, 50)

        final = Dense(1, input_shape=(max_turn, session_hidden_size))(v)  # (?, 10, 1)
        weight = Lambda(lambda x: K.exp(K.max(x, 2)))(final)
        weight2 = Lambda(lambda x: x / K.sum(x, axis=1))(weight)   # (?, 10)

        multiplication = Multiply()([units, Reshape((max_turn, 1))(weight2)])

        final2 = Lambda(lambda x: K.sum(x, axis=1) + 1e-6)(multiplication)  # (?, 50)
        return W1_layer

        #
        # # enc_output shape == (batch_size, max_length, hidden_size)
        #
        # # hidden shape == (batch_size, hidden size)
        # # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # # we are doing this to perform addition to calculate the score
        # hidden_with_time_axis = K.expand_dims(hidden, 1)
        #
        # # score shape == (batch_size, max_length, hidden_size)
        # score = K.tanh(W1(units) + W2(hidden_with_time_axis))

        # # attention_weights shape == (batch_size, max_length, 1)
        # # we get 1 at the last axis because we are applying score to self.V
        # attention_weights = K.softmax(V(score), axis=1)
        #
        # # context_vector shape after sum == (batch_size, hidden_size)
        # context_vector = K.sum(attention_weights * units, axis=1, keepdims=True)

        ###############

        # exp1 = Lambda(lambda x: expand_tile(x, axis=1))(units)
        # exp2 = Lambda(lambda x: expand_tile(x, axis=2))(units)
        # units_pairs = Concatenate(axis=3)([exp1, exp2])
        # query = Dense(n_hidden, activation="tanh")(units_pairs)
        # attention = Dense(1, activation=softvaxaxis2)(query)
        # attended_units = Lambda(lambda x: K.sum(attention * x, axis=2))(exp1)
        # output = Dense(n_output_features, activation=activation)(attended_units)

        # This is the final Attention and put the output to a classifier
        # W = theano.shared(ortho_weight(session_hidden_size), borrow=True)
        # W2 = theano.shared(glorot_uniform((hiddensize, session_hidden_size)), borrow=True)
        # b = theano.shared(value=np.zeros((session_hidden_size,), dtype='float32'), borrow=True)
        # U_s = theano.shared(glorot_uniform((session_hidden_size, 1)), borrow=True)
        #
        # final = T.dot(T.tanh(T.dot(res, W) + \
        #                      T.dot(T.stack(q_embedding_self_att_rnn, 1)[:, :, -1, :], W2) \
        #                      + b), U_s)
        # weight = T.exp(T.max(final, 2)) * sessionmask
        # weight2 = weight / T.sum(weight, 1)[:, None]
        # final2 = T.sum(res * weight2[:, :, None], 1) + 1e-6

        # return attended_units

    def match_layer_by_words(args):
        """ Utterance-Response Matching Layer """
        from keras.layers import concatenate, dot
        u = args[0]  # utterances Tensor
        r = args[1]  # response Tensor
        # concatenate all max_turn (10) M_1 word-word similarity matrices (50x50)
        return concatenate(
            [
                dot(
                    [Reshape((1, maxlen, word_dim))(u[:, turn]),  # (?, 50, 200) -> (?, 1, 50, 200)
                     Reshape((1, maxlen, word_dim))(r)            # (?, 50, 200) -> (?, 1, 50, 200)
                     ], axes=(-1, -1)  # dot product by the last axis (embeddings dimension)
                ) for turn in range(max_turn)
            ], axis=1
        )

    def match_layer_by_segments(args):
        """ Utterance-Response Matching Layer """
        from keras.layers import concatenate, dot
        u = args[0]  # utterances Tensor
        r = args[1]  # response Tensor
        # concatenate all max_turn (10) M_2 sequence-sequence similarity matrices (50x50)
        return concatenate(
            [
                dot(
                    # TODO: K.Dot(u)
                    [   Dense(sent_dim, use_bias=False)(
                            Reshape((1, maxlen, sent_dim))(u[:, turn])
                        ),
                        Reshape((1, maxlen, sent_dim))(r)            # (?, 50, 200) -> (?, 1, 50, 200)
                     ], axes=(-1, -1)  # dot product by the last axis (embeddings dimension)
                ) for turn in range(max_turn)
            ], axis=1
        )

    def fusion_utterance_layer(u):
        """ Fusion of each Sj with the last utterance St """
        from keras.layers import concatenate
        return concatenate(
            [
                concatenate(
                    [Reshape((1, maxlen, sent_dim))(u[:, turn]),
                     Reshape((1, maxlen, sent_dim))(u[:, -1])
                     ], axis=-1
                ) for turn in range(max_turn)
            ], axis=1
        )

    def fusion_response_layer(args):
        from keras.layers import concatenate
        u = args[0]  # utterances Tensor
        r = args[1]  # response Tensor
        return concatenate([r, u[:, -1]], axis=-1)

    def self_attention_utterance_layer(u):
        from keras.layers import concatenate
        return concatenate(
            [
                Reshape((1, maxlen, sent_dim*2))(additive_self_attention(u[:, turn], n_hidden=sent_dim * 2)) for turn in range(max_turn)
            ], axis=1
        )

    def concat_sa_utterance_layer(args):
        from keras.layers import concatenate
        u = args[0]   # utterances Tensor
        sa = args[1]  # self-attention of utterances Tensor
        return concatenate(
            [
                concatenate(
                    [Reshape((1, maxlen, sent_dim * 2))(u[:, turn]),
                     Reshape((1, maxlen, sent_dim * 2))(sa[:, turn])
                     ], axis=-1
                ) for turn in range(max_turn)
            ], axis=1
        )

    def concat_sa_response_layer(args):
        from keras.layers import concatenate
        r = args[0]   # response Tensor
        sa = args[1]  # self-attention of response Tensor
        return concatenate([r, sa], axis=-1)

    def tanh(x):
        return K.tanh(x)


    # Inputs
    context_input = Input(shape=(max_turn, maxlen), dtype='int32')   # (?, 10, 50)
    response_input = Input(shape=(maxlen,), dtype='int32')           # (?, 50)

    # 1. Utterance representations
    embedding_layer = Embedding(num_words,
                                word_dim,
                                weights=[embedding_matrix],
                                input_length=maxlen,
                                trainable=True)
    sentence2vec = GRU(sent_dim, return_sequences=True)   # GRU for encoding each sentence into a vector

    context_word_embedding = TimeDistributed(embedding_layer)(context_input)
    response_word_embedding = embedding_layer(response_input)

    context_sent_embedding = TimeDistributed(sentence2vec)(context_word_embedding)  # (?, 10, 50, 200)
    response_sent_embedding = sentence2vec(response_word_embedding)                 # (?, 50, 200)

    # 2. Turns-aware Aggregation
    fusion_utterance_context = Lambda(fusion_utterance_layer)(context_sent_embedding)  # (?, 10, 50, 400)
    fusion_response_context  = Lambda(fusion_response_layer)(
        [context_sent_embedding, response_sent_embedding]       # (?, 50, 400)
    )

    # 3. Matching Attention Flow
    sa_utterance = Lambda(self_attention_utterance_layer)(fusion_utterance_context)      # (?, 10, 50, 400)
    sa_response = additive_self_attention(fusion_response_context, n_hidden=sent_dim*2)  # (?, 50, 400)

    concat_sa_utterance = Lambda(concat_sa_utterance_layer)([fusion_utterance_context, sa_utterance])  # (?,10,50,800)
    concat_sa_response = Lambda(concat_sa_response_layer)([fusion_response_context, sa_response])      # (?, 50, 800)

    att2vec = GRU(sent_dim, return_sequences=True)  # GRU for concatenated fused vectors and self-attended vectors

    matching_att_utterance = TimeDistributed(att2vec)(concat_sa_utterance)
    matching_att_response = att2vec(concat_sa_response)

    # 4. Response Matching
    word_match = Lambda(match_layer_by_words)([context_word_embedding, response_word_embedding])
    sent_match = Lambda(match_layer_by_segments)([matching_att_utterance, matching_att_response])

    word_match = Reshape((max_turn, 1, maxlen, maxlen))(word_match)   # (?, 10, 50, 50) -> (?, 10, 1, 50, 50)
    sent_match = Reshape((max_turn, 1, maxlen, maxlen))(sent_match)   # (?, 10, 50, 50) -> (?, 10, 1, 50, 50)
    match_2ch = concatenate([word_match, sent_match], axis=2)   # (?, 10, 2, 50, 50) M_1 & M_2 as 2 channels

    conv = TimeDistributed(Conv2D(8, (3, 3), activation='relu', data_format='channels_first'))(match_2ch)
    pool = TimeDistributed(MaxPooling2D(pool_size=(3, 3), data_format='channels_first'))(conv)
    flat = TimeDistributed(Flatten())(pool)                      # (?, 10, 2048)
    flat = Dense(session_hidden_size, activation='tanh')(flat)   # (?, 10, 50)

    # 5. Attentive Turns Aggregation
    lastGRU = GRU(session_hidden_size, return_sequences=True)

    aggr = lastGRU(flat)  # aggr (?, 10, 50)

    # Final Attention
    #########################################
    W1_layer = Dense(50, input_shape=(max_turn, session_hidden_size))(aggr)
    W2_layer = Dense(50, input_shape=(max_turn, sent_dim))(Lambda(lambda x: x[:, :, -1, :])(matching_att_utterance))
    sum_ = Add()([W1_layer, W2_layer])
    v = Activation(tanh)(sum_)  # (?, 10, 50)

    final = Dense(1, input_shape=(max_turn, session_hidden_size))(v)  # (?, 10, 1)
    weight = Lambda(lambda x: K.exp(K.max(x, 2)))(final)
    weight2 = Lambda(lambda x: tf.divide(x, Reshape((1, ))(K.sum(x, axis=1))))(weight)  # (?, 10)

    multiplication = Multiply()([aggr, Reshape((max_turn, 1))(weight2)])

    Hm = Lambda(lambda x: K.sum(x, axis=1) + 1e-6)(multiplication)  # (?, 50)
    # Hm = Lambda(additive_attention)([aggr, matching_att_utterance[:, :, -1, :]])
    #########################################

    output = Dense(1, activation='sigmoid')(Hm)

    model = Model(inputs=[context_input, response_input], outputs=output)
    # adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

def build_DUA_2(max_turn=10, maxlen=50, word_dim=200, sent_dim=200, session_hidden_size=50,
              num_words=50000, embedding_matrix=None):
    def additive_self_attention(units, n_hidden=None, n_output_features=None, activation=None):
        """ Computes additive self attention for time series of vectors (with batch dimension)
                the formula: score(h_i, h_j) = <v, tanh(W_1 h_i + W_2 h_j)>
                v is a learnable vector of n_hidden dimensionality,
                W_1 and W_2 are learnable [n_hidden, n_input_features] matrices
            Args:
                units: tf tensor with dimensionality [batch_size, time_steps, n_input_features]
                n_hidden: number of2784131 units in hidden representation of similarity measure
                n_output_features: number of features in output dense layer
                activation: activation at the output
            Returns:
                output: self attended tensor with dimensionality [batch_size, time_steps, n_output_features]
            """
        n_input_features = K.int_shape(units)[2]
        if n_hidden is None:
            n_hidden = n_input_features
        if n_output_features is None:
            n_output_features = n_input_features
        exp1 = Lambda(lambda x: expand_tile(x, axis=1))(units)
        exp2 = Lambda(lambda x: expand_tile(x, axis=2))(units)
        units_pairs = Concatenate(axis=3)([exp1, exp2])
        query = Dense(n_hidden, activation="tanh", kernel_initializer=initializers.Orthogonal())(units_pairs)
        attention = Dense(1, activation=softvaxaxis2, kernel_initializer='glorot_uniform')(query)
        attended_units = Lambda(lambda x: K.sum(attention * x, axis=2))(exp1)
        # output = Dense(n_output_features, activation=activation)(attended_units)

        return attended_units

    def additive_attention(args):
        units  = args[0]
        hidden = args[1]
        """ Computes additive self attention for time series of vectors (with batch dimension)
                the formula: score(h_i, h_j) = <v, tanh(W_1 h_i + W_2 h_j)>
                v is a learnable vector of n_hidden dimensionality,
                W_1 and W_2 are learnable [n_hidden, n_input_features] matrices
            Args:
                units: tf tensor with dimensionality [batch_size, time_steps, n_input_features]
                n_hidden: number of2784131 units in hidden representation of similarity measure
                n_output_features: number of features in output dense layer
                activation: activation at the output
            Returns:
                output: self attended tensor with dimensionality [batch_size, time_steps, n_output_features]
            """
        # n_input_features = K.int_shape(units)[2]
        # if n_hidden is None:
        #     n_hidden = n_input_features
        # if n_output_features is None:
        #     n_output_features = n_input_features
        #
        # exp1 = Lambda(lambda x: expand_tile(x, axis=1))(units)
        # exp2 = Lambda(lambda x: expand_tile(x, axis=2))(units)
        # units_pairs = Concatenate(axis=3)([exp1, exp2])
        # query = Dense(n_hidden, activation="tanh")(units_pairs)
        # attention = Dense(1, activation=softvaxaxis2)(query)
        # attended_units = Lambda(lambda x: K.sum(attention * x, axis=2))(exp1)

        ############################


        W1_layer = Dense(50, input_shape=(max_turn, session_hidden_size), kernel_initializer=initializers.Orthogonal())(units)
        W2_layer = Dense(50, input_shape=(max_turn, sent_dim), kernel_initializer=initializers.Orthogonal())(hidden)
        v = Lambda(lambda x: K.tanh(x))(W1_layer + W2_layer)   # (?, 10, 50)

        final = Dense(1, input_shape=(max_turn, session_hidden_size), kernel_initializer='glorot_uniform')(v)  # (?, 10, 1)
        weight = Lambda(lambda x: K.exp(K.max(x, 2)))(final)
        weight2 = Lambda(lambda x: x / K.sum(x, axis=1))(weight)   # (?, 10)

        multiplication = Multiply()([units, Reshape((max_turn, 1))(weight2)])

        final2 = Lambda(lambda x: K.sum(x, axis=1) + 1e-6)(multiplication)  # (?, 50)
        return W1_layer

        #
        # # enc_output shape == (batch_size, max_length, hidden_size)
        #
        # # hidden shape == (batch_size, hidden size)
        # # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # # we are doing this to perform addition to calculate the score
        # hidden_with_time_axis = K.expand_dims(hidden, 1)
        #
        # # score shape == (batch_size, max_length, hidden_size)
        # score = K.tanh(W1(units) + W2(hidden_with_time_axis))

        # # attention_weights shape == (batch_size, max_length, 1)
        # # we get 1 at the last axis because we are applying score to self.V
        # attention_weights = K.softmax(V(score), axis=1)
        #
        # # context_vector shape after sum == (batch_size, hidden_size)
        # context_vector = K.sum(attention_weights * units, axis=1, keepdims=True)

        ###############

        # exp1 = Lambda(lambda x: expand_tile(x, axis=1))(units)
        # exp2 = Lambda(lambda x: expand_tile(x, axis=2))(units)
        # units_pairs = Concatenate(axis=3)([exp1, exp2])
        # query = Dense(n_hidden, activation="tanh")(units_pairs)
        # attention = Dense(1, activation=softvaxaxis2)(query)
        # attended_units = Lambda(lambda x: K.sum(attention * x, axis=2))(exp1)
        # output = Dense(n_output_features, activation=activation)(attended_units)

        # This is the final Attention and put the output to a classifier
        # W = theano.shared(ortho_weight(session_hidden_size), borrow=True)
        # W2 = theano.shared(glorot_uniform((hiddensize, session_hidden_size)), borrow=True)
        # b = theano.shared(value=np.zeros((session_hidden_size,), dtype='float32'), borrow=True)
        # U_s = theano.shared(glorot_uniform((session_hidden_size, 1)), borrow=True)
        #
        # final = T.dot(T.tanh(T.dot(res, W) + \
        #                      T.dot(T.stack(q_embedding_self_att_rnn, 1)[:, :, -1, :], W2) \
        #                      + b), U_s)
        # weight = T.exp(T.max(final, 2)) * sessionmask
        # weight2 = weight / T.sum(weight, 1)[:, None]
        # final2 = T.sum(res * weight2[:, :, None], 1) + 1e-6

        # return attended_units

    def match_layer_by_words(args):
        """ Utterance-Response Matching Layer """
        from keras.layers import concatenate, dot
        u = args[0]  # utterances Tensor
        r = args[1]  # response Tensor
        # concatenate all max_turn (10) M_1 word-word similarity matrices (50x50)
        return concatenate(
            [
                dot(
                    [Reshape((1, maxlen, word_dim))(u[:, turn]),  # (?, 50, 200) -> (?, 1, 50, 200)
                     Reshape((1, maxlen, word_dim))(r)            # (?, 50, 200) -> (?, 1, 50, 200)
                     ], axes=(-1, -1)  # dot product by the last axis (embeddings dimension)
                ) for turn in range(max_turn)
            ], axis=1
        )

    def match_layer_by_segments(args):
        """ Utterance-Response Matching Layer """
        from keras.layers import concatenate, dot
        u = args[0]  # utterances Tensor
        r = args[1]  # response Tensor
        # concatenate all max_turn (10) M_2 sequence-sequence similarity matrices (50x50)
        return concatenate(
            [
                dot(
                    # TODO: K.Dot(u)
                    [   Dense(sent_dim, use_bias=False, kernel_initializer=initializers.Orthogonal())(
                            Reshape((1, maxlen, sent_dim))(u[:, turn])
                        ),
                        Reshape((1, maxlen, sent_dim))(r)            # (?, 50, 200) -> (?, 1, 50, 200)
                     ], axes=(-1, -1)  # dot product by the last axis (embeddings dimension)
                ) for turn in range(max_turn)
            ], axis=1
        )

    def fusion_utterance_layer(u):
        """ Fusion of each Sj with the last utterance St """
        from keras.layers import concatenate
        return concatenate(
            [
                concatenate(
                    [Reshape((1, maxlen, sent_dim))(u[:, turn]),
                     Reshape((1, maxlen, sent_dim))(u[:, -1])
                     ], axis=-1
                ) for turn in range(max_turn)
            ], axis=1
        )

    def fusion_response_layer(args):
        from keras.layers import concatenate
        u = args[0]  # utterances Tensor
        r = args[1]  # response Tensor
        return concatenate([r, u[:, -1]], axis=-1)

    def self_attention_utterance_layer(u):
        from keras.layers import concatenate
        return concatenate(
            [
                Reshape((1, maxlen, sent_dim*2))(additive_self_attention(u[:, turn], n_hidden=sent_dim * 2)) for turn in range(max_turn)
            ], axis=1
        )

    def concat_sa_utterance_layer(args):
        from keras.layers import concatenate
        u = args[0]   # utterances Tensor
        sa = args[1]  # self-attention of utterances Tensor
        return concatenate(
            [
                concatenate(
                    [Reshape((1, maxlen, sent_dim * 2))(u[:, turn]),
                     Reshape((1, maxlen, sent_dim * 2))(sa[:, turn])
                     ], axis=-1
                ) for turn in range(max_turn)
            ], axis=1
        )

    def concat_sa_response_layer(args):
        from keras.layers import concatenate
        r = args[0]   # response Tensor
        sa = args[1]  # self-attention of response Tensor
        return concatenate([r, sa], axis=-1)

    def tanh(x):
        return K.tanh(x)


    # Inputs
    context_input = Input(shape=(max_turn, maxlen), dtype='int32')   # (?, 10, 50)
    response_input = Input(shape=(maxlen,), dtype='int32')           # (?, 50)

    # 1. Utterance representations
    embedding_layer = Embedding(num_words,
                                word_dim,
                                weights=[embedding_matrix],
                                input_length=maxlen,
                                trainable=True)
    sentence2vec = GRU(sent_dim, return_sequences=True, kernel_initializer=initializers.Orthogonal())   # GRU for encoding each sentence into a vector

    context_word_embedding = TimeDistributed(embedding_layer)(context_input)
    response_word_embedding = embedding_layer(response_input)

    context_sent_embedding = TimeDistributed(sentence2vec)(context_word_embedding)  # (?, 10, 50, 200)
    response_sent_embedding = sentence2vec(response_word_embedding)                 # (?, 50, 200)

    # 2. Turns-aware Aggregation
    fusion_utterance_context = Lambda(fusion_utterance_layer)(context_sent_embedding)  # (?, 10, 50, 400)
    fusion_response_context  = Lambda(fusion_response_layer)(
        [context_sent_embedding, response_sent_embedding]       # (?, 50, 400)
    )

    # 3. Matching Attention Flow
    sa_utterance = Lambda(self_attention_utterance_layer)(fusion_utterance_context)      # (?, 10, 50, 400)
    sa_response = additive_self_attention(fusion_response_context, n_hidden=sent_dim*2)  # (?, 50, 400)

    concat_sa_utterance = Lambda(concat_sa_utterance_layer)([fusion_utterance_context, sa_utterance])  # (?,10,50,800)
    concat_sa_response = Lambda(concat_sa_response_layer)([fusion_response_context, sa_response])      # (?, 50, 800)

    att2vec = GRU(sent_dim, return_sequences=True, kernel_initializer=initializers.Orthogonal())  # GRU for concatenated fused vectors and self-attended vectors

    matching_att_utterance = TimeDistributed(att2vec)(concat_sa_utterance)
    matching_att_response = att2vec(concat_sa_response)

    # 4. Response Matching
    word_match = Lambda(match_layer_by_words)([context_word_embedding, response_word_embedding])
    sent_match = Lambda(match_layer_by_segments)([matching_att_utterance, matching_att_response])

    word_match = Reshape((max_turn, 1, maxlen, maxlen))(word_match)   # (?, 10, 50, 50) -> (?, 10, 1, 50, 50)
    sent_match = Reshape((max_turn, 1, maxlen, maxlen))(sent_match)   # (?, 10, 50, 50) -> (?, 10, 1, 50, 50)
    match_2ch = concatenate([word_match, sent_match], axis=2)   # (?, 10, 2, 50, 50) M_1 & M_2 as 2 channels

    conv = TimeDistributed(Conv2D(8, (3, 3), activation='relu', data_format='channels_first'))(match_2ch)
    pool = TimeDistributed(MaxPooling2D(pool_size=(3, 3), data_format='channels_first'))(conv)
    flat = TimeDistributed(Flatten())(pool)                      # (?, 10, 2048)
    flat = Dense(session_hidden_size, activation='tanh', kernel_initializer='glorot_uniform')(flat)   # (?, 10, 50)

    # 5. Attentive Turns Aggregation
    lastGRU = GRU(session_hidden_size, return_sequences=True, kernel_initializer=initializers.Orthogonal())

    aggr = lastGRU(flat)  # aggr (?, 10, 50)

    # Final Attention
    #########################################
    W1_layer = Dense(50, input_shape=(max_turn, session_hidden_size), kernel_initializer=initializers.Orthogonal())(aggr)
    W2_layer = Dense(50, input_shape=(max_turn, sent_dim), kernel_initializer=initializers.Orthogonal())(Lambda(lambda x: x[:, :, -1, :])(matching_att_utterance))
    sum_ = Add()([W1_layer, W2_layer])
    v = Activation(tanh)(sum_)  # (?, 10, 50)

    final = Dense(1, input_shape=(max_turn, session_hidden_size), kernel_initializer='glorot_uniform')(v)  # (?, 10, 1)
    weight = Lambda(lambda x: K.exp(K.max(x, 2)))(final)
    weight2 = Lambda(lambda x: tf.divide(x, Reshape((1, ))(K.sum(x, axis=1))))(weight)  # (?, 10)

    multiplication = Multiply()([aggr, Reshape((max_turn, 1))(weight2)])

    Hm = Lambda(lambda x: K.sum(x, axis=1) + 1e-6)(multiplication)  # (?, 50)
    # Hm = Lambda(additive_attention)([aggr, matching_att_utterance[:, :, -1, :]])
    #########################################

    output = Dense(1, activation='sigmoid', kernel_initializer='glorot_uniform')(Hm)

    model = Model(inputs=[context_input, response_input], outputs=output)
    # adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

def build_DUA_3(max_turn=10, maxlen=50, word_dim=200, sent_dim=200, session_hidden_size=50,
              num_words=50000, embedding_matrix=None):
    """ Trainable embeddings, default initializators """
    def additive_self_attention(units, n_hidden=None, n_output_features=None, activation=None):
        """ Computes additive self attention for time series of vectors (with batch dimension)
                the formula: score(h_i, h_j) = <v, tanh(W_1 h_i + W_2 h_j)>
                v is a learnable vector of n_hidden dimensionality,
                W_1 and W_2 are learnable [n_hidden, n_input_features] matrices
            Args:
                units: tf tensor with dimensionality [batch_size, time_steps, n_input_features]
                n_hidden: number of2784131 units in hidden representation of similarity measure
                n_output_features: number of features in output dense layer
                activation: activation at the output
            Returns:
                output: self attended tensor with dimensionality [batch_size, time_steps, n_output_features]
            """
        n_input_features = K.int_shape(units)[2]
        if n_hidden is None:
            n_hidden = n_input_features
        if n_output_features is None:
            n_output_features = n_input_features
        exp1 = Lambda(lambda x: expand_tile(x, axis=1))(units)
        exp2 = Lambda(lambda x: expand_tile(x, axis=2))(units)
        units_pairs = Concatenate(axis=3)([exp1, exp2])
        query = Dense(n_hidden, activation="tanh", kernel_initializer=initializers.glorot_uniform())(units_pairs)
        attention = Dense(1, activation=softvaxaxis2, kernel_initializer=initializers.glorot_uniform())(query)
        attended_units = Lambda(lambda x: K.sum(attention * x, axis=2))(exp1)
        # output = Dense(n_output_features, activation=activation)(attended_units)

        return attended_units

    def match_layer_by_words(args):
        """ Utterance-Response Matching Layer """
        from keras.layers import concatenate, dot
        u = args[0]  # utterances Tensor
        r = args[1]  # response Tensor
        # concatenate all max_turn (10) M_1 word-word similarity matrices (50x50)
        return concatenate(
            [
                dot(
                    [Reshape((1, maxlen, word_dim))(u[:, turn]),  # (?, 50, 200) -> (?, 1, 50, 200)
                     Reshape((1, maxlen, word_dim))(r)            # (?, 50, 200) -> (?, 1, 50, 200)
                     ], axes=(-1, -1)  # dot product by the last axis (embeddings dimension)
                ) for turn in range(max_turn)
            ], axis=1
        )

    def match_layer_by_segments(args):
        """ Utterance-Response Matching Layer """
        from keras.layers import concatenate, dot
        u = args[0]  # utterances Tensor
        r = args[1]  # response Tensor
        # concatenate all max_turn (10) M_2 sequence-sequence similarity matrices (50x50)
        return concatenate(
            [
                dot(
                    # TODO: K.Dot(u)
                    [   Dense(sent_dim, use_bias=False, kernel_initializer=initializers.glorot_uniform())(
                            Reshape((1, maxlen, sent_dim))(u[:, turn])
                        ),
                        Reshape((1, maxlen, sent_dim))(r)            # (?, 50, 200) -> (?, 1, 50, 200)
                     ], axes=(-1, -1)  # dot product by the last axis (embeddings dimension)
                ) for turn in range(max_turn)
            ], axis=1
        )

    def fusion_utterance_layer(u):
        """ Fusion of each Sj with the last utterance St """
        from keras.layers import concatenate
        return concatenate(
            [
                concatenate(
                    [Reshape((1, maxlen, sent_dim))(u[:, turn]),
                     Reshape((1, maxlen, sent_dim))(u[:, -1])
                     ], axis=-1
                ) for turn in range(max_turn)
            ], axis=1
        )

    def fusion_response_layer(args):
        from keras.layers import concatenate
        u = args[0]  # utterances Tensor
        r = args[1]  # response Tensor
        return concatenate([r, u[:, -1]], axis=-1)

    def self_attention_utterance_layer(u):
        from keras.layers import concatenate
        return concatenate(
            [
                Reshape((1, maxlen, sent_dim*2))(additive_self_attention(u[:, turn], n_hidden=sent_dim * 2)) for turn in range(max_turn)
            ], axis=1
        )

    def concat_sa_utterance_layer(args):
        from keras.layers import concatenate
        u = args[0]   # utterances Tensor
        sa = args[1]  # self-attention of utterances Tensor
        return concatenate(
            [
                concatenate(
                    [Reshape((1, maxlen, sent_dim * 2))(u[:, turn]),
                     Reshape((1, maxlen, sent_dim * 2))(sa[:, turn])
                     ], axis=-1
                ) for turn in range(max_turn)
            ], axis=1
        )

    def concat_sa_response_layer(args):
        from keras.layers import concatenate
        r = args[0]   # response Tensor
        sa = args[1]  # self-attention of response Tensor
        return concatenate([r, sa], axis=-1)

    def tanh(x):
        return K.tanh(x)


    # Inputs
    context_input = Input(shape=(max_turn, maxlen), dtype='int32')   # (?, 10, 50)
    response_input = Input(shape=(maxlen,), dtype='int32')           # (?, 50)

    # 1. Utterance representations
    embedding_layer = Embedding(num_words,
                                word_dim,
                                weights=[embedding_matrix],
                                input_length=maxlen,
                                trainable=True)
    sentence2vec = GRU(sent_dim, return_sequences=True, kernel_initializer=initializers.orthogonal())   # GRU for encoding each sentence into a vector

    context_word_embedding = TimeDistributed(embedding_layer)(context_input)
    response_word_embedding = embedding_layer(response_input)

    context_sent_embedding = TimeDistributed(sentence2vec)(context_word_embedding)  # (?, 10, 50, 200)
    response_sent_embedding = sentence2vec(response_word_embedding)                 # (?, 50, 200)

    # 2. Turns-aware Aggregation
    fusion_utterance_context = Lambda(fusion_utterance_layer)(context_sent_embedding)  # (?, 10, 50, 400)
    fusion_response_context  = Lambda(fusion_response_layer)(
        [context_sent_embedding, response_sent_embedding]       # (?, 50, 400)
    )

    # 3. Matching Attention Flow
    sa_utterance = Lambda(self_attention_utterance_layer)(fusion_utterance_context)      # (?, 10, 50, 400)
    sa_response = additive_self_attention(fusion_response_context, n_hidden=sent_dim*2)  # (?, 50, 400)  # TODO: why sent_dim*2?

    concat_sa_utterance = Lambda(concat_sa_utterance_layer)([fusion_utterance_context, sa_utterance])  # (?,10,50,800)
    concat_sa_response = Lambda(concat_sa_response_layer)([fusion_response_context, sa_response])      # (?, 50, 800)

    att2vec = GRU(sent_dim, return_sequences=True, kernel_initializer=initializers.orthogonal())  # GRU for concatenated fused vectors and self-attended vectors

    matching_att_utterance = TimeDistributed(att2vec)(concat_sa_utterance)
    matching_att_response = att2vec(concat_sa_response)

    # 4. Response Matching
    word_match = Lambda(match_layer_by_words)([context_word_embedding, response_word_embedding])
    sent_match = Lambda(match_layer_by_segments)([matching_att_utterance, matching_att_response])

    word_match = Reshape((max_turn, 1, maxlen, maxlen))(word_match)   # (?, 10, 50, 50) -> (?, 10, 1, 50, 50)
    sent_match = Reshape((max_turn, 1, maxlen, maxlen))(sent_match)   # (?, 10, 50, 50) -> (?, 10, 1, 50, 50)
    match_2ch = concatenate([word_match, sent_match], axis=2)   # (?, 10, 2, 50, 50) M_1 & M_2 as 2 channels

    conv = TimeDistributed(Conv2D(8, (3, 3), activation='relu', data_format='channels_first', kernel_initializer=initializers.he_normal()))(match_2ch)
    pool = TimeDistributed(MaxPooling2D(pool_size=(3, 3), data_format='channels_first'))(conv)
    flat = TimeDistributed(Flatten())(pool)                      # (?, 10, 2048)
    flat = Dense(session_hidden_size, activation='tanh', kernel_initializer=initializers.glorot_uniform())(flat)   # (?, 10, 50)

    # 5. Attentive Turns Aggregation
    lastGRU = GRU(session_hidden_size, return_sequences=True, kernel_initializer=initializers.orthogonal())

    h = lastGRU(flat)  # h: (?, 10, 50)

    # Final Attention
    #########################################
    W1_layer = Dense(session_hidden_size, input_shape=(max_turn, session_hidden_size), kernel_initializer=initializers.glorot_uniform())(h)
    W2_layer = Dense(session_hidden_size, input_shape=(max_turn, sent_dim), kernel_initializer=initializers.glorot_uniform())(Lambda(lambda x: x[:, :, -1, :])(matching_att_utterance))
    sum_ = Add()([W1_layer, W2_layer])
    v = Activation(tanh)(sum_)  # (?, 10, 50)

    final = Dense(1, input_shape=(max_turn, session_hidden_size), kernel_initializer=initializers.glorot_uniform())(v)  # (?, 10, 1)
    weight = Lambda(lambda x: K.squeeze(K.exp(x), axis=-1))(final)
    weight2 = Lambda(lambda x: tf.divide(x, Reshape((1, ))(K.sum(x, axis=1))))(weight)  # (?, 10)

    multiplication = Multiply()([h, Reshape((max_turn, 1))(weight2)])

    L = Lambda(lambda x: K.sum(x, axis=1))(multiplication)  # (?, 50)
    # L = Lambda(additive_attention)([h, matching_att_utterance[:, :, -1, :]])
    #########################################

    output = Dense(1, activation='sigmoid', kernel_initializer=initializers.glorot_uniform())(L)

    model = Model(inputs=[context_input, response_input], outputs=output)
    # adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    return model
