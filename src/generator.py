# import stuff
import pandas as pd
import sys
import tensorflow as tf
import numpy as np
import os
import json
import timeit
from datetime import datetime
# some params

model_tag = sys.argv[1]

model_params = json.load(open('outfiles/%s_model_params.json' % model_tag))
char2idx = json.load(open('outfiles/%s_char2idx.json' % model_tag))
idx2char = list(char2idx.keys())

def create_model(vocab_size, embedding_dim, rnn_units, batch_size):
    '''
    create model according to params
    :param vocab_size:
    :param embedding_dim:
    :param rnn_units:
    :param batch_size:
    :return:
    '''
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape = [batch_size,None]),
        tf.keras.layers.LSTM(rnn_units,
                             return_sequences = True,
                             stateful =True,
                             recurrent_initializer = 'glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ], name='Geminio (A spell in Harry Potter to create duplicates of objects)')
    return model


def generate_seq(model, seed):
    '''
    generate new seqs via the supplied model and seed
    :param model:
    :param seed:
    :return:
    '''
    iterations = 10**7
    input_vect = [char2idx[s] for s in seed]
    input_vect = tf.expand_dims(input_vect, 0)
    generated_seq = ''
    temperature = 1.
    model.reset_states()
    print('Casting the Geminio spel %s times' % iterations)
    for i in range(iterations):
        prediction = model(input_vect)
        prediction = tf.squeeze(prediction, 0)
        prediction = prediction / temperature
        predicted_char = tf.random.categorical(prediction, num_samples=1)[-1, 0].numpy()
        input_vect = tf.expand_dims([predicted_char], 0)
        generated_seq += idx2char[predicted_char]
        # print('Casting the geminio spell %s/%s' % (i, iterations))
        if i%10**3 == 0:
            outname = 'outfiles/%s_geminio.txt' % model_tag
            outfile = open(outname, 'w')
            outfile.write(generated_seq)
            print('Geminio! output is written to %s' % outname)
    print(model.summary())


# run stuff
checkpoint_dir = '%s_training_checkpoints' % model_tag
vocab_size = model_params['vocab_size']
embedding_dim = model_params['embedding_dim']
rnn_units = model_params['rnn_units']
model = create_model(vocab_size,embedding_dim,rnn_units, 1)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

# time it
start = datetime.now()
generate_seq(model, 'VICTR')
end = datetime.now()
print('start {} end {}'.format(start, end))
