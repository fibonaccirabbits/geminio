# a crude implementation of rnn based generator

# import stuff
import pandas as pd
import sys
import tensorflow as tf
import numpy as np
import os
import json

# set df display to max
pd.set_option('max_column', None)


infile = sys.argv[1]
tag = infile.split('/')[-1].split('_')[0]
seqs = open(infile).read()[:100000]
vocab = sorted(set(seqs))
char2idx = {u:i for i,u in enumerate(vocab)}
idx2char = np.array(vocab)
text_as_int = np.array([char2idx[c] for c in seqs])

print(idx2char)

seq_length = 11
examples_per_epoch = len(seqs)//seq_length+1
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

# split input target
def split_input_target(seq):
    '''
    split input output, return input-output pairs
    :param seq:
    :return:
    '''
    input_seq = seq[:-1]
    target_seq = seq[1:]
    return input_seq, target_seq

dataset = sequences.map(split_input_target)
batch_size = 64
buffer_size = 1000
dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)

# get dataset size since there is no dim/shape attributes in tf.dataset
dataset_size = 0
for item in dataset:
    dataset_size += 1
print(dataset_size)

scaller = 1

# split train, val, test
train_size = int(0.7/scaller * dataset_size)
val_size = int(0.15/scaller * dataset_size)
test_size = int(0.15/scaller * dataset_size)

print('Trains batches {}, val batches {}, test batches {}'.format(train_size, val_size, test_size))

train_dataset = dataset.take(train_size)
test_dataset = dataset.skip(train_size)
val_dataset = test_dataset.skip(val_size)
test_dataset = test_dataset.take(test_size)


vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 1024

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
    ])
    return model

model = create_model(vocab_size, embedding_dim,rnn_units, batch_size)
model.summary()

# sanity checks
for input_example_batch, output_example_batch in dataset.take(1):
    example_batch_predictions = model(input_example_batch)
    print(example_batch_predictions.shape)
    sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
    print(sampled_indices)
    sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()
    print(''.join(idx2char[sampled_indices]))
    print(''.join(idx2char[input_example_batch[0].numpy()]))

def loss(labels, logits):
    '''
    loss function for the net output
    :param labels:
    :param logits:
    :return:
    '''
    loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits = True)
    return loss



example_loss = loss(output_example_batch, example_batch_predictions)
print(example_loss.numpy().mean())

model.compile(optimizer='adam', loss=loss)

checkpoint_dir = '%s_training_checkpoints' % tag
checkpoint_prefix =  os.path.join(checkpoint_dir, 'ckpt_{epoch}')

print(checkpoint_prefix)
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath = checkpoint_prefix,
    save_weights_only = True,
    verbose = 1,
    save_best_only=True
)



nb_epoch = 2

# catch and write history

history = model.fit(train_dataset, epochs = nb_epoch,
                    callbacks = [checkpoint_callback],
                    validation_data=val_dataset)
history_outfile = 'outfiles/%s_history.txt' % tag
history_contents =[]
for key in history.history:
    for i,val in enumerate(history.history[key]):
        history_content = [key, i+1, val]
        history_contents.append(history_content)
historydf = pd.DataFrame(history_contents, columns=['loss_cat', 'epoch', 'value'])
historydf.to_csv(history_outfile, index=False)

# write model params

model_params = {
'nb_epoch': nb_epoch,
'batch_size':batch_size,
'buffer_size':buffer_size,
'vocab_size':vocab_size,
'embedding_dim':embedding_dim,
'rnn_units':rnn_units
}

model_params_outname = 'outfiles/%s_model_params.json' % tag
char2idx_outname = 'outfiles/%s_char2idx.json' % tag
json.dump(model_params, open(model_params_outname, 'w'))
json.dump(char2idx, open(char2idx_outname, 'w'))

