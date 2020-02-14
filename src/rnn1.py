# a crude implementation of rnn based generator

# import stuff
import pandas as pd
import sys
import tensorflow as tf
import numpy as np
import os

# set df display to max
pd.set_option('max_column', None)


infile = 'datasets/splitted/BindingsFor1FBI_X_FuC_5.25_Part8_of_128.txt'
print([open(infile).readlines()[0]])
df = pd.read_csv(infile, sep= '\t', names=['idx','seq', 'slided_seq', 'binding','not_sure1', 'not_sure2`','coord_rule',
                                           'bindng_rule'])
# smaller dataset
df = df.iloc[:1000,]
print(df.head())
# sys.exit()

merged_seqs = ' '.join(df.slided_seq)
vocab = sorted(set(merged_seqs))
char2idx = {u:i for i,u in enumerate(vocab)}
idx2char = np.array(vocab)
text_as_int = np.array([char2idx[c] for c in merged_seqs])

seq_length = 11
examples_per_epoch = len(merged_seqs)//seq_length+1
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

# split train, val, test
train_size = int(0.7 * dataset_size)
val_size = int(0.15 * dataset_size)
test_size = int(0.15 * dataset_size)

train_dataset = dataset.take(train_size)
test_dataset = dataset.skip(train_size)
val_dataset = dataset.skip(val_size)
test_dataset = dataset.take(test_size)


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

checkpoint_dir = 'training_checkpoints'
checkpoint_prefix =  os.path.join(checkpoint_dir, 'ckpt_{epoch}')

print(checkpoint_prefix)
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath = checkpoint_prefix,
    save_weights_only = True,
    verbose = 1
)


nb_epoch = 2

history = model.fit(train_dataset, epochs = nb_epoch,
                    callbacks = [checkpoint_callback],
                    validation_data=val_dataset)
history_outfile = 'outfiles/history.txt'
history_contents =[]
for key in history.history:
    for i,val in enumerate(history.history[key]):
        history_content = [key, i+1, val]
        history_contents.append(history_content)
historydf = pd.DataFrame(history_contents, columns=['loss_cat', 'epoch', 'value'])
print(historydf)


