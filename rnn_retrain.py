import tensorflow as tf
import numpy as np
import my_txtutils as txt
from tensorflow.contrib import layers
from tensorflow.contrib import rnn  # rnn stuff temporarily in contrib, moving back to code in TF 1.1
import os
import time
import math
tf.set_random_seed(0)

# these must match what was saved !
ALPHASIZE = txt.ALPHASIZE
NLAYERS = 3
INTERNALSIZE = 512

##################################### REPLACE THESE WITH YOUR TRAINED MODEL CHECKPOINT FILES #############################
author = "twitter_checkpoints/rnn_train_1498231514-72000000" # This is the .index file in the checkpoint directory
META_GRAPH = 'twitter_checkpoints/rnn_train_1498231514-72000000.meta'
ncnt = 0


########################## REtrain #####################################
SEQLEN = 30
BATCHSIZE = 100
ALPHASIZE = txt.ALPHASIZE
INTERNALSIZE = 512 # Was 512
NLAYERS = 3 # Was 3
learning_rate = 0.001  # fixed learning rate
dropout_pkeep = 0.8    # some dropout
EPOCHS = 40

shakedir = "pickup/*.txt"
#shakedir = "../tensorflow/**/*.py"
codetext, valitext, bookranges = txt.read_data_files(shakedir, validation=True)


# display some stats on the data
epoch_size = len(codetext) // (BATCHSIZE * SEQLEN)
txt.print_data_stats(len(codetext), len(valitext), epoch_size)


################## RETRIANT MODEL ###################################3
lr = tf.placeholder(tf.float32, name='lr')  # learning rate
pkeep = tf.placeholder(tf.float32, name='pkeep')  # dropout parameter
batchsize = tf.placeholder(tf.int32, name='batchsize')

# inputs
X = tf.placeholder(tf.uint8, [None, None], name='X')    # [ BATCHSIZE, SEQLEN ]
Xo = tf.one_hot(X, ALPHASIZE, 1.0, 0.0)                 # [ BATCHSIZE, SEQLEN, ALPHASIZE ]
# expected outputs = same sequence shifted by 1 since we are trying to predict the next character
Y_ = tf.placeholder(tf.uint8, [None, None], name='Y_')  # [ BATCHSIZE, SEQLEN ]
Yo_ = tf.one_hot(Y_, ALPHASIZE, 1.0, 0.0)               # [ BATCHSIZE, SEQLEN, ALPHASIZE ]
# input state
Hin = tf.placeholder(tf.float32, [None, INTERNALSIZE*NLAYERS], name='Hin')  # [ BATCHSIZE, INTERNALSIZE * NLAYERS]
cells = [rnn.GRUCell(INTERNALSIZE) for _ in range(NLAYERS)]
dropcells = [rnn.DropoutWrapper(cell,input_keep_prob=pkeep) for cell in cells]
multicell = rnn.MultiRNNCell(dropcells, state_is_tuple=False)
multicell = rnn.DropoutWrapper(multicell, output_keep_prob=pkeep)  # dropout for the softmax layer
Yr, H = tf.nn.dynamic_rnn(multicell, Xo, dtype=tf.float32, initial_state=Hin)
H = tf.identity(H, name='H')  # just to give it a name
Yflat = tf.reshape(Yr, [-1, INTERNALSIZE])    # [ BATCHSIZE x SEQLEN, INTERNALSIZE ]
Ylogits = layers.linear(Yflat, ALPHASIZE)     # [ BATCHSIZE x SEQLEN, ALPHASIZE ]
Yflat_ = tf.reshape(Yo_, [-1, ALPHASIZE])     # [ BATCHSIZE x SEQLEN, ALPHASIZE ]
loss = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Yflat_)  # [ BATCHSIZE x SEQLEN ]
loss = tf.reshape(loss, [batchsize, -1])      # [ BATCHSIZE, SEQLEN ]
Yo = tf.nn.softmax(Ylogits, name='Yo')        # [ BATCHSIZE x SEQLEN, ALPHASIZE ]
Y = tf.argmax(Yo, 1)                          # [ BATCHSIZE x SEQLEN ]
Y = tf.reshape(Y, [batchsize, -1], name="Y")  # [ BATCHSIZE, SEQLEN ]
train_step = tf.train.AdamOptimizer(lr).minimize(loss)

# stats for display
seqloss = tf.reduce_mean(loss, 1)
batchloss = tf.reduce_mean(seqloss)
accuracy = tf.reduce_mean(tf.cast(tf.equal(Y_, tf.cast(Y, tf.uint8)), tf.float32))
loss_summary = tf.summary.scalar("batch_loss", batchloss)
acc_summary = tf.summary.scalar("batch_accuracy", accuracy)
summaries = tf.summary.merge([loss_summary, acc_summary])

# Init Tensorboard stuff. This will save Tensorboard information into a different
# folder at each run named 'log/<timestamp>/'. Two sets of data are saved so that
# you can compare training and validation curves visually in Tensorboard.
timestamp = str(math.trunc(time.time()))
summary_writer = tf.summary.FileWriter("twitter2_log/" + timestamp + "-training")
validation_writer = tf.summary.FileWriter("twitter2_log/" + timestamp + "-validation")

# Init for saving models. They will be saved into a directory named 'checkpoints'.
# Only the last checkpoint is kept.
if not os.path.exists("twitter2_checkpoints"):
    os.mkdir("twitter2_checkpoints")
saver = tf.train.Saver(max_to_keep=1000)

# for display: init the progress bar
DISPLAY_FREQ = 50
_50_BATCHES = DISPLAY_FREQ * BATCHSIZE * SEQLEN
progress = txt.Progress(DISPLAY_FREQ, size=111+2, msg="Training on next "+str(DISPLAY_FREQ)+" batches")

# init
istate = np.zeros([BATCHSIZE, INTERNALSIZE*NLAYERS])  # initial zero input state
init = tf.global_variables_initializer()

with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph(META_GRAPH)
    new_saver.restore(sess, author)
    step = 0

    for x, y_, epoch in txt.rnn_minibatch_sequencer(codetext, BATCHSIZE, SEQLEN, nb_epochs=EPOCHS):

        # train on one minibatch
        feed_dict = {X: x, Y_: y_, Hin: istate, lr: learning_rate, pkeep: dropout_pkeep, batchsize: BATCHSIZE}
        _, y, ostate = sess.run([train_step, Y, H], feed_dict=feed_dict)

        if epoch == 15 and learning_rate != 0.0001:
            learning_rate = 0.0001
            print("learning rate is ", learning_rate)
        if epoch == 30 and learning_rate != 0.00001:
            learning_rate = 0.00001
            print("learning rate is ", learning_rate)

        # log training data for Tensorboard display a mini-batch of sequences (every 50 batches)
        if step % _50_BATCHES == 0:
            feed_dict = {X: x, Y_: y_, Hin: istate, pkeep: 1.0, batchsize: BATCHSIZE}  # no dropout for validation
            y, l, bl, acc, smm = sess.run([Y, seqloss, batchloss, accuracy, summaries], feed_dict=feed_dict)
            txt.print_learning_learned_comparison(x, y, l, bookranges, bl, acc, epoch_size, step, epoch)
            summary_writer.add_summary(smm, step)

        # run a validation step every 50 batches
        # The validation text should be a single sequence but that's too slow (1s per 1024 chars!),
        # so we cut it up and batch the pieces (slightly inaccurate)
        # tested: validating with 5K sequences instead of 1K is only slightly more accurate, but a lot slower.
        if step % _50_BATCHES == 0 and len(valitext) > 0:
            VALI_SEQLEN = 1*1024  # Sequence length for validation. State will be wrong at the start of each sequence.
            bsize = len(valitext) // VALI_SEQLEN
            txt.print_validation_header(len(codetext), bookranges)
            vali_x, vali_y, _ = next(txt.rnn_minibatch_sequencer(valitext, bsize, VALI_SEQLEN, 1))  # all data in 1 batch
            vali_nullstate = np.zeros([bsize, INTERNALSIZE*NLAYERS])
            feed_dict = {X: vali_x, Y_: vali_y, Hin: vali_nullstate, pkeep: 1.0,  # no dropout for validation
                         batchsize: bsize}
            ls, acc, smm = sess.run([batchloss, accuracy, summaries], feed_dict=feed_dict)
            txt.print_validation_stats(ls, acc)
            # save validation data for Tensorboard
            validation_writer.add_summary(smm, step)

        # display a short text generated with the current weights and biases (every 150 batches)
        if step // 3 % _50_BATCHES == 0:
            txt.print_text_generation_header()
            ry = np.array([[txt.convert_from_alphabet(ord("K"))]])
            rh = np.zeros([1, INTERNALSIZE * NLAYERS])


            for k in range(144):
                ryo, rh = sess.run([Yo, H], feed_dict={X: ry, pkeep: 1.0, Hin: rh, batchsize: 1})
                rc = txt.sample_from_probabilities(ryo, topn=10 if epoch <= 1 else 2)
                print(chr(txt.convert_to_alphabet(rc)), end="")
                ry = np.array([[rc]])
            txt.print_text_generation_footer()

        # save a checkpoint (every 500 batches)
        if step // 10 % _50_BATCHES == 0:
            saved_file = saver.save(sess, 'twitter2_checkpoints/rnn_train_' + timestamp, global_step=step)
            print("Saved file: " + saved_file)

        # display progress bar
        progress.step(reset=step % _50_BATCHES == 0)

        # loop state around
        istate = ostate
        step += BATCHSIZE * SEQLEN
