# Credit
This project was forked from
https://github.com/martin-gorner/tensorflow-rnn-shakespeare

assume all code is attributed to and owned by Martin Gorner.

# Original Code And Presentation
Code for the Recurrent Neural Network in the presentation "Tensorflow and deep learning - without a PhD, Part 2"

The presentation itself is available here:

* [Video](https://t.co/cIePWmdxVE)
* [Slides](https://goo.gl/jrd7AR)

This sample has now been updated for Tensorflow 1.1. Please make sure you redownload the checkpoint files if you use rnn_play.py.


## Usage:

### To Train

#### Step 1 - Train on Twitter Data
```$ python3 rnn_train.py```
This will train the model and save it with checkpoint files.

After training, you can see what the model outputs by using

```$ python3 rnn_play.py```

You will need to replace the files with your checkpoint files.

#### Step 2 - Train on pickup line data

```$ python3 rnn_retrain.py```

You will need to replace the checkpoint file names for this to work.


### To Play

Point to the correct checkpoint files and do
```$ python3 rnn_play.py```


Each training session will generate saved checkpoints as well as save a final model when it has finished. you can monitor the progress using
Tensorboard

```$ tensorboard --logdir="your/log/path"```
