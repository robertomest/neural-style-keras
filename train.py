'''
This script can be used to train a pastiche network.
'''

from __future__ import print_function
import os
import argparse

import time
import h5py

import numpy as np
import tensorflow as tf
import keras
import keras.backend as K
from keras.optimizers import Adam
from model import pastiche_model
from training import get_loss_net, get_content_losses, get_style_losses, tv_loss
from utils import preprocess_input, config_gpu, save_checkpoint

if __name__ == '__main__':
    def_cl = ['block3_conv3']
    def_sl = ['block1_conv2', 'block2_conv2',
              'block3_conv3', 'block4_conv3']

    # Argument parser
    parser = argparse.ArgumentParser(description='Train a pastiche network.')
    parser.add_argument('--lr', help='Learning rate.', type=float, default=0.001)
    parser.add_argument('--content_weight', type=float, default=1.)
    parser.add_argument('--style_weight', type=float, default=1e-4)
    parser.add_argument('--tv_weight', type=float, default=1e-4)
    parser.add_argument('--content_layers', type=str, nargs='+', default=def_cl)
    parser.add_argument('--style_layers', type=str, nargs='+', default=def_sl)
    parser.add_argument('--width_factor', type=int, default=2)
    parser.add_argument('--nb_classes', type=int, default=1)
    parser.add_argument('--norm_by_channels', default=False, action='store_true')
    parser.add_argument('--num_iterations', type=int, default=40000)
    parser.add_argument('--save_every', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--coco_path', type=str, default='data/coco/ms-coco-256.h5')
    parser.add_argument('--gram_dataset_path', type=str, default='grams.h5')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoint.h5')
    parser.add_argument('--gpu', type=str, default='')
    parser.add_argument('--allow_growth', default=False, action='store_true')

    args = parser.parse_args()
    # Arguments parsed

    config_gpu(args.gpu, args.allow_growth)

    print('Creating pastiche model...')
    class_targets = K.placeholder(shape=(None,), dtype=tf.int32)
    # The model will be trained with 256 x 256 images of the coco dataset.
    pastiche_net = pastiche_model(256, width_factor=args.width_factor,
                                  nb_classes=args.nb_classes,
                                  targets=class_targets)
    x = pastiche_net.input
    o = pastiche_net.output

    print('Loading loss network...')
    loss_net, outputs_dict, content_targets_dict = get_loss_net(pastiche_net.output, input_tensor=pastiche_net.input)

    # Placeholder sizes
    ph_sizes = {k : K.int_shape(content_targets_dict[k])[-1] for k in args.style_layers}

    # Our style targets are precomputed and are fed through these placeholders
    style_targets_dict = {k : K.placeholder(shape=(None, ph_sizes[k], ph_sizes[k])) for k in args.style_layers}


    print('Setting up training...')
    style_losses = get_style_losses(outputs_dict, style_targets_dict, args.style_layers,
                                    norm_by_channels=args.norm_by_channels)

    content_losses = get_content_losses(outputs_dict, content_targets_dict, args.content_layers)

    # Use total variation to improve local coherence
    total_var_loss = tv_loss(pastiche_net.output)

    # Compute total loss
    total_loss = K.variable(0.)
    for loss in style_losses:
        total_loss += args.style_weight * loss
    for loss in content_losses:
        total_loss += args.content_weight * loss
    total_loss += args.tv_weight * total_var_loss


    ## Make training function

    # Get a list of inputs
    inputs = [pastiche_net.input, class_targets] + \
             [style_targets_dict[k] for k in args.style_layers] + \
             [K.learning_phase()]

    # Get trainable params
    params = pastiche_net.trainable_weights
    constraints = pastiche_net.constraints

    opt = Adam(lr=args.lr)
    updates = opt.get_updates(params, constraints, total_loss)

    # List of outputs
    outputs = [total_loss] + content_losses + style_losses + [total_var_loss]

    f_train = K.function(inputs, outputs, updates)

    X = h5py.File(args.coco_path, 'r')['train2014']['images']
    dataset_size = X.shape[0]
    batches_per_epoch = int(np.ceil(dataset_size / args.batch_size))
    batch_idx = 0

    print('Loading Gram matrices from dataset file...')
    if args.norm_by_channels:
        print('Normalizing the stored Gram matrices by the number of channels.')
    Y = {}
    with h5py.File(args.gram_dataset_path, 'r') as f:
        styles = f.attrs['img_names']
        for k, v in f.iteritems():
            Y[k] = np.array(v)
            if args.norm_by_channels:
                #Correct the Gram matrices from the dataset
                Y[k] /= Y[k].shape[-1]

    # Get a log going
    log = {}
    log['args'] = args
    log['styles'] = styles[:args.nb_classes]
    log['total_loss'] = []
    log['style_loss'] = {k: [] for k in args.style_layers}
    log['content_loss'] = {k: [] for k in args.content_layers}
    log['tv_loss'] = []

    # Strip the extension if there is one
    checkpoint_path = os.path.splitext(args.checkpoint_path)[0]

    start_time = time.time()
    for it in range(args.num_iterations):
        if batch_idx >= batches_per_epoch:
            print('Epoch done. Going back to the beginning...')
            batch_idx = 0

        # Get the batch
        idx = args.batch_size * batch_idx
        batch = X[idx:idx+args.batch_size]
        batch = preprocess_input(batch)
        batch_idx += 1

        # Get class information for each image on the batch
        batch_classes = np.random.randint(args.nb_classes, size=(args.batch_size,))

        batch_targets = [Y[l][batch_classes] for l in args.style_layers]

        # Do a step
        start_time2 = time.time()
        out = f_train([batch, batch_classes] + batch_targets + [1.])
        stop_time2 = time.time()
        # Log the statistics

        log['total_loss'].append(out[0])
        offset = 1
        for i, k in enumerate(args.content_layers):
            log['content_loss'][k].append(args.content_weight * out[offset + i])
        offset += len(args.content_layers)
        for i, k in enumerate(args.style_layers):
            log['style_loss'][k].append(args.style_weight * out[offset + i])
        log['tv_loss'].append(args.tv_weight * out[-1])

        stop_time = time.time()
        print('Iteration %d/%d: loss = %f. t = %f (%f)' %(it + 1,
              args.num_iterations, out[0], stop_time - start_time,
              stop_time2 - start_time2))

        if not ((it + 1) % args.save_every):
            print('Saving checkpoint in %s.h5...' %(checkpoint_path))
            save_checkpoint(checkpoint_path, pastiche_net, log)
            print('Checkpoint saved.')

        start_time = time.time()
    save_checkpoint(checkpoint_path, pastiche_net, log)
