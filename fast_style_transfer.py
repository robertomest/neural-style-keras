'''
Use a trained pastiche net to stylize images.
'''

from __future__ import print_function
import os
import argparse

import numpy as np
import tensorflow as tf
import keras
import keras.backend as K

from utils import config_gpu, preprocess_image_scale, deprocess_image
import h5py
import yaml
import time
from scipy.misc import imsave

from model import pastiche_model

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Use a trained pastiche network.')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoint')
    parser.add_argument('--img_size', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--input_path', type=str, default='pastiche_input')
    parser.add_argument('--output_path', type=str, default='pastiche_output')
    parser.add_argument('--use_style_name', default=False, action='store_true')
    parser.add_argument('--gpu', type=str, default='')
    parser.add_argument('--allow_growth', default=False, action='store_true')

    args = parser.parse_args()

    config_gpu(args.gpu, args.allow_growth)

    # Strip the extension if there is one
    checkpoint_path = os.path.splitext(args.checkpoint_path)[0]

    with h5py.File(checkpoint_path + '.h5', 'r') as f:
        model_args = yaml.load(f.attrs['args'])
        style_names = f.attrs['style_names']

    print('Creating pastiche model...')
    class_targets = K.placeholder(shape=(None,), dtype=tf.int32)
    # Intantiate the model using information stored on tha yaml file
    pastiche_net = pastiche_model(None, width_factor=model_args.width_factor,
                                  nb_classes=model_args.nb_classes,
                                  targets=class_targets)
    with h5py.File(checkpoint_path + '.h5', 'r') as f:
        pastiche_net.load_weights_from_hdf5_group(f['model_weights'])

    inputs = [pastiche_net.input, class_targets, K.learning_phase()]

    transfer_style = K.function(inputs, [pastiche_net.output])

    num_batches = int(np.ceil(model_args.nb_classes / float(args.batch_size)))

    for img_name in os.listdir(args.input_path):
        print('Processing %s' %img_name)
        img = preprocess_image_scale(os.path.join(args.input_path, img_name),
                                     img_size=args.img_size)
        imgs = np.repeat(img, model_args.nb_classes, axis=0)
        out_name = os.path.splitext(os.path.split(img_name)[-1])[0]

        for batch_idx in range(num_batches):
            idx = batch_idx * args.batch_size

            batch = imgs[idx:idx + args.batch_size]
            indices = batch_idx * args.batch_size + np.arange(batch.shape[0])

            if args.use_style_name:
                names = style_names[idx:idx + args.batch_size]
            else:
                names = indices
            print('  Processing styles %s' %str(names))

            out = transfer_style([batch, indices, 0.])[0]

            for name, im in zip(names, out):
                print('Saving file %s_style_%s.png' %(out_name, str(name)))
                imsave(os.path.join(args.output_path, '%s_style_%s.png' %(out_name, str(name))),
                       deprocess_image(im[None, :, :, :].copy()))
