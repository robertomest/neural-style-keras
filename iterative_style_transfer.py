'''
Original neural style transfer algorithm that iteratively updates the input
image in order to minimize the loss. Script inspired on the Keras example
available at
https://github.com/fchollet/keras/blob/master/examples/neural_style_transfer.py
'''

import os
import argparse
import time

import numpy as np
import tensorflow
import keras
import keras.backend as K
from keras.optimizers import Adam
from keras.applications import vgg19
from keras.layers import Input

from training import get_content_features, get_style_features, get_content_losses, get_style_losses, tv_loss
from utils import config_gpu, preprocess_image_scale, deprocess_image

from scipy.misc import imsave


if __name__ == '__main__':
    def_cl = ['block4_conv2']
    def_sl = ['block1_conv1', 'block2_conv1',
              'block3_conv1', 'block4_conv1',
              'block5_conv1']

    parser = argparse.ArgumentParser(description='Iterative style transfer.')
    parser.add_argument('--content_image_path', type=str,
                        default='content_imgs/tuebingen.jpg',
                        help='Path to the image to transform.')
    parser.add_argument('--style_image_path', type=str,
                        default='style_imgs/starry_night.jpg', nargs='+',
                        help='Path to the style reference image. Can be a list.')
    parser.add_argument('--output_path', type=str, default='pastiche_output')
    parser.add_argument('--lr', help='Learning rate.', type=float, default=10.)
    parser.add_argument('--num_iterations', type=int, default=1000)
    parser.add_argument('--content_weight', type=float, default=1.)
    parser.add_argument('--style_weight', type=float, default=1e-4)
    parser.add_argument('--tv_weight', type=float, default=1e-4)
    parser.add_argument('--content_layers', type=str, nargs='+', default=def_cl)
    parser.add_argument('--style_layers', type=str, nargs='+', default=def_sl)
    parser.add_argument('--norm_by_channels', default=False, action='store_true')
    parser.add_argument('--img_size', type=int, default=512,
                        help='Maximum heigth/width of generated image.')
    parser.add_argument('--style_img_size', type=int, default=None,
                        help='Maximum height/width of the style images.')
    parser.add_argument('--print_and_save', type=float, default=100,
                        help='Print and save image every chosen iterations.')
    parser.add_argument('--init', type=str, default='random',
                        help='How to initialize the pastiche images.')
    parser.add_argument('--std_init', type=float, default=0.001,
                        help='Standard deviation for random init.')
    parser.add_argument('--gpu', type=str, default='')
    parser.add_argument('--allow_growth', default=False, action='store_true')

    args = parser.parse_args()
    # Arguments parsed

    config_gpu(args.gpu, args.allow_growth)

    ## Precomputing the targets for content and style
    # Load content and style images
    content_image = preprocess_image_scale(args.content_image_path,
                                           img_size=args.img_size)
    style_images = [preprocess_image_scale(img, img_size=args.style_img_size)
                     for img in args.style_image_path]
    nb_styles = len(style_images)

    model = vgg19.VGG19(weights='imagenet', include_top=False)
    outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

    content_features = get_content_features(outputs_dict, args.content_layers)
    style_features = get_style_features(outputs_dict, args.style_layers,
                                        norm_by_channels=args.norm_by_channels)

    get_content_fun = K.function([model.input], content_features)
    get_style_fun = K.function([model.input], style_features)

    content_targets = get_content_fun([content_image])
    # List of list of features
    style_targets_list = [get_style_fun([img]) for img in style_images]

    # List of batched features
    style_targets = []
    for l in range(len(args.style_layers)):
        batched_features = []
        for i in range(nb_styles):
            batched_features.append(style_targets_list[i][l][None])
        style_targets.append(np.concatenate(batched_features))

    if args.init == 'content':
        pastiche_image = K.variable(np.repeat(content_image, nb_styles, axis=0))
    else:
        if args.init != 'random':
            print('Could not recognize init arg \'%s\'. Falling back to random.' %args.init)
        pastiche_image = K.variable(args.std_init*np.random.randn(nb_styles, *content_image.shape[1:]))

    # Store targets as variables
    content_targets_dict = {k: K.variable(v) for k, v in zip(args.content_layers, content_targets)}
    style_targets_dict = {k: K.variable(v) for k, v in zip(args.style_layers, style_targets)}

    model = vgg19.VGG19(weights='imagenet', include_top=False, input_tensor=Input(tensor=pastiche_image))
    outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

    content_losses = get_content_losses(outputs_dict, content_targets_dict,
                                        args.content_layers)
    style_losses = get_style_losses(outputs_dict, style_targets_dict,
                                    args.style_layers,
                                    norm_by_channels=args.norm_by_channels)

    # Total variation loss is used to improve local coherence
    total_var_loss = tv_loss(pastiche_image)

    # Compute total loss
    total_loss = K.variable(0.)
    for loss in style_losses:
        total_loss += args.style_weight * loss
    for loss in content_losses:
        total_loss += args.content_weight * loss
    total_loss += args.tv_weight * total_var_loss

    opt = Adam(lr=args.lr)
    updates = opt.get_updates([pastiche_image], {}, total_loss)
    # List of outputs
    outputs = [total_loss] + content_losses + style_losses + [total_var_loss]

    # Function that makes a step after backpropping to the image
    make_step = K.function([], outputs, updates)


    # Perform optimization steps and save the results
    start_time = time.time()

    for i in range(args.num_iterations):
        out = make_step([])
        if (i + 1) % args.print_and_save == 0:
            print('Iteration %d/%d' %(i + 1, args.num_iterations))
            N = len(content_losses)
            for j, l in enumerate(out[1:N+1]):
                print('    Content loss %d: %g' %(j, args.content_weight * l))
            for j, l in enumerate(out[N+1:-1]):
                print('    Style loss %d: %g' %(j, args.style_weight * l))

            print('    Total style loss: %g' %(args.style_weight * sum(out[N+1:-1])))
            print('    TV loss: %g' %(args.tv_weight * out[-1]))
            print('    Total loss: %g' %out[0])
            stop_time = time.time()
            print('Did %d iterations in %.2fs.' %(args.print_and_save, stop_time - start_time))
            x = K.get_value(pastiche_image)
            for s in range(nb_styles):
                fname = args.output_path + '_style%d_%d.png' %(s, (i + 1) / args.print_and_save)
                print('Saving image to %s.\n' %fname)
                img = deprocess_image(x[s:s+1])
                imsave(fname, img)
            start_time = time.time()
