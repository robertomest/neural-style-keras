import keras.backend as K
from keras.applications import vgg16

'''
Module that defines loss functions and other auxiliary functions used when
training a pastiche model.
'''

def gram_matrix(x, norm_by_channels=False):
    '''
    Returns the Gram matrix of the tensor x.
    '''
    if K.ndim(x) == 3:
        features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
        shape = K.shape(x)
        C, H, W = shape[0], shape[1], shape[2]
        gram = K.dot(features, K.transpose(features))
    elif K.ndim(x) == 4:
        # Swap from (H, W, C) to (B, C, H, W)
        x = K.permute_dimensions(x, (0, 3, 1, 2))
        shape = K.shape(x)
        B, C, H, W = shape[0], shape[1], shape[2], shape[3]
        # Reshape as a batch of 2D matrices with vectorized channels
        features = K.reshape(x, K.stack([B, C, H*W]))
        # This is a batch of Gram matrices (B, C, C).
        gram = K.batch_dot(features, features, axes=2)
    else:
        raise ValueError('The input tensor should be either a 3d (H, W, C) or 4d (B, H, W, C) tensor.')
    # Normalize the Gram matrix
    if norm_by_channels:
        denominator = C * H * W # Normalization from Johnson
    else:
        denominator = H * W # Normalization from Google
    gram = gram /  K.cast(denominator, x.dtype)

    return gram


def content_loss(x, target):
    '''
    Content loss is simply the MSE between activations of a layer
    '''
    return K.mean(K.square(target - x))


def style_loss(x, target, norm_by_channels=False):
    '''
    Style loss is the MSE between Gram matrices computed using activation maps.
    '''
    x_gram = gram_matrix(x, norm_by_channels=norm_by_channels)
    return K.mean(K.square(target - x_gram))



def tv_loss(x):
    '''
    Total variation loss is used to keep the image locally coherent
    '''
    assert K.ndim(x) == 4
    a = K.square(x[:, :-1, :-1, :] - x[:, 1:, :-1, :])
    b = K.square(x[:, :-1, :-1, :] - x[:, :-1, 1:, :])
    return K.sum(K.mean(a + b, axis=0))


def get_content_features(out_dict, layer_names):
    return [out_dict[l] for l in layer_names]


def get_style_features(out_dict, layer_names, norm_by_channels=False):
    features = []
    for l in layer_names:
        layer_features = out_dict[l]
        S = gram_matrix(layer_features, norm_by_channels=norm_by_channels)
        features.append(S)
    return features


def get_loss_net(pastiche_net_output, input_tensor=None):
    '''
    Instantiates a VGG net and applies its layers on top of the pastiche net's
    output.
    '''
    loss_net = vgg16.VGG16(weights='imagenet', include_top=False,
                           input_tensor=input_tensor)
    targets_dict = dict([(layer.name, layer.output) for layer in loss_net.layers])
    i = pastiche_net_output
    # We need to apply all layers to the output of the style net
    outputs_dict = {}
    for l in loss_net.layers[1:]: # Ignore the input layer
        i = l(i)
        outputs_dict[l.name] = i

    return loss_net, outputs_dict, targets_dict


def get_style_losses(outputs_dict, targets_dict, style_layers,
                   norm_by_channels=False):
    '''
    Returns the style loss for the desired layers
    '''
    return [style_loss(outputs_dict[l], targets_dict[l],
                      norm_by_channels=norm_by_channels)
            for l in style_layers]

def get_content_losses(outputs_dict, targets_dict, content_layers):
    return [content_loss(outputs_dict[l], targets_dict[l])
            for l in content_layers]
