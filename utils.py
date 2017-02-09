'''
Module constains utilitary functions.
'''

def config_gpu(gpu, allow_growth):
    # Choosing gpu
    if gpu == '-1':
        config = tf.ConfigProto(device_count ={'GPU': 0})
    else:
        if gpu == 'all' or gpu == '':
            gpu = ''
        config = tf.ConfigProto()
        config.gpu_options.visible_device_list = gpu
    if allow_growth == True:
        config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    K.set_session(session)

def save_checkpoint(checkpoint_path, pastiche_net, log):
    with h5py.File(checkpoint_path + '.h5', 'w') as f:
        g  = f.create_group('model_weights')
        pastiche_net.save_weights_to_hdf5_group(g)
        g =  f.create_group('log')
        g.create_dataset('total_loss', data=np.array(log['total_loss']))
        g.create_dataset('tv_loss', data=np.array(log['tv_loss']))
        g2 = g.create_group('style_loss')
        for k, v in log['style_loss'].items():
            g2.create_dataset(k, data=v)
        g2 = g.create_group('content_loss')
        for k, v in log['content_loss'].items():
            g2.create_dataset(k, data=v)
        f.attrs['args'] = yaml.dump(log['args'])
        f.attrs['styles'] = log['styles']

def preprocess_input(x):
    return vgg16.preprocess_input(x.astype('float32'))

def preprocess_image_crop(image_path, img_size):
    '''
    Preprocess the image scaling it so that its smaller size is img_size.
    The larger size is then cropped in order to produce a square image.
    '''
    img = load_img(image_path)
    scale = float(img_size) / min(img.size)
    new_size = (int(np.ceil(scale * img.size[0])), int(np.ceil(scale * img.size[1])))
    # print('old size: %s,new size: %s' %(str(img.size), str(new_size)))
    img = img.resize(new_size, resample=Image.BILINEAR)
    img = img_to_array(img)
    crop_h = img.shape[0] - img_size
    crop_v = img.shape[1] - img_size
    img = img[crop_h:img_size+crop_h, crop_v:img_size+crop_v, :]
    img = np.expand_dims(img, axis=0)
    img = vgg16.preprocess_input(img)
    return img

# util function to open, resize and format pictures into appropriate tensors
def preprocess_image_scale(image_path, img_size=None):
    '''
    Preprocess the image scaling it so that its larger size is max_size.
    This function preserves aspect ratio.
    '''
    img = load_img(image_path)
    if img_size:
        scale = float(img_size) / max(img.size)
        new_size = (int(np.ceil(scale * img.size[0])), int(np.ceil(scale * img.size[1])))
        img = img.resize(new_size, resample=Image.BILINEAR)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg16.preprocess_input(img)
    return img


# util function to convert a tensor into a valid image
def deprocess_image(x):
    x = x[0]
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x
