
import os
import tensorflow as tf
import numpy as np

def load_data_using_tfdata(dataset, input_shape, steps_per_epoch, folders, addCharacteristics):
    """
    Load the images in batches using Tensorflow (tfdata).
    Cache can be used to speed up the process.
    Faster method in comparison to image loading using Keras.
    Returns:
    Data Generator to be used while training the model.
    """
    def parse_image(file_path):
        # convert the path to a list of path components
        parts = tf.strings.split(file_path, os.path.sep)
        class_names = np.array(os.listdir(dir_path + '/train'))
        # The second to last is the class-directory
        class_names.sort()
        label = parts[-2] == class_names
        label = tf.dtypes.cast(label, tf.int8)
        print("pars", parts)
        print("class_names",class_names)
        if addCharacteristics:
            r = np.zeros((label.shape[0]+s.shape[1]))
            print("This is not working",r,  r.shape, label.numpy(), label.numpy().shape)
            r[:label.shape[0]] = label[:]
            r[label.shape[0]:] = s[label,:]
            label = r
        # load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        # convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_jpeg(img, channels=3)
        # Use `convert_image_dtype` to convert to floats in the [0,1] range
        img = tf.image.convert_image_dtype(img, tf.float32)
        # resize the image to the desired size.
        img = tf.image.resize(img, [img_dims[0], img_dims[1]])
        return img, label

    def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000):
        # If a small dataset, only load it once, and keep it in memory.
        # use `.cache(filename)` to cache preprocessing work for datasets
        # that don't fit in memory.
        if cache:
            if isinstance(cache, str):
                ds = ds.cache(cache)
            else:
                ds = ds.cache()
        ds = ds.shuffle(buffer_size=shuffle_buffer_size)
        # Repeat forever
        ds = ds.repeat()
        ds = ds.batch(batch_size)
        # `prefetch` lets the dataset fetch batches in the background
        # while the model is training.
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds

    dir_path = dataset
    img_dims = input_shape
    batch_size = steps_per_epoch
    s = np.genfromtxt('characteristics2.csv', delimiter=',')

    data_generator = {}
    for x in folders:
        dir_extend = dir_path + '/' + x
        list_ds = tf.data.Dataset.list_files(str(dir_extend+'/*/*'))
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        # Set `num_parallel_calls` so that multiple images are
        # processed in parallel
        labeled_ds = list_ds.map(
            parse_image, num_parallel_calls=AUTOTUNE)
        
        data_generator[x + '_count'] = len(labeled_ds)
        
        # cache = True, False, './file_name'
        # If the dataset doesn't fit in memory use a cache file,
        # eg. cache='./data.tfcache'
        data_generator[x] = prepare_for_training(
            labeled_ds, cache='data.tfcache')

    return data_generator