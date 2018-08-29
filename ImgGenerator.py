import numpy as np
import keras
from augmentations import augmentation_clss #pip install albumentations
import warnings
warnings.filterwarnings(action='ignore', category=DeprecationWarning)


try:
    from PIL import ImageEnhance
    from PIL import Image as pil_image
except ImportError:
    pil_image = None
    ImageEnhance = None

if pil_image is not None:
    _PIL_INTERPOLATION_METHODS = {
        'nearest': pil_image.NEAREST,
        'bilinear': pil_image.BILINEAR,
        'bicubic': pil_image.BICUBIC,
    }
    # These methods were only introduced in version 3.4.0 (2016).
    if hasattr(pil_image, 'HAMMING'):
        _PIL_INTERPOLATION_METHODS['hamming'] = pil_image.HAMMING
    if hasattr(pil_image, 'BOX'):
        _PIL_INTERPOLATION_METHODS['box'] = pil_image.BOX
    # This method is new in version 1.1.3 (2013).
    if hasattr(pil_image, 'LANCZOS'):
        _PIL_INTERPOLATION_METHODS['lanczos'] = pil_image.LANCZOS


        
class Iterator(object):
    """Base class for image data iterators.
    Every `Iterator` must implement the `_get_batches_of_transformed_samples`
    method.
    # Arguments
        n: Integer, total number of samples in the dataset to loop over.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seeding for data shuffling.
    """

    def __init__(self, n, batch_size, shuffle, seed):
        self.n = n
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.batch_index = 0
        self.total_batches_seen = 0
        self.index_array = None
        self.index_generator = self._flow_index()

    def _set_index_array(self):
        self.index_array = np.arange(self.n)
        if self.shuffle:
            self.index_array = np.random.permutation(self.n)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise ValueError('Asked to retrieve element {idx}, '
                             'but the Sequence '
                             'has length {length}'.format(idx=idx,
                                                          length=len(self)))
        if self.seed is not None:
            np.random.seed(self.seed + self.total_batches_seen)
        self.total_batches_seen += 1
        if self.index_array is None:
            self._set_index_array()
        index_array = self.index_array[self.batch_size * idx:
                                       self.batch_size * (idx + 1)]

        return self._get_batches_of_transformed_samples(index_array)

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size  # round up

    def on_epoch_end(self):
        self._set_index_array()

    def reset(self):
        self.batch_index = 0

    def _flow_index(self):
        # Ensure self.batch_index is 0.
        self.reset()
        while 1:
            if self.seed is not None:
                np.random.seed(self.seed + self.total_batches_seen)
            if self.batch_index == 0:
                self._set_index_array()

            current_index = (self.batch_index * self.batch_size) % self.n
            if self.n > current_index + self.batch_size:
                self.batch_index += 1
            else:
                self.batch_index = 0
            self.total_batches_seen += 1
            yield self.index_array[current_index:
                                   current_index + self.batch_size]

    def __iter__(self):
        # Needed if we want to do something like:
        # for x, y in data_gen.flow(...):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)

    def _get_batches_of_transformed_samples(self, index_array):
        """Gets a batch of transformed samples.
        # Arguments
            index_array: Array of sample indices to include in batch.
        # Returns
            A batch of transformed samples.
        """
        raise NotImplementedError
        
class ImgFiletDataGen(Iterator):
    
    # Initialization
    def __init__(self, img_files, labels, rescale=1., aug_mode=None, 
                 target_size=(256, 256), color_mode='rgb',
                 classes=None, class_mode='categorical',
                 batch_size=32, shuffle=True, seed=None,
                 data_format='channels_last',
                 save_to_dir=None, save_prefix='', save_format='png',
                 follow_links=False,
                 interpolation='nearest',
                 dtype='float32'):
        self.img_files = img_files
        self.labels = labels
        self.rescale = rescale
        self.augmentation = aug_mode
        self.target_size = tuple(target_size)
        if color_mode not in {'rgb', 'rgba', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb", "rgba", or "grayscale".')
        self.color_mode = color_mode
        self.data_format = data_format
        if self.color_mode == 'rgba':
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (4,)
            else:
                self.image_shape = (4,) + self.target_size
        elif self.color_mode == 'rgb':
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
        else:
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size
        self.classes = classes
        if class_mode not in {'categorical', 'binary', 'sparse',
                              'input', None}:
            raise ValueError('Invalid class_mode:', class_mode,
                             '; expected one of "categorical", '
                             '"binary", "sparse", "input"'
                             ' or None.')
        self.class_mode = class_mode
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.interpolation = interpolation
        self.dtype = dtype

        white_list_formats = {'png', 'jpg', 'jpeg', 'bmp',
                              'ppm', 'tif', 'tiff'}
        
        if len(self.img_files)!=len(self.labels):
            raise ValueError('Number of image files and labels should be the same')
            
        self.num_classes = len(self.labels)
        self.samples = len(self.img_files)

        super(ImgFiletDataGen, self).__init__(self.samples,
                                                batch_size,
                                                shuffle,
                                                seed)



    # Generates data containing batch_size samples
    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros(
            (len(index_array),) + self.image_shape,
            dtype=self.dtype)
        
        # build batch of image data
        for i, j in enumerate(index_array):
            fname = self.img_files[j]
            img = self.load_img(fname,
                           color_mode=self.color_mode,
                           target_size=self.target_size,
                           interpolation=self.interpolation)
            x = np.array(img)#img_to_array(img, data_format=self.data_format)
            # Pillow images should be closed after `load_img`,
            # but not PIL images.
            if hasattr(img, 'close'):
                img.close()
                
            if self.augmentation:
                aug=augmentation_clss(self.augmentation)
                augmented_img=aug.augment_img(x)
                while(np.array_equal(augmented_img,np.zeros(augmented_img.shape)) == True):  #Avoid black images
                    augmented_img=aug.augment_img(x)
                x = augmented_img
                
              
            if self.rescale:
                x = x.astype(np.float32) * self.rescale
                
            batch_x[i] = x
        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i, j in enumerate(index_array):
                img = batch_x[i]
                fname = '{prefix}_{index}_{hash}.{format}'.format(
                    prefix=self.save_prefix,
                    index=j,
                    hash=np.random.randint(1e7),
                    format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        # build batch of labels
        if self.class_mode == 'input':
            batch_y = batch_x.copy()
        elif self.class_mode == 'sparse':
            batch_y = self.labels[index_array]
        elif self.class_mode == 'binary':
            batch_y = self.labels[index_array].astype(self.dtype)
        elif self.class_mode == 'categorical':
            batch_y = np.array(keras.utils.to_categorical(y, num_classes=self.n_classes))
        else:
            return batch_x
        #print(self.img_files[index_array])
        #print(batch_y)
        return batch_x, batch_y
    
    
    def next(self):
        """For python 2.x.
        # Returns
            The next batch.
        """
        
        index_array = next(self.index_generator)
        return self._get_batches_of_transformed_samples(index_array)

    
    #load_img code extracted from keras_preprocessing/image.py
    def load_img(self, path, grayscale=False, color_mode='rgb', target_size=None, interpolation='nearest'):
        """Loads an image into PIL format.
        # Arguments
            path: Path to image file.
            color_mode: One of "grayscale", "rbg", "rgba". Default: "rgb".
                The desired image format.
            target_size: Either `None` (default to original size)
                or tuple of ints `(img_height, img_width)`.
            interpolation: Interpolation method used to resample the image if the
                target size is different from that of the loaded image.
                Supported methods are "nearest", "bilinear", and "bicubic".
                If PIL version 1.1.3 or newer is installed, "lanczos" is also
                supported. If PIL version 3.4.0 or newer is installed, "box" and
                "hamming" are also supported. By default, "nearest" is used.
        # Returns
            A PIL Image instance.
        # Raises
            ImportError: if PIL is not available.
            ValueError: if interpolation method is not supported.
        """
        if grayscale is True:
            warnings.warn('grayscale is deprecated. Please use '
                          'color_mode = "grayscale"')
            color_mode = 'grayscale'
        if pil_image is None:
            raise ImportError('Could not import PIL.Image. '
                              'The use of `array_to_img` requires PIL.')
        img = pil_image.open(path)
        if color_mode == 'grayscale':
            if img.mode != 'L':
                img = img.convert('L')
        elif color_mode == 'rgba':
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
        elif color_mode == 'rgb':
            if img.mode != 'RGB':
                img = img.convert('RGB')
        else:
            raise ValueError('color_mode must be "grayscale", "rbg", or "rgba"')
        if target_size is not None:
            width_height_tuple = (target_size[1], target_size[0])
            if img.size != width_height_tuple:
                if interpolation not in _PIL_INTERPOLATION_METHODS:
                    raise ValueError(
                        'Invalid interpolation method {} specified. Supported '
                        'methods are {}'.format(
                            interpolation,
                            ", ".join(_PIL_INTERPOLATION_METHODS.keys())))
                resample = _PIL_INTERPOLATION_METHODS[interpolation]
                img = img.resize(width_height_tuple, resample)
        return img 