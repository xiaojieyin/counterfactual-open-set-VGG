import torch.utils.data as data

from PIL import Image

import os
import os.path
import random

CUB_CLASSES_SINGLE = [
    'Groove_billed_Ani', 'Bobolink', 'Cardinal', 'Yellow_breasted_Chat', 'Eastern_Towhee',
    'Chuck_will_Widow', 'Brown_Creeper', 'Northern_Flicker', 'Frigatebird', 'Northern_Fulmar',
    'Gadwall', 'Boat_tailed_Grackle', 'Pigeon_Guillemot', 'Green_Violetear', 'Dark_eyed_Junco',
    'Red_legged_Kittiwake', 'Horned_Lark', 'Pacific_Loon', 'Mallard', 'Western_Meadowlark',
    'Mockingbird', 'Nighthawk', 'Ovenbird', 'Western_Wood_Pewee', 'Sayornis',
    'American_Pipit', 'Whip_poor_Will', 'Horned_Puffin', 'American_Redstart', 'Geococcyx',
    'Cape_Glossy_Starling', 'Green_tailed_Towhee', 'Common_Yellowthroat',
]  # 33

CUB_CLASSRS_WARBLER = [
    'Bay_breasted_Warbler', 'Black_and_white_Warbler', 'Black_throated_Blue_Warbler', 'Blue_winged_Warbler',
    'Canada_Warbler',
    'Cape_May_Warbler', 'Cerulean_Warbler', 'Chestnut_sided_Warbler', 'Golden_winged_Warbler', 'Hooded_Warbler',
    'Kentucky_Warbler', 'Magnolia_Warbler', 'Mourning_Warbler', 'Myrtle_Warbler', 'Nashville_Warbler',
    'Orange_crowned_Warbler', 'Palm_Warbler', 'Pine_Warbler', 'Prairie_Warbler', 'Prothonotary_Warbler',
    'Swainson_Warbler', 'Tennessee_Warbler', 'Wilson_Warbler', 'Worm_eating_Warbler', 'Yellow_Warbler',
]  # 25


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def find_classes(dir, seed, num_classes, fold):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()

    # if seed is not None:
    #     random.seed(seed)
    # random_classes = CUB_CLASSES_SINGLE.copy()
    # random.shuffle(random_classes)
    # unknown_classes = CUB_CLASSRS_WARBLER + random_classes[:25]
    # if num_classes is not None:
    #     if fold == 'known':
    #         classes = [e for e in classes if e.lstrip('.0123456789').split('/')[0] not in unknown_classes]
    #     elif fold == 'unknown':
    #         classes = [e for e in classes if e.lstrip('.0123456789').split('/')[0] in unknown_classes]

    if num_classes is not None:
        if fold == 'known':
            classes = classes[:num_classes]
        elif fold == 'unknown':
            classes = classes[num_classes:]

    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, classes, class_to_idx, extensions):
    images = []
    dir = os.path.expanduser(dir)
    for class_name in sorted(classes):
        d = os.path.join(dir, class_name)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[class_name])
                    images.append(item)
    return images


class OpenSetDatasetFolder(data.Dataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (list[string]): A list of allowed extensions.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
    """

    def __init__(self, root, loader, extensions, transform=None, target_transform=None, seed=None, num_classes=None,
                 fold=None):
        classes, class_to_idx = find_classes(root, seed, num_classes, fold)
        samples = make_dataset(root, classes, class_to_idx, extensions)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                                                                            "Supported extensions are: " + ",".join(
                extensions)))

        self.root = root
        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class OpenSetImageFolder(OpenSetDatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader, seed=None, num_classes=None, fold='known'):
        super(OpenSetImageFolder, self).__init__(root, loader, IMG_EXTENSIONS,
                                                 transform=transform,
                                                 target_transform=target_transform,
                                                 seed=seed,
                                                 num_classes=num_classes,
                                                 fold=fold)
        self.imgs = self.samples
