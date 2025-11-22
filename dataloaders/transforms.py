try:
    from torchvision.transforms import v2 as _T  # torchvision >= 0.15
    F = _T.functional
except Exception:
    from torchvision.transforms import functional as F  # older torchvision
import numpy as np
import numbers
from PIL import Image
import matplotlib.pyplot as plt


def imshow(img):
    plt.imshow(img)
    plt.axis("off")
    plt.show()


def get_padding(image, size):    
    w, h = image.size
    max_wh = size
    h_padding = (max_wh - w) / 2
    v_padding = (max_wh - h) / 2
    l_pad = h_padding if h_padding % 1 == 0 else h_padding+0.5
    t_pad = v_padding if v_padding % 1 == 0 else v_padding+0.5
    r_pad = h_padding if h_padding % 1 == 0 else h_padding-0.5
    b_pad = v_padding if v_padding % 1 == 0 else v_padding-0.5
    padding = (int(l_pad), int(t_pad), int(r_pad), int(b_pad))
    return padding


class Pad(object):
    def __init__(self, size=512, fill=0, padding_mode='constant'):
        assert isinstance(fill, (numbers.Number, str, tuple))
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']

        self.fill = fill
        self.padding_mode = padding_mode
        self.size = size

    def __call__(self, img):
        """
        Args:
            img (PIL Image or numpy array): Image to be padded.

        Returns:
            PIL Image: Padded image.
        """
        # Handle both numpy arrays and PIL Images
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        return F.pad(img, get_padding(img, self.size), self.fill, self.padding_mode)
    
    def __repr__(self):
        return self.__class__.__name__ + '(padding={0}, fill={1}, padding_mode={2})'.\
            format(self.fill, self.padding_mode)
    

class Rotate(object):
    def __init__(self, angle=45):
        self.angle = angle

    def __call__(self, img):
        # Handle both numpy arrays and PIL Images
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        angle = np.random.randint(0, self.angle)
        return F.rotate(img, angle)
    

class RandomHorizontalFlip(object):
    def __init__(self):
        pass

    def __call__(self, img):
        # Handle both numpy arrays and PIL Images
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        try:
            return F.horizontal_flip(img)
        except AttributeError:
            return F.hflip(img)  
        

class RandomVerticalFlip(object):
    def __init__(self):
        pass

    def __call__(self, img):
        # Handle both numpy arrays and PIL Images
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        try:
            return F.vertical_flip(img)
        except AttributeError:
            return F.vflip(img)
        

class RandomScale(object):
    def __init__(self, scale_range=(0.8, 1.2)):
        """
        Random scale augmentation.
        Args:
            scale_range: tuple (min_scale, max_scale) for random scaling factor
        """
        self.scale_range = scale_range

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be scaled.
        Returns:
            PIL Image: Scaled image.
        """
        scale_factor = np.random.uniform(self.scale_range[0], self.scale_range[1])
        
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        
        w, h = img.size
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        
        # Use NEAREST for depth/mask to avoid interpolation artifacts
        # Use BILINEAR for RGB
        if img.mode == 'F' or img.mode == 'L':  # depth or mask
            return F.resize(img, (new_h, new_w), interpolation=Image.NEAREST)
        else:  # RGB
            return F.resize(img, (new_h, new_w), interpolation=Image.BILINEAR)
    
    def __repr__(self):
        return self.__class__.__name__ + f'(scale_range={self.scale_range})'


class AdjustBrightness(object):
    def __init__(self, brightness_factor=0.2):
        self.bf = brightness_factor

    def __call__(self, img):
        return F.adjust_brightness(img, self.bf)