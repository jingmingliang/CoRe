import torch
import torchvision.transforms.functional as F
import numpy as np


def gaussian_noise(img, mean=0, std=1):
    """添加高斯噪声"""
    noise = torch.randn_like(img) * std + mean
    return img + noise


def salt_and_pepper_noise(img, prob=0.05, salt_ratio=0.5):
    """添加椒盐噪声"""
    img = F.to_pil_image(img)
    img = np.array(img)
    h, w = img.shape[:2]
    mask = np.random.rand(h, w)
    salt = mask < prob * salt_ratio
    pepper = mask > (1 - prob * (1 - salt_ratio))
    img[salt] = 255
    img[pepper] = 0
    img = F.to_tensor(img)
    return img


def spike_noise(img, prob=0.05):
    """添加毛刺噪声"""
    img = F.to_pil_image(img)
    img = np.array(img)
    h, w = img.shape[:2]
    mask = np.random.rand(h, w)
    noise = mask < prob
    img[noise] = np.random.choice([0, 255])
    img = F.to_tensor(img)
    return img


def brightness_noise(img, factor=0.1):
    """添加亮度噪声"""
    img = F.adjust_brightness(img, brightness_factor=1 + factor * (torch.rand(1) - 0.5))
    return img


class RandomNoise(object):
    def __init__(self, noise_type='gaussian', **kwargs):
        self.noise_type = noise_type
        self.kwargs = kwargs

    def __call__(self, img):
        if self.noise_type == 'gaussian':
            img = gaussian_noise(img, **self.kwargs)
        elif self.noise_type == 'salt_and_pepper':
            img = salt_and_pepper_noise(img, **self.kwargs)
        elif self.noise_type == 'spike':
            img = spike_noise(img, **self.kwargs)
        elif self.noise_type == 'brightness':
            img = brightness_noise(img, **self.kwargs)
        else:
            raise ValueError(f'Noise type {self.noise_type} not supported')

        return img


# 例子
# transform = transforms.Compose([
#     transforms.RandomCrop(32),
#     transforms.RandomHorizontalFlip(),
#     RandomNoise(noise_type='gaussian', mean=0, std=0.1),
#     transforms.ToTensor(),
# ])
#以上代码定义了一个transform，包含了随机裁剪、随机水平翻转、添加高斯噪声、转换为tensor等操作。
# 其中RandomNoise类的参数noise_type可以选择四种噪声类型之一，同时可以手动传入其他参数，例如mean、std等。
