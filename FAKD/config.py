#edge_information,psnr,ssim,gradient panelty
from torch import nn
from torchvision import transforms
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
from datetime import datetime
import os


now_time = datetime.now()
time = now_time.strftime("%Y-%m-%d-%X")[:-6]  # 格式化时间字符串
DEVICE = "cuda:0"
model = 'fakd'
HIGH_RES = 256
BATCH_SIZE = 32
NUM_WORKERS = 8
LEARNING_RATE = 1e-5
comment = 'RRDB23+RaGAN+gram+pixelshuffle+distill'
EPOCHS = 200
lambda_pixel = 1
lambda_adv = 5e-3
lambda_edge = 1e-2
lambda_gram = 1e-2
lambda_content = 1.0
lambda_KD = 1e-1
gan_model = 'ragan'
warmup = 0
D_warmup = 0

# student_path = './saved_models/distill/2022-04-24-21/generator.pth'
#student_path = './saved_models/distill/2022-04-26-20/generator.pth'
student_path = '../WSRGAN/saved_models/normal/2022-05-06-18/generator240.pth'
teacher_path = '../WSRGAN/saved_models/normal/2022-04-28-17/generator239.pth'
discriminator_path = ''


def loader_model(student,teacher,disc=False):
    if student_path != '':
        if os.path.exists(student_path):
            student.load_state_dict(torch.load(student_path), strict=True)
        else:
            raise FileNotFoundError("not found weights file: {}".format(student_path))
    if teacher_path != '':
        if os.path.exists(teacher_path):
            teacher.load_state_dict(torch.load(teacher_path), strict=True)
        else:
            raise FileNotFoundError("not found weights file: {}".format(teacher_path))
    if discriminator_path != '':
        if os.path.exists(discriminator_path):
            disc.load_state_dict(torch.load(discriminator_path), strict=True)
        else:
            raise FileNotFoundError("not found weights file: {}".format(discriminator_path))


def cal_psnr(real,fake):
    psnr = 10*torch.log10(1/((real - fake) ** 2).mean())
    return psnr

'''edge information'''
# 定义sobel算子参数
sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32')
# 将sobel算子转换为适配卷积操作的卷积核
sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
# 卷积输出通道，这里我设置为3
sobel_kernel = np.repeat(sobel_kernel, 3, axis=1)
# 输入图的通道，这里我设置为3
sobel_kernel = np.repeat(sobel_kernel, 3, axis=0)
def edge_conv2d(im,device="cuda"):
    # 用nn.Conv2d定义卷积操作
    conv_op = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)

    conv_op.weight.data = torch.from_numpy(sobel_kernel).to(device)
    # print(conv_op.weight.size())
    # print(conv_op, '\n')

    edge_detect = conv_op(im)
    # 将输出转换为图片格式
    # edge_detect = edge_detect.squeeze().detach().numpy()
    return edge_detect

def cal_gradient_penalty(netD, real_data, fake_data, device='cuda', type='mixed', constant=1.0, lambda_gp=10):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028
    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( ||gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss
    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty
    else:
        return 0.0
#ssim
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)