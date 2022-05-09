import torch.nn as nn
from torchvision.models import vgg19
import config
import torch

class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg19_model = vgg19(pretrained=True).to(config.DEVICE)
        self.vgg19_feature = nn.Sequential(*list(vgg19_model.features.children())[:31])
        self.selected_layers = [7,14,23,30]

        mean = torch.Tensor([0.485,0.456,0.406]).view(1,3,1,1).to(config.DEVICE)
        std = torch.Tensor([0.229,0.224,0.225]).view(1,3,1,1).to(config.DEVICE)
        self.register_buffer('mean',mean)
        self.register_buffer('std',std)

        for k,v in self.vgg19_feature.named_parameters():
            v.requires_grad = False

        self.loss = nn.L1Loss()
        self.vgg19_feature.eval()

    def forward(self, fake, real):
        gen_feature = {}
        real_feature = {}
        G_real = {}
        G_fake = {}
        input = (fake - self.mean) / self.std
        target = (real - self.mean) /self.std
        for index, layer in enumerate(self.vgg19_feature):
            input = layer(input)
            target = layer(target)
            if index in self.selected_layers:
                gen_feature[index] = input
                real_feature[index] = target

        content_loss = 0.1*self.loss(gen_feature[7],real_feature[7])+ \
                            0.1 * self.loss(gen_feature[14], real_feature[14])+ \
                            0.4 * self.loss(gen_feature[23], real_feature[23])+ \
                            0.4 * self.loss(gen_feature[30], real_feature[30])

        #gram_loss
        batch = input.shape[0]
        channel = input.shape[1]
        fm_size = torch.tensor(input.shape[2:])
        M = torch.prod(fm_size)
        for i in self.selected_layers:
            F_real = real_feature[i].reshape(batch, channel, -1)
            F_fake = gen_feature[i].reshape(batch, channel, -1)
            G_real[i] = torch.bmm(F_real, F_real.permute(0, 2, 1)) / M
            G_fake[i] = torch.bmm(F_fake, F_fake.permute(0, 2, 1)) / M
        gram_loss = 0.1 * self.loss(G_fake[7],G_real[7])+ \
                    0.1 * self.loss(G_fake[14], G_real[14]) + \
                    0.4 * self.loss(G_fake[23], G_real[23]) + \
                    0.4 * self.loss(G_fake[30], G_real[30])

        # self.gram_loss += 0.25 * k * ((G_real - G_fake) ** 2).sum() / channel ** 2

        return content_loss,gram_loss

class CharbonnierLoss(nn.Module):
    def __init__(self,eps=1e-6):
        super(CharbonnierLoss,self).__init__()
        self.eps = eps
    def forward(self,x,y):
        diff = torch.add(x,-y)
        error = torch.sqrt(diff*diff + self.eps)
        loss = torch.mean(error)
        return loss


# Define GAN loss: [vanilla | lsgan | wgan-gp]
class GANLoss(nn.Module):
    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type.lower()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'gan' or self.gan_type == 'ragan':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan-gp':

            def wgan_loss(input, target):
                # target is boolean
                return -1 * input.mean() if target else input.mean()

            self.loss = wgan_loss
        else:
            raise NotImplementedError('GAN type [{:s}] is not found'.format(self.gan_type))

    def get_target_label(self, input, target_is_real):
        if self.gan_type == 'wgan-gp':
            return target_is_real
        if target_is_real:
            return torch.empty_like(input).fill_(self.real_label_val)
        else:
            return torch.empty_like(input).fill_(self.fake_label_val)

    def forward(self, input, target_is_real):
        target_label = self.get_target_label(input, target_is_real)
        loss = self.loss(input, target_label)
        return loss


class SemCKDLoss(nn.Module):
    """Cross-Layer Distillation with Semantic Calibration, AAAI2021"""

    def __init__(self):
        super(SemCKDLoss, self).__init__()
        self.crit = nn.MSELoss(reduction='none')

    def forward(self, s_value, f_target, weight):
        bsz, num_stu, num_tea = weight.shape
        ind_loss = torch.zeros(bsz, num_stu, num_tea).to(config.DEVICE)

        for i in range(num_stu):
            for j in range(num_tea):
                ind_loss[:, i, j] = self.crit(s_value[i][j], f_target[i][j]).reshape(bsz, -1).mean(-1)

        loss = (weight * ind_loss).sum() / (1.0 * bsz * num_stu)
        return loss

class FAKDLoss(nn.Module):
    def __init__(self):
        super(FAKDLoss,self).__init__()
        self.crit = nn.L1Loss()

    def forward(self,fea_s,fea_t):
        step = 0
        Affi_t = []
        Affi_s = []
        for feat_s in fea_s:
            feat_t = fea_t[step]
            batch = feat_s.shape[0]
            channel = feat_s.shape[1]
            fm_size = torch.tensor(feat_s.shape[2:])
            M = torch.prod(fm_size)
            F_t = feat_t.reshape(batch,channel,-1)
            F_s = feat_s.reshape(batch,channel,-1)
            Affi_t.append(torch.bmm(F_t.permute(0,2,1),F_t) / M)
            Affi_s.append(torch.bmm(F_s.permute(0,2,1),F_s) / M)
        loss_kd = self.crit(Affi_s[0],Affi_t[0]) + self.crit(Affi_s[1],Affi_t[1]) + self.crit(Affi_s[2],Affi_t[2])
        return loss_kd


class FA(nn.Module):
    def __init__(self):
        super(FA,self).__init__()

    def forward(self,fea_s,fea_t):
        Affi_t = []
        Affi_s = []
        for feat_s in fea_s:
            batch = feat_s.shape[0]
            channel = feat_s.shape[1]
            fm_size = torch.tensor(feat_s.shape[2:])
            M = torch.prod(fm_size)
            F_s = feat_s.reshape(batch,channel,-1)
            Affi_s.append(torch.bmm(F_s.permute(0,2,1),F_s) / M)
        for feat_t in fea_t:
            batch = feat_t.shape[0]
            channel = feat_t.shape[1]
            fm_size = torch.tensor(feat_t.shape[2:])
            M = torch.prod(fm_size)
            F_t = feat_t.reshape(batch,channel,-1)
            Affi_t.append(torch.bmm(F_t.permute(0, 2, 1), F_t) / M)
        return Affi_s,Affi_t
