import torch
from tqdm import tqdm
from torch import nn
import config
import sys
from torch.autograd import Variable
from torchvision.utils import save_image
from loss import CharbonnierLoss,GANLoss


l1 = nn.L1Loss().to(config.DEVICE)
l_pix = CharbonnierLoss().to(config.DEVICE)
loss_cri = GANLoss(config.gan_model)
def get_static(n):
    accu=[]
    for _ in range(n):
        accu.append(torch.zeros(1).to(config.DEVICE))
    return accu

def train_one_epoch(gen,disc,dataloader,opt_gen,opt_disc,vgg_loss,epoch,tb_step):

    accu_G,accu_D,accu_content,accu_gram,accu_adv = get_static(5)

    dataloader = tqdm(dataloader, file=sys.stdout)

    for i, imgs in enumerate(dataloader):
        # 配置模型的输入
        imgs_lr = Variable(imgs["lr"].type(torch.cuda.FloatTensor)).to(config.DEVICE)
        imgs_hr = Variable(imgs["hr"].type(torch.cuda.FloatTensor)).to(config.DEVICE)

        # 生成高分辨率图像
        gen_hr = gen(imgs_lr)
        if tb_step < config.warmup:
            loss_pixel = l_pix(gen_hr, imgs_hr)
            opt_gen.zero_grad()
            loss_pixel.backward()
            opt_gen.step()
            dataloader.desc = "[train epoch {}] loss_pixel:{:.7f}"\
                .format(epoch,loss_pixel)
            tb_step += 1
            continue

        """更新D"""
        opt_disc.zero_grad()
        if config.gan_model == 'wgan-gp':
            pred_real = disc(imgs_hr).mean()
            pred_fake = disc(gen_hr.detach()).mean()
            loss_Ra = -(pred_real - pred_fake)
            pg = config.cal_gradient_penalty(disc, imgs_hr, gen_hr.detach(), config.DEVICE)
            loss_D = loss_Ra + pg
        elif config.gan_model =='ragan':
            pred_real = disc(imgs_hr)
            pred_fake = disc(gen_hr.detach())
            l_d_real = loss_cri(pred_real - torch.mean(pred_fake),True)
            l_d_fake = loss_cri(pred_fake - torch.mean(pred_real),False)
            loss_D = (l_d_real + l_d_fake) / 2

        loss_D.backward()
        opt_disc.step()
        opt_disc.zero_grad()
        accu_D += loss_D.detach()

        """更新G"""
        for p in disc.parameters():
            p.require_grad = False
        opt_gen.zero_grad()
        if config.gan_model == 'wgan-gp':
            pred_fake = disc(gen_hr).mean()
            loss_gan = - (pred_fake * config.lambda_adv)
        elif config.gan_model == 'ragan':
            pred_fake = disc(gen_hr)
            pred_real = disc(imgs_hr).detach()
            l_g_real = loss_cri(pred_real - torch.mean(pred_fake),False)
            l_g_fake = loss_cri(pred_fake - torch.mean(pred_real),True)
            loss_gan = (l_g_fake + l_g_real) / 2 * config.lambda_adv

        gen_edge = config.edge_conv2d(gen_hr, config.DEVICE)
        real_edge = config.edge_conv2d(imgs_hr, config.DEVICE)
        loss_pixel = l_pix(gen_hr, imgs_hr) * config.lambda_pixel
        loss_edge = l1(gen_edge, real_edge) * config.lambda_edge

        loss_content,loss_gram = vgg_loss(gen_hr,imgs_hr)
        loss_gram = config.lambda_gram * loss_gram

        loss_G = loss_content + loss_pixel + loss_edge + loss_gram + loss_gan
        loss_G.backward()
        opt_gen.step()
        opt_gen.zero_grad()

        for p in disc.parameters():
            p.require_grad = True

        accu_G += loss_G.detach()
        accu_content += loss_content.detach()
        accu_adv += loss_gan.detach()
        accu_gram += loss_gram.detach()
        tb_step += 1
        dataloader.desc = "[train epoch {}] loss_D:{:.3f} loss_G: {:.3f}, loss_content:{:.3f}, loss_gram:{:.3f},loss_adv:{:.3f},lr:{:.7f}".format(
            epoch,
            accu_D.item() / (i + 1),
            accu_G.item() / (i + 1),
            accu_content.item() / (i + 1),
            accu_gram.item() / (i + 1),
            accu_adv.item() / (i + 1),
            opt_gen.state_dict()['param_groups'][0]['lr'])

    return accu_D.item()/(i+1), accu_G.item()/(i+1), accu_content.item()/(i+1) ,accu_adv.item()/(i+1),accu_gram.item()/(i+1),tb_step

@torch.no_grad()
def evaluate(model, dataloader, epoch,tb_step):
    model.eval()
    accu_loss,accu_psnr,accu_ssim = get_static(3)
    dataloader = tqdm(dataloader, file=sys.stdout)
    ssim_loss = config.SSIM()
    for i, imgs in enumerate(dataloader):
        # 配置模型的输
        imgs_lr = Variable(imgs["lr"].type(torch.cuda.FloatTensor)).to(config.DEVICE)
        imgs_hr = Variable(imgs["hr"].type(torch.cuda.FloatTensor)).to(config.DEVICE)

        # 生成高分辨率图像
        if config.model == "distilling":
            gen_hr,_ = model(imgs_lr)
        else:
            gen_hr = model(imgs_lr)
        loss = l_pix(gen_hr, imgs_hr)
        psnr = config.cal_psnr(imgs_hr,gen_hr)

        ssim = ssim_loss(imgs_hr,gen_hr)
        accu_ssim += ssim
        accu_psnr += psnr
        accu_loss += loss.detach()

        dataloader.desc = "[val epoch {}] loss_pixel: {:.3f} psnr:{:.3f} ssim:{:.3f}".format(epoch,
                                            accu_loss.item() / (i + 1), accu_psnr.item() / (i + 1), accu_ssim.item() / (i + 1))
        if i == 0:
            img_grid = torch.cat((gen_hr, imgs_hr), -1)
            save_image(img_grid, "image/"+config.model+"/"+config.time +"/"+ str(tb_step)+'.png', nrow=4, normalize=False)
            del img_grid

    return accu_loss.item() / (i + 1), accu_psnr.item() / (i + 1),accu_ssim.item() / (i + 1)
