import torch
from tqdm import tqdm
from torch import nn
import config
import sys
from torch.autograd import Variable
from torchvision.utils import save_image
from loss import CharbonnierLoss,GANLoss
from loss import SemCKDLoss


l1 = nn.L1Loss().to(config.DEVICE)
l_pix = CharbonnierLoss().to(config.DEVICE)
loss_cri = GANLoss(config.gan_model)
criterion_kd = SemCKDLoss().to(config.DEVICE)
def get_static(n):
    accu=[]
    for _ in range(n):
        accu.append(torch.zeros(1).to(config.DEVICE))
    return accu

def train_distill(model_s,model_t,SelfA,disc,dataloader,opt_gen,opt_disc,vgg_loss,epoch,tb_step):

    accu_G,accu_D,accu_content,accu_gram,accu_adv,accu_KD = get_static(6)

    dataloader = tqdm(dataloader, file=sys.stdout)

    for i, imgs in enumerate(dataloader):
        # 配置模型的输入
        imgs_lr = Variable(imgs["lr"].type(torch.cuda.FloatTensor)).to(config.DEVICE)
        imgs_hr = Variable(imgs["hr"].type(torch.cuda.FloatTensor)).to(config.DEVICE)

        if imgs_lr.shape[0] != config.BATCH_SIZE:
            continue

        # 生成高分辨率图像
        fake_s,fea_s = model_s(imgs_lr)

        if tb_step < config.warmup:
            opt_gen.zero_grad()
            loss_pixel = l_pix(fake_s, imgs_hr)
            loss_pixel.backward()
            opt_gen.step()
            dataloader.desc = "[train epoch {}] loss_pixel:{:.3f}".format(
                epoch,
                loss_pixel)
            tb_step += 1
            continue

        with torch.no_grad():
            fake_t,fea_t = model_t(imgs_lr)
            fea_t = [f.detach() for f in fea_t]

        """更新D"""
        if tb_step > config.D_warmup:
            opt_disc.zero_grad()
            if config.gan_model == 'wgan-gp':
                pred_real = disc(imgs_hr).mean()
                pred_fake = disc(fake_s.detach()).mean()
                loss_Ra = -(pred_real - pred_fake)
                pg = config.cal_gradient_penalty(disc, imgs_hr, fake_s.detach(), config.DEVICE)
                loss_D = loss_Ra + pg
            elif config.gan_model == 'ragan':
                pred_real = disc(imgs_hr)
                pred_fake = disc(fake_s.detach())
                l_d_real = loss_cri(pred_real - torch.mean(pred_fake), True)
                l_d_fake = loss_cri(pred_fake - torch.mean(pred_real), False)
                loss_D = (l_d_real + l_d_fake) / 2

            loss_D.backward()
            opt_disc.step()
            opt_disc.zero_grad()
            accu_D += loss_D.detach()

        """更新G"""
        for p in disc.parameters():
            p.require_grad = False
        opt_gen.zero_grad()

        #KD
        s_value, f_target, weight = SelfA(fea_s[1:-1], fea_t[1:-1])
        loss_kd = criterion_kd(s_value, f_target, weight) * config.lambda_KD

        gen_edge = config.edge_conv2d(fake_s, config.DEVICE)
        real_edge = config.edge_conv2d(imgs_hr, config.DEVICE)
        loss_pixel = l_pix(fake_s, imgs_hr) * config.lambda_pixel
        loss_edge = l1(gen_edge, real_edge) * config.lambda_edge

        loss_content,loss_gram = vgg_loss(fake_s,imgs_hr)
        loss_gram = config.lambda_gram * loss_gram

        loss_G = loss_pixel + loss_edge  + loss_kd + loss_content + loss_gram
        if tb_step > config.D_warmup:
            if config.gan_model == 'wgan-gp':
                pred_fake = disc(fake_s).mean()
                loss_gan = - (pred_fake * config.lambda_adv)
            elif config.gan_model == 'ragan':
                pred_fake = disc(fake_s)
                pred_real = disc(imgs_hr).detach()
                l_g_real = loss_cri(pred_real - torch.mean(pred_fake), False)
                l_g_fake = loss_cri(pred_fake - torch.mean(pred_real), True)
                loss_gan = (l_g_fake + l_g_real) / 2 * config.lambda_adv
            accu_adv += loss_gan.detach()
            loss_G += loss_gan

        loss_G.backward()
        opt_gen.step()
        opt_gen.zero_grad()

        for p in disc.parameters():
            p.require_grad = True

        accu_G += loss_G.detach()
        accu_content += loss_content.detach()
        accu_KD += loss_kd.detach()
        accu_gram += loss_gram.detach()
        tb_step += 1
        dataloader.desc = "[train epoch {}] loss_D:{:.3f} loss_G: {:.3f}, loss_content:{:.3f}, loss_gram:{:.3f},loss_adv:{:.3f}.loss_KD:{:.3f},lr:{:.7f}".format(
            epoch,
            accu_D.item() / (i + 1),
            accu_G.item() / (i + 1),
            accu_content.item() / (i + 1),
            accu_gram.item() / (i + 1),
            accu_adv.item() / (i + 1),
            accu_KD.item() / (i + 1),
            opt_gen.state_dict()['param_groups'][0]['lr'])

    return accu_D.item()/(i+1), accu_G.item()/(i+1), accu_content.item()/(i+1) ,accu_adv.item()/(i+1),accu_gram.item()/(i+1),accu_KD.item()/(i+1),tb_step

def train_fakd(model_s,model_t,disc,Fakd,vgg_loss,dataloader,opt_gen,opt_disc,epoch,tb_step):

    accu_G,accu_content,accu_s,accu_KD,accu_gram = get_static(5)

    dataloader = tqdm(dataloader, file=sys.stdout)
    ssim_loss = config.SSIM()

    for i, imgs in enumerate(dataloader):
        # 配置模型的输入
        imgs_lr = Variable(imgs["lr"].type(torch.cuda.FloatTensor)).to(config.DEVICE)
        imgs_hr = Variable(imgs["hr"].type(torch.cuda.FloatTensor)).to(config.DEVICE)

        if imgs_lr.shape[0] != config.BATCH_SIZE:
            continue

        # 生成高分辨率图像
        fake_s,fea_s = model_s(imgs_lr)

        with torch.no_grad():
            fake_t,fea_t = model_t(imgs_lr)
            fea_t = [f.detach() for f in fea_t]

        """更新D"""
        if tb_step >= config.D_warmup:
            opt_disc.zero_grad()
            if config.gan_model == 'wgan-gp':
                pred_real = disc(imgs_hr).mean()
                pred_fake = disc(fake_s.detach()).mean()
                loss_Ra = -(pred_real - pred_fake)
                pg = config.cal_gradient_penalty(disc, imgs_hr, fake_s.detach(), config.DEVICE)
                loss_D = loss_Ra + pg
            elif config.gan_model == 'ragan':
                pred_real = disc(imgs_hr)
                pred_fake = disc(fake_s.detach())
                l_d_real = loss_cri(pred_real - torch.mean(pred_fake), True)
                l_d_fake = loss_cri(pred_fake - torch.mean(pred_real), False)
                loss_D = (l_d_real + l_d_fake) / 2
        loss_D.backward()
        opt_disc.step()
        opt_disc.zero_grad()
        """更新G"""
        for p in disc.parameters():
            p.require_grad = False
        opt_gen.zero_grad()

        #KD
        loss_kd = Fakd(fea_s,fea_t)
        loss_s = l_pix(fake_s, imgs_hr) * config.lambda_pixel
        loss_content,loss_gram = vgg_loss(fake_s,imgs_hr)
        loss_gram = config.lambda_gram * loss_gram


        loss_G =  loss_s + loss_kd + loss_content + loss_gram

        if tb_step > config.D_warmup:
            if config.gan_model == 'wgan-gp':
                pred_fake = disc(fake_s).mean()
                loss_gan = - (pred_fake * config.lambda_adv)
            elif config.gan_model == 'ragan':
                pred_fake = disc(fake_s)
                pred_real = disc(imgs_hr).detach()
                l_g_real = loss_cri(pred_real - torch.mean(pred_fake), False)
                l_g_fake = loss_cri(pred_fake - torch.mean(pred_real), True)
                loss_gan = (l_g_fake + l_g_real) / 2 * config.lambda_adv
            loss_G += loss_gan

        loss_G.backward()
        opt_gen.step()
        opt_gen.zero_grad()

        accu_G += loss_G.detach()
        accu_s += loss_s.detach()
        accu_content += loss_content.detach()
        accu_gram += loss_gram.detach()
        accu_KD += loss_kd.detach()
        tb_step += 1
        for p in disc.parameters():
            p.require_grad = True
        dataloader.desc = "[train epoch {}]loss_G: {:.3f}, loss_content:{:.3f},loss_gram:{:.3f}, loss_student:{:.3f},loss_kd:{:.3f}lr:{:.7f}".format(
            epoch,
            accu_G.item() / (i + 1),
            accu_content.item() / (i + 1),
            accu_gram.item() / (i + 1),
            accu_s.item() / (i + 1),
            accu_KD.item() / (i + 1),
            opt_gen.state_dict()['param_groups'][0]['lr'])

    return accu_content.item()/(i+1),accu_gram.item()/(i+1),accu_KD.item()/(i+1),tb_step


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
        if config.model == "distill" or config.model == 'fakd':
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
        if i == 10:
            img_grid = torch.cat((gen_hr, imgs_hr), -1)
            save_image(img_grid, "image/{}/{}/{}.png".format(config.model,config.time,str(tb_step)), nrow=4, normalize=False)
            del img_grid

    return accu_loss.item() / (i + 1), accu_psnr.item() / (i + 1),accu_ssim.item() / (i + 1)