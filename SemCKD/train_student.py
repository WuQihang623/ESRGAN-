import os
from torch import nn
import torch
from models.models import Generator
from models.RRDBNet import RRDBNet
from models.discriminator_vgg_arch import Discriminator_VGG_128,Discriminator_VGG_256,NLayerDiscriminator
from models.SelfA import SelfA
from torch import optim
from datasets import ImageDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from loss import VGGLoss
import config
from utils import train_distill,evaluate

torch.backends.cudnn.benchmark = True

def main():
    dataloader = DataLoader(
        ImageDataset("../../data/train_256", hr_shape=config.HIGH_RES),
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=8,
    )
    dataloader_val = DataLoader(
        ImageDataset("../../data/val_256", hr_shape=config.HIGH_RES),
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=8,
    )
    model_t = Generator(3,3,64,23,32,[f for f in range(0,23,2)]).to(config.DEVICE)
    model_s = Generator(3,3,64,6,32,[f for f in range(6)]).to(config.DEVICE)
    # disc = Discriminator_VGG_256(in_nc=3,nf=64).to(config.DEVICE)
    disc = NLayerDiscriminator(input_nc=3).to(config.DEVICE)

    data = torch.randn(2,3,config.HIGH_RES//4,config.HIGH_RES//4).to(config.DEVICE)
    model_s.eval()
    model_t.eval()
    _,feat_s = model_s(data)
    _,feat_t = model_t(data)
    s_n = [f.shape[1] for f in feat_s[1:-1]]
    t_n = [f.shape[1] for f in feat_t[1:-1]]

    self_attention = SelfA(len(feat_s) - 2, len(feat_t) - 2, config.BATCH_SIZE, s_n, t_n).to(config.DEVICE)

    opt_student = optim.Adam(model_s.parameters(), lr=config.LEARNING_RATE, betas=(0.9, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.9, 0.999))
    scheduler_G = torch.optim.lr_scheduler.StepLR(opt_student, 30, gamma=0.5, last_epoch=-1)
    scheduler_D = torch.optim.lr_scheduler.StepLR(opt_disc, 30, gamma=0.5, last_epoch=-1)
    tb_writer = SummaryWriter(comment=config.comment)
    tb_step = 0
    model_s.train()
    disc.train()
    vgg_loss = VGGLoss()
    last_loss,last_content,last_gram = 100,100,100

    config.loader_model(model_s,model_t,disc)

    for epoch in range(config.EPOCHS):
        # train
        loss_D, loss_G, loss_content,loss_adv,loss_gram,loss_kd,tb_step = train_distill(
            model_s,model_t,self_attention,disc,dataloader,opt_student,opt_disc,vgg_loss,epoch,tb_step)
        scheduler_G.step()
        scheduler_D.step()
        # val
        loss_val, psnr, ssim = evaluate(model_s,dataloader_val,epoch,tb_step)

        tags = ["loss_D", "loss_G", "loss_content", "loss_adv","loss_gram","loss_kd","val_loss", "learning_rate", "psnr", "ssim"]
        tb_writer.add_scalar(tags[0], loss_D, epoch)
        tb_writer.add_scalar(tags[1], loss_G, epoch)
        tb_writer.add_scalar(tags[2], loss_content, epoch)
        tb_writer.add_scalar(tags[3], loss_adv, epoch)
        tb_writer.add_scalar(tags[4], loss_gram, epoch)
        tb_writer.add_scalar(tags[5], loss_kd, epoch)
        tb_writer.add_scalar(tags[6], loss_val, epoch)
        tb_writer.add_scalar(tags[7], opt_student.param_groups[0]["lr"], epoch)
        tb_writer.add_scalar(tags[8], psnr, epoch)
        tb_writer.add_scalar(tags[9], ssim, epoch)

        # if loss_val < last_loss or loss_content < last_content or loss_gram < last_gram:
        if epoch % 5 == 0:
            torch.save(model_s.state_dict(),
                           "saved_models/{}/{}/generator{}.pth".format(config.model, config.time, epoch))
            torch.save(disc.state_dict(),
                       "saved_models/{}/{}/disc.pth".format(config.model, config.time))
        if loss_val < last_loss:
            last_loss = loss_val
        if last_content < loss_content:
            last_content = loss_content
        if last_gram < loss_gram:
            last_gram = loss_gram

if __name__ == "__main__":
    os.makedirs('./saved_models/'+config.model+'/' +config.time,exist_ok=True)
    os.makedirs('./image/'+config.model+'/' + config.time, exist_ok=True)
    main()