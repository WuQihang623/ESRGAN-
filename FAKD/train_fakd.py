import os
from torch import nn
import torch
from models.models import Generator
from models.RRDBNet import RRDBNet
from models.discriminator_vgg_arch import Discriminator_VGG_128,Discriminator_VGG_256,NLayerDiscriminator
from torch import optim
from datasets import ImageDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from loss import VGGLoss,FAKDLoss
import config
from utils import train_fakd,evaluate

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
    model_t = Generator(3,3,64,23,32,[5,13,22]).to(config.DEVICE)
    model_s = Generator(3,3,64,6,32,[1,3,5]).to(config.DEVICE)
    # disc = Discriminator_VGG_256(in_nc=3,nf=64).to(config.DEVICE)

    opt_student = optim.Adam(model_s.parameters(), lr=config.LEARNING_RATE, betas=(0.9, 0.999))
    scheduler_G = torch.optim.lr_scheduler.StepLR(opt_student, 30, gamma=0.5, last_epoch=-1)
    disc = NLayerDiscriminator(input_nc=3).to(config.DEVICE)
    tb_writer = SummaryWriter(comment=config.comment)
    tb_step = 0
    model_s.train()
    model_t.eval()
    disc.train()
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.9, 0.999))
    scheduler_D = torch.optim.lr_scheduler.StepLR(opt_disc, 30, gamma=0.5, last_epoch=-1)

    Fakd = FAKDLoss().to(config.DEVICE)
    vgg_loss = VGGLoss().to(config.DEVICE)
    last_loss = 100

    config.loader_model(model_s,model_t)

    for epoch in range(config.EPOCHS):
        # train
        loss_content,loss_gram,loss_kd,tb_step = train_fakd(
            model_s,model_t,disc,Fakd,vgg_loss,dataloader,opt_student,opt_disc,epoch,tb_step)
        scheduler_G.step()
        scheduler_D.step()

        # val
        loss_val, psnr, ssim = evaluate(model_s,dataloader_val,epoch,tb_step)

        tags = ["loss_content", "loss_gram","loss_kd","val_loss", "learning_rate", "psnr", "ssim"]
        tb_writer.add_scalar(tags[0], loss_content, epoch)
        tb_writer.add_scalar(tags[1], loss_gram, epoch)
        tb_writer.add_scalar(tags[2], loss_kd, epoch)
        tb_writer.add_scalar(tags[3], loss_val, epoch)
        tb_writer.add_scalar(tags[4], opt_student.param_groups[0]["lr"], epoch)
        tb_writer.add_scalar(tags[5], psnr, epoch)
        tb_writer.add_scalar(tags[6], ssim, epoch)

        # if loss_val < last_loss:
        if epoch%5 == 0:
            torch.save(model_s.state_dict(), "saved_models/{}/{}/generator{}.pth".format(config.model,config.time,epoch))
        if loss_val < last_loss:
            last_loss = loss_val

if __name__ == "__main__":
    os.makedirs('./saved_models/{}/{}'.format(config.model,config.time),exist_ok=True)
    os.makedirs('./image/{}/{}'.format(config.model,config.time), exist_ok=True)
    main()