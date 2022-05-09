import os
from torch import nn
import torch
from models.model import Generator,Discriminator,initialize_weights
from models.discriminator_vgg_arch import Discriminator_VGG_128,Discriminator_VGG_256,NLayerDiscriminator
from torch import optim
from dataset import ImageDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from loss import VGGLoss
import config
from utils import train_one_epoch,evaluate

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
    gen = Generator(nb=6).to(config.DEVICE)
    # disc = Discriminator(in_channels=3).to(config.DEVICE)
    # disc = Discriminator_VGG_256(in_nc=3,nf=64).to(config.DEVICE)
    disc = NLayerDiscriminator(input_nc=3).to(config.DEVICE)
    # initialize_weights(gen)
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.9, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.9, 0.999))
    scheduler_G = torch.optim.lr_scheduler.StepLR(opt_gen, 100, gamma=0.5, last_epoch=-1)
    scheduler_D = torch.optim.lr_scheduler.StepLR(opt_disc, 100, gamma=0.5, last_epoch=-1)
    tb_writer = SummaryWriter(comment=config.comment)
    tb_step = 0
    gen.train()
    disc.train()
    vgg_loss = VGGLoss()
    last_loss,last_content,last_gram = 100,100,100

    config.loader_model(gen,disc)

    for epoch in range(config.last_EPOCH,config.EPOCHS):
        # train
        loss_D, loss_G, loss_content,loss_adv,loss_gram,tb_step = train_one_epoch(
            gen,disc,dataloader,opt_gen,opt_disc,vgg_loss,epoch,tb_step)
        if tb_step > config.warmup:
            scheduler_G.step()
            scheduler_D.step()
        # val
        loss_val, psnr, ssim = evaluate(gen,dataloader_val,epoch,tb_step)

        tags = ["loss_D", "loss_G", "loss_content", "loss_adv","loss_gram","val_loss", "learning_rate", "psnr", "ssim"]
        tb_writer.add_scalar(tags[0], loss_D, epoch)
        tb_writer.add_scalar(tags[1], loss_G, epoch)
        tb_writer.add_scalar(tags[2], loss_content, epoch)
        tb_writer.add_scalar(tags[3], loss_adv, epoch)
        tb_writer.add_scalar(tags[4], loss_gram, epoch)
        tb_writer.add_scalar(tags[5], loss_val, epoch)
        tb_writer.add_scalar(tags[6], opt_gen.param_groups[0]["lr"], epoch)
        tb_writer.add_scalar(tags[7], psnr, epoch)
        tb_writer.add_scalar(tags[8], ssim, epoch)

        # if loss_val < last_loss or loss_content < last_content or loss_gram < last_gram:
        if epoch % 20 == 0:
            torch.save(gen.state_dict(), "saved_models/normal/" + config.time + '/generator' + str(epoch) + '.pth')
            torch.save(disc.state_dict(), "saved_models/normal/" + config.time + '/discriminator.pth')
        last_loss,last_content,last_gram = loss_val,loss_content,loss_gram


if __name__ == "__main__":
    os.makedirs('./saved_models/'+config.model+'/' +config.time,exist_ok=True)
    os.makedirs('./image/'+config.model+'/' + config.time, exist_ok=True)
    main()