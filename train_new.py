from __future__ import division
from __future__ import print_function
import albumentations as A
from albumentations.pytorch import ToTensorV2
import time
import argparse
import numpy as np
import itertools
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import make_grid, save_image

# from tensorboardX import SummaryWriter
from tqdm import tqdm
from copy import deepcopy

from utils import *
from models import *
from fid_score import *
from inception_score import *

from dataset import HorseZebraDataset
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--image_size', type=int, default=64, help='Size of image for discriminator input.')
# parser.add_argument('--initial_size', type=int, default=8, help='Initial size for generator.')
parser.add_argument('--patch_size', type=int, default=4, help='Patch size for generated image.')
parser.add_argument('--num_classes', type=int, default=1, help='Number of classes for discriminator.')
parser.add_argument('--lr_gen', type=float, default=4e-5, help='Learning rate for generator.')
parser.add_argument('--lr_dis', type=float, default=4e-5, help='Learning rate for discriminator.')
# parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay.')
parser.add_argument('--latent_dim', type=int, default=1024, help='Latent dimension.')
parser.add_argument('--n_critic', type=int, default=5, help='n_critic.')
parser.add_argument('--max_iter', type=int, default=500000, help='max_iter.')
parser.add_argument('--gener_batch_size', type=int, default=1, help='Batch size for generator.')
parser.add_argument('--dis_batch_size', type=int, default=32, help='Batch size for discriminator.')
parser.add_argument('--epoch', type=int, default=5000, help='Number of epoch.')
parser.add_argument('--output_dir', type=str, default='checkpoint', help='Checkpoint.')
parser.add_argument('--dim', type=int, default=384, help='Embedding dimension.')
parser.add_argument('--img_name', type=str, default="img_name", help='Name of pictures file.')
parser.add_argument('--optim', type=str, default="Adam", help='Choose your optimizer')
parser.add_argument('--loss', type=str, default="wgangp_eps", help='Loss function')
parser.add_argument('--phi', type=int, default="1", help='phi')
parser.add_argument('--beta1', type=int, default="0", help='beta1')
parser.add_argument('--beta2', type=float, default="0.99", help='beta2')
parser.add_argument('--lr_decay', type=str, default=True, help='lr_decay')
parser.add_argument('--diff_aug', type=str, default="translation,cutout,color", help='Data Augmentation')
parser.add_argument('--load_model', type=str, default=True)
parser.add_argument('--save_model', type=str, default=True)
parser.add_argument('--CHECKPOINT_GEN_H', type=str, default="saved_model/genh.pth.tar")
parser.add_argument('--CHECKPOINT_GEN_Z', type=str, default="saved_model/genz.pth.tar")
parser.add_argument('--CHECKPOINT_CRITIC_H', type=str, default="saved_model/critich.pth.tar")
parser.add_argument('--CHECKPOINT_CRITIC_Z', type=str, default="saved_model/criticz.pth.tar")

TRAIN_DIR = "data/train"
# VAL_DIR = "data/val"
NUM_WORKERS = 4

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"

device = torch.device(dev)
print("Device:", device)

args = parser.parse_args()

disc_H = Discriminator().to(device)
disc_Z = Discriminator().to(device)
gen_Z = Generator().to(device)
gen_H = Generator().to(device)
# print(disc_H)
# print(gen_H)

disc_H.apply(inits_weight)
disc_Z.apply(inits_weight)
gen_Z.apply(inits_weight)
gen_H.apply(inits_weight)

opt_gen = optim.Adam(
    itertools.chain(gen_Z.parameters(), gen_H.parameters()),
    # list(gen_Z.parameters()) + list(gen_H.parameters()),
    lr=args.lr_gen,
    betas=(args.beta1, args.beta2),
)
optimizer_D_H = torch.optim.Adam(disc_H.parameters(), lr=args.lr_dis, betas=(args.beta1, args.beta2))
optimizer_D_Z = torch.optim.Adam(disc_Z.parameters(), lr=args.lr_dis, betas=(args.beta1, args.beta2))

scheduler1 = torch.optim.lr_scheduler.StepLR(opt_gen, step_size=10, gamma=0.98)
scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer_D_H, step_size=10, gamma=0.98)
scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer_D_Z, step_size=10, gamma=0.98)
# gen_scheduler = LinearLrDecay(opt_gen, args.lr_gen, 0.0, 0, args.max_iter * args.n_critic)
# dis_scheduler = LinearLrDecay(opt_disc, args.lr_dis, 0.0, 0, args.max_iter * args.n_critic)
print("optim:", args.optim)


# writer=SummaryWriter()
# writer_dict = {'writer':writer}
# writer_dict["train_global_steps"]=0
# writer_dict["valid_global_steps"]=0

# def compute_gradient_penalty(D, real_samples, fake_samples, phi):
#     """Calculates the gradient penalty loss for WGAN GP"""
#     # Random weight term for interpolation between real and fake samples
#     alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(real_samples.get_device())
#     # Get random interpolation between real and fake samples
#     interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
#     # print("i", interpolates.shape)
#     d_interpolates = D(interpolates)
#     # print("d", d_interpolates.shape)
#     fake = torch.ones([real_samples.shape[0], 1], requires_grad=False).to(real_samples.get_device())
#     # print(fake.shape)
#     # Get gradient w.r.t. interpolates
#     gradients = torch.autograd.grad(
#         outputs=d_interpolates,
#         inputs=interpolates,
#         grad_outputs=fake,
#         create_graph=True,
#         retain_graph=True,
#         only_inputs=True,
#     )[0]
#     gradients = gradients.contiguous().view(gradients.size(0), -1)
#     gradient_penalty = ((gradients.norm(2, dim=1) - phi) ** 2).mean()
#     return gradient_penalty
#
#
# def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
#     """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028
#     Arguments:
#         netD (network)              -- discriminator network
#         real_data (tensor array)    -- real images
#         fake_data (tensor array)    -- generated images from the generator
#         device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
#         type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
#         constant (float)            -- the constant used in formula ( ||gradient||_2 - constant)^2
#         lambda_gp (float)           -- weight for this loss
#     Returns the gradient penalty loss
#     """
#     if lambda_gp > 0.0:
#         if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
#             interpolatesv = real_data
#         elif type == 'fake':
#             interpolatesv = fake_data
#         elif type == 'mixed':
#             alpha = torch.rand(real_data.shape[0], 1, device=device)
#             alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
#             interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
#         else:
#             raise NotImplementedError('{} not implemented'.format(type))
#         interpolatesv.requires_grad_(True)
#         disc_interpolates = netD(interpolatesv)
#         gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
#                                         grad_outputs=torch.ones(disc_interpolates.size()).to(device),
#                                         create_graph=True, retain_graph=True, only_inputs=True)
#         gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
#         gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
#         return gradient_penalty, gradients
#     else:
#         return 0.0, None

# def backward_D_basic(self, netD, real, fake):
#     """Calculate GAN loss for the discriminator
#
#     Parameters:
#         netD (network)      -- the discriminator D
#         real (tensor array) -- real images
#         fake (tensor array) -- images generated by a generator
#
#     Return the discriminator loss.
#     We also call loss_D.backward() to calculate the gradients.
#     """
#     # Real
#     pred_real = netD(real)
#     loss_D_real = self.criterionGAN(pred_real, True)
#     # Fake
#     pred_fake = netD(fake.detach())
#     loss_D_fake = self.criterionGAN(pred_fake, False)
#     # wgan-gp
#     gradient_penalty, gradients = networks.cal_gradient_penalty(
#         netD, real, fake, self.device, lambda_gp=10.0
#     )
#     gradient_penalty.backward(retain_graph=True)
#     # Combined loss and calculate gradients
#     loss_D = (loss_D_real + loss_D_fake) * 0.5
#     loss_D.backward()
#     return loss_D



L1 = nn.L1Loss()
mse = nn.MSELoss()

if args.load_model:
    load_checkpoint(
        args.CHECKPOINT_GEN_H, gen_H, opt_gen, args.lr_gen,
    )
    load_checkpoint(
        args.CHECKPOINT_GEN_Z, gen_Z, opt_gen, args.lr_gen,
    )
    load_checkpoint(
        args.CHECKPOINT_CRITIC_H, disc_H, optimizer_D_H, args.lr_dis,
    )
    load_checkpoint(
        args.CHECKPOINT_CRITIC_Z, disc_Z, optimizer_D_Z, args.lr_dis,
    )
transforms = A.Compose(
    [
        A.Resize(width=64, height=64),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2(),
    ],
    additional_targets={"image0": "image"},
)

dataset = HorseZebraDataset(
    root_horse=TRAIN_DIR + "/horses", root_zebra=TRAIN_DIR + "/zebras", transform=transforms
)
# val_dataset = HorseZebraDataset(
#     root_horse="cyclegan_test/horse1", root_zebra="cyclegan_test/zebra1", transform=transforms
# )
# val_loader = DataLoader(
#     val_dataset,
#     batch_size=1,
#     shuffle=False,
#     pin_memory=True,
# )
loader = DataLoader(
    dataset,
    batch_size=args.gener_batch_size,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True
)
# g_scaler = torch.cuda.amp.GradScaler()
# d_scaler = torch.cuda.amp.GradScaler()

LAMBDA_IDENTITY = 5
LAMBDA_CYCLE = 10


# def set_requires_grad(nets, requires_grad=False):
#     """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
#     Parameters:
#         nets (network list)   -- a list of networks
#         requires_grad (bool)  -- whether the networks require gradients or not
#     """
#     if not isinstance(nets, list):
#         nets = [nets]
#     for net in nets:
#         if net is not None:
#             for param in net.parameters():
#                 param.requires_grad = requires_grad


def train_fn(disc_H, disc_Z, gen_Z, gen_H, loader, opt_gen,optimizer_D_H,optimizer_D_Z,l1, mse, epoch):
    # writer = writer_dict['writer']
    loop = tqdm(loader, leave=True)

    for idx, (zebra, horse) in enumerate(loop):
        # global_steps = writer_dict['train_global_steps']
        zebra = zebra.to(device)
        horse = horse.to(device)

        opt_gen.zero_grad()

        # # identity loss (remove these for efficiency if you set lambda_identity=0)
        # identity_zebra = gen_Z(zebra)
        # identity_horse = gen_H(horse)
        # identity_zebra_loss = l1(zebra, identity_zebra)
        # identity_horse_loss = l1(horse, identity_horse)

        # if idx % args.n_critic == 0:
        fake_horse = gen_H(zebra)
        fake_zebra = gen_Z(horse)
        D_H_fake = disc_H(fake_horse)
        D_Z_fake = disc_Z(fake_zebra)
        loss_G_H = mse(D_H_fake, torch.ones_like(D_H_fake))
        loss_G_Z = mse(D_Z_fake, torch.ones_like(D_Z_fake))

        # cycle loss
        cycle_zebra = gen_Z(fake_horse)
        cycle_horse = gen_H(fake_zebra)
        cycle_zebra_loss = l1(zebra, cycle_zebra)
        cycle_horse_loss = l1(horse, cycle_horse)



        # add all togethor
        G_loss = (
                loss_G_Z
                + loss_G_H
                + cycle_zebra_loss * LAMBDA_CYCLE
                + cycle_horse_loss * LAMBDA_CYCLE
                # + identity_horse_loss * LAMBDA_IDENTITY
                # + identity_zebra_loss * LAMBDA_IDENTITY
        )
        # print(loss_G_Z,loss_G_H,cycle_zebra_loss,cycle_horse_loss)

        G_loss.backward()
        opt_gen.step()


        ###### Discriminator A ######
        # with torch.cuda.amp.autocast():
        optimizer_D_H.zero_grad()
        # fake_horse = gen_H(zebra)
        D_H_real = disc_H(horse)
        D_H_fake = disc_H(fake_horse.detach())
        D_H_real_loss = mse(D_H_real, torch.ones_like(D_H_real))
        D_H_fake_loss = mse(D_H_fake, torch.zeros_like(D_H_fake))
        D_H_loss = (D_H_real_loss + D_H_fake_loss)*0.5
        D_H_loss.backward()
        optimizer_D_H.step()


        ###### Discriminator B ######
        optimizer_D_Z.zero_grad()
        # fake_zebra = gen_Z(horse)
        D_Z_real = disc_Z(zebra)
        D_Z_fake = disc_Z(fake_zebra.detach())
        D_Z_real_loss = mse(D_Z_real, torch.ones_like(D_Z_real))
        D_Z_fake_loss = mse(D_Z_fake, torch.zeros_like(D_Z_fake))
        D_Z_loss = (D_Z_real_loss + D_Z_fake_loss)*0.5
        D_Z_loss.backward()
        optimizer_D_Z.step()


        # gradient_penalty1 = compute_gradient_penalty(disc_H, horse, fake_horse.detach(), args.phi)
        # gradient_penalty2 = compute_gradient_penalty(disc_Z, zebra, fake_zebra.detach(), args.phi)

        # alpha = torch.rand(horse.size(0), 1, 1, 1).to(device)
        # x_hat = (alpha * horse.data + (1 - alpha) * fake_horse.data).requires_grad_(True)
        # out_src = disc_H(x_hat)
        # d_loss_gp1 = gradient_penalty(out_src, x_hat)
        #
        # alpha = torch.rand(zebra.size(0), 1, 1, 1).to(device)
        # x_hat = (alpha * zebra.data + (1 - alpha) * fake_zebra.data).requires_grad_(True)
        # out_src = disc_Z(x_hat)
        # d_loss_gp2 = gradient_penalty(out_src, x_hat)

        # put it togethor
        # D_loss = (D_H_loss + D_Z_loss) *0.5
        # print(D_loss, D_H_loss, D_Z_loss, gradient_penalty1, gradient_penalty2)

        # optimizer_D_A.zero_grad()
        # d_scaler.scale(D_H_loss).backward()
        # d_scaler.step(optimizer_D_A)
        # optimizer_D_B.zero_grad()
        # d_scaler.scale(D_Z_loss).backward()
        # d_scaler.step(optimizer_D_B)
        # d_scaler.update()
        # writer.add_scalar("D_loss", D_loss.item(), global_steps)



        # if idx % 200 == 0:
        #     save_image(fake_horse * 0.5 + 0.5, f"saved_images/horse_{epoch}_{idx}.png")
        #     save_image(fake_zebra * 0.5 + 0.5, f"saved_images/zebra_{epoch}_{idx}.png")
        #
        if idx % 200 == 0:
            fimgs_H = fake_horse[:25]
            sample_imgs_H = torch.cat((fimgs_H,zebra[:25]),0)
            fimgs_Z = fake_zebra[:25]
            sample_imgs_Z = torch.cat((fimgs_Z,horse[:25]),0)

            # img_grid = make_grid(sample_imgs, nrow=5, normalize=True, scale_each=True)
            save_image(sample_imgs_H, f'saved_images/horse750/generated_img_H{epoch}_{idx % len(loop)}.jpg', nrow=5,
                       normalize=True, scale_each=True)
            save_image(sample_imgs_Z, f'saved_images/zebra750/generated_img_Z{epoch}_{idx % len(loop)}.jpg', nrow=5,
                       normalize=True, scale_each=True)
            tqdm.write("[Epoch %d] [Batch %d/%d] [DH loss: %f] [DZ loss: %f] [G loss: %f]" %
                       (epoch, idx % len(loop), len(loop), D_H_loss.item(),D_Z_loss.item(), G_loss.item()))


# def validate(generator, writer_dict, fid_stat):
#
#     writer = writer_dict['writer']
#     global_steps = writer_dict['valid_global_steps']
#
#     generator = generator.eval()
#     fid_score = get_fid(fid_stat, epoch, generator, num_img=5000, val_batch_size=60 * 2, latent_dim=1024,
#                         writer_dict=None, cls_idx=None)
#
#     print(f"FID score: {fid_score}")
#
#     writer.add_scalar('FID_score', fid_score, global_steps)
#
#     writer_dict['valid_global_steps'] = global_steps + 1
#     return fid_score


def main():
    best = 1e4
    for epoch in range(751,args.epoch):
        # lr_schedulers = (gen_scheduler, dis_scheduler) if args.lr_decay else None

        train_fn(disc_H, disc_Z, gen_Z, gen_H, loader, opt_gen,optimizer_D_H,optimizer_D_Z, L1, mse, epoch)
        scheduler1.step()
        scheduler2.step()
        scheduler3.step()
        checkpoint = {'epoch': epoch}
        checkpoint['generator_state_Z'] = gen_Z.state_dict()
        checkpoint['generator_state_H'] = gen_H.state_dict()
        checkpoint['discriminator_state_Z'] = disc_Z.state_dict()
        checkpoint['discriminator_state_H'] = disc_H.state_dict()

        # score = validate(gen_Z, writer_dict, fid_stat)

        if args.save_model and epoch % 50 == 0 and epoch != 0:
            save_checkpoint(gen_H, opt_gen, filename=args.CHECKPOINT_GEN_H + str(epoch))
            save_checkpoint(gen_Z, opt_gen, filename=args.CHECKPOINT_GEN_Z + str(epoch))
            save_checkpoint(disc_H, optimizer_D_H, filename=args.CHECKPOINT_CRITIC_H + str(epoch))
            save_checkpoint(disc_Z, optimizer_D_Z, filename=args.CHECKPOINT_CRITIC_Z + str(epoch))


if __name__ == "__main__":
    main()
