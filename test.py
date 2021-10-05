import torch
from utils import *
from models import *
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataset import HorseZebraDataset
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

TRAIN_DIR = "data/train"
VAL_DIR = "data/val"

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"

device = torch.device(dev)
print("Device:", device)

gen_Z = Generator().to(device)
gen_H = Generator().to(device)
disc_H = Discriminator().to(device)

def load_cp(checkpoint_file, model):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location="cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()


transforms = A.Compose(
    [
        A.Resize(width=64, height=64),
        # A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2(),
    ],
    additional_targets={"image0": "image"},
)

dataset = HorseZebraDataset(
    # root_horse=VAL_DIR + "/horses", root_zebra=VAL_DIR + "/zebras", transform=transforms
    root_horse=TRAIN_DIR + "/horses", root_zebra=TRAIN_DIR + "/zebras", transform=transforms
)

load_cp(
    "saved_model/genh.pth.tar", gen_H
)
load_cp(
    "saved_model/genz.pth.tar", gen_Z
)
load_cp(
    "saved_model/critich.pth.tar", disc_H
)

pics = 25
concatenates = True

loader = DataLoader(
    dataset,
    batch_size=pics,
    shuffle=False,
    num_workers=1,
    pin_memory=True
)

def test():
    with torch.no_grad():
        for i, (zebra, horse) in enumerate(loader):
            # Prepare input images.
            zebra = zebra.to(device)
            horse = horse.to(device)

            # Translate images.
            fake_horse = gen_H(zebra)
            fake_zebra = gen_Z(horse)



            fimgs_H = fake_horse[:pics]
            fimgs_Z = fake_zebra[:pics]
            if concatenates == True:
                fimgs_H = torch.cat((zebra[:pics], fimgs_H), 3)
                fimgs_Z = torch.cat((horse[:pics], fimgs_Z), 3)

            save_image(fimgs_H, f'saved_images/test_horse/generated_img_H{i}.jpg', nrow=5,
                       normalize=True, scale_each=True)
            save_image(fimgs_Z, f'saved_images/test_zebra/generated_img_Z{i}.jpg', nrow=5,
                       normalize=True, scale_each=True)
            # save_image(horse, f'saved_images/real_horse/generated_img_H{i}.jpg', nrow=5,
            #            normalize=True, scale_each=True)
            # save_image(zebra, f'saved_images/real_zebra/generated_img_Z{i}.jpg', nrow=5,
            #            normalize=True, scale_each=True)
def count_parameters(model):
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))


if __name__ == "__main__":
    test()
    # count_parameters(gen_H)
    # count_parameters(disc_H)