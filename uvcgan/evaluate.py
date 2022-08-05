from uvcgan.config import Args
import torch
from uvcgan.models.generator import construct_generator
from uvcgan.torch.funcs       import get_torch_device_smart, seed_everything
import os



if __name__ == "__main__":

    device = get_torch_device_smart()
    args   = Args.load('/dss/dsshome1/lxc09/ra49tad2/crossmoda-challenge/uvcgan/outdir/selfie2anime/model_d(cyclegan)_m(cyclegan)_d(basic)_g(vit-unet)_cyclegan_vit-unet-12-none-lsgan-paper-cycle_high-256/')
    config = args.config
    print(config)
    model = construct_generator(
        config.generator, config.image_shape, device
    )

    ckpt = torch.load(os.path.join('/dss/dsshome1/lxc09/ra49tad2/crossmoda-challenge/uvcgan/outdir/selfie2anime/model_d(cyclegan)_m(cyclegan)_d(basic)_g(vit-unet)_cyclegan_vit-unet-12-none-lsgan-paper-cycle_high-256/net_gen_ab.pth'))
    state_dict = ckpt

    epoch = -1

    if epoch == -1:
        epoch = max(model.find_last_checkpoint_epoch(), 0)

    print("Load checkpoint at epoch %s" % epoch)

    seed_everything(args.config.seed)
    model.load(epoch)
    gen_ab = model.models.gen_ab
    gen_ab.eval()

    

