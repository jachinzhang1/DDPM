import torch
import torch.nn as nn
from dataset import get_dataloader, get_img_shape
from ddpm import DDPM
from network import *
import cv2
import einops
import numpy as np
import os
from time import time
import matplotlib.pyplot as plt
import argparse
from torch.utils.tensorboard import SummaryWriter


batch_loss = []

parser = argparse.ArgumentParser()

parser.add_argument('--ckpt_path', type=str, default='./ckpts')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--n_epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--n_steps', type=int, default=1000)
parser.add_argument('--config_id', type=int, default=4)
parser.add_argument('--warm_start', type=bool, default=False)
parser.add_argument('--pretrained_id', type=int, default=3)

args = parser.parse_args()

def plot_loss(losses):
    plt.plot(losses)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.savefig('loss.png')
    # plt.show()


def train(ddpm: DDPM, net, device, ckpt_path):
    n_steps = ddpm.n_steps
    dataloader = get_dataloader(args.batch_size)
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    model_num = len(os.listdir(ckpt_path))

    total_time = 0

    for epoch in range(args.n_epochs):
        begin_time = time()
        for i, (x, _) in enumerate(dataloader):
            current_batch_size = x.shape[0]
            x = x.to(device)
            t = torch.randint(0, n_steps, (current_batch_size,)).to(device)
            eps = torch.randn_like(x).to(device)
            x_t = ddpm.sample_forward(x, t, eps)
            eps_pred = net(x_t, t.reshape(current_batch_size, 1))
            loss = loss_fn(eps_pred, eps)
            batch_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 10 == 0 or (i + 1) == len(dataloader) or i == 0:
                print(f'Epoch: {epoch + 1}/{args.n_epochs}, Step: {i + 1}/{len(dataloader)}, Loss: {loss.item():.4f}')
        lasting_time = time() - begin_time
        total_time += lasting_time
        step_passed = epoch * len(dataloader) + i + 1
        step_total = args.n_epochs * len(dataloader)
        estimated_time = (total_time / step_passed) * (step_total - step_passed)
        print(f'Epoch {epoch + 1}/{args.n_epochs} finished in {lasting_time:.2f}s. {total_time:.2f}s in total. About {int(estimated_time)}s left.')
        
    torch.save(net.state_dict(), './ckpts/model-%d' % (model_num + 1))
    
    plot_loss(batch_loss)


def sample_imgs(
        ddpm: DDPM,
        net,
        output_path,
        n_samples=81,
        device='cuda',
        simple_var=True,
):
    net = net.to(device)
    net.eval()
    with torch.no_grad():
        shape = (n_samples, *get_img_shape())
        imgs = ddpm.sample_backward(
            shape,
            net,
            device,
            simple_var
        ).detach().cpu()
        imgs = (imgs + 1) / 2 * 255
        imgs = imgs.clamp(0, 255).to(torch.uint8)
        imgs = einops.rearrange(imgs, 
                                '(b1 b2) c h w -> (b1 h) (b2 w) c',
                                b1=int(n_samples**0.5))
        imgs = imgs.numpy()
        cv2.imwrite(output_path, imgs)


configs = [
    convnet_small_cfg,
    convnet_medium_cfg,
    convnet_big_cfg,
    unet_1_cfg,
    unet_res_cfg
]

if __name__ == '__main__':
    assert args.config_id < len(configs)

    if os.path.exists(args.ckpt_path) is False:
        os.mkdir(args.ckpt_path)
    
    config = configs[args.config_id]
    net = build_network(config, args.n_steps)

    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f'Total params: {total_params}')

    with SummaryWriter(comment='UNet') as writer:
        writer.add_graph(net.to(args.device), (torch.randn(1, *get_img_shape()).to(args.device), torch.tensor([0]).to(args.device)))

    if args.warm_start:
        net.load_state_dict(torch.load('./ckpts/model-%d' % args.pretrained_id))
    ddpm = DDPM(args.device, args.n_steps)

    train(ddpm, net, args.device, args.ckpt_path)

    if os.path.exists('./sample') is False:
        os.mkdir('./sample')
    sample_num = len(os.listdir('./sample'))
    sample_imgs(ddpm, net, './sample/sample-%d.png' % (sample_num + 1))