
import torch
from torch import Tensor, optim, nn
import cv2
from network import ReLUNeuralField, SIRENNeuralField, GARF
import numpy as np
import os
from tqdm import tqdm
import click
from typing import Union

def render(img: Tensor, network: Union[ReLUNeuralField, SIRENNeuralField, GARF]) -> Tensor:
    h, w, _ = img.shape

    us = torch.arange(w).reshape(1, w).expand(h, w).reshape(-1).to(img.device) + 0.5
    vs = torch.arange(h).reshape(h, 1).expand(h, w).reshape(-1).to(img.device) + 0.5

    ps = torch.stack([(1.0/w) * us, (1.0/h) * vs], dim=1)
    return network(ps).reshape(h,w,3)

def render_nabla(img: Tensor, network: Union[ReLUNeuralField, SIRENNeuralField, GARF]) -> Tensor:
    h, w, _ = img.shape

    us = torch.arange(w).reshape(1, w).expand(h, w).reshape(-1).to(img.device) + 0.5
    vs = torch.arange(h).reshape(h, 1).expand(h, w).reshape(-1).to(img.device) + 0.5

    ps = torch.stack([(1.0/w) * us, (1.0/h) * vs], dim=1)
    ps.requires_grad_(True)
    img_out = network(ps).reshape(-1,3)[:,0]

    grad_outputs = torch.ones_like(img_out)
    ps_grad = torch.autograd.grad(img_out, [ps], grad_outputs=grad_outputs, create_graph=True)[0] * 0.03

    ps_grad_np = ps_grad.detach().cpu().numpy()
    hue = ((np.arctan2(ps_grad_np[:,1], ps_grad_np[:,0]) + np.pi) * (90/np.pi)).clip(min=0, max=180).astype(np.uint8).reshape(h,w)
    val = (np.sqrt(np.sum(np.square(ps_grad_np), 1)) * 255).clip(min=0, max=255).astype(np.uint8).reshape(h,w)
    sat = np.ones_like(hue) * 255
    hsv = np.stack([hue, val, val], 2)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return rgb
    
@click.command()
@click.option(
        "--network_type", 
        type=str,
        default="GARF",
        help="Network name select from [ReLU, SIREN, GARF]",
)
@click.option(
        "--use_pe", 
        is_flag=True,
)
@click.option(
        "--pe_dim", 
        type=int,
        default=0,
)
@click.option(
        "--layer_count", 
        type=int,
        default=4,
)
@click.option(
        "--layer_width", 
        type=int,
        default=64,
)
def run(network_type:str, use_pe:bool, pe_dim:int, layer_count:int, layer_width:int):
    target_data = "Mandrill"
    log_dir = "result/{}/{}".format(target_data, network_type)
    img_np = cv2.imread("data/Mandrill.bmp")
    img = torch.from_numpy((1.0/256) * img_np.astype(np.float32))
    network: Union[ReLUNeuralField, SIRENNeuralField, GARF]
    if network_type == "ReLU":
        network = ReLUNeuralField(use_pe, pe_dim, layer_count, layer_width)
    elif network_type == "SIREN":
        network = SIRENNeuralField(use_pe, pe_dim, layer_count, layer_width)
    elif network_type == "GARF":
        network = GARF(use_pe, pe_dim, layer_count, layer_width)
    else:
        print("Invalid network type selected.")
        return

    optimizer = optim.Adam(network.parameters(), lr=0.01)
    mse_loss = nn.MSELoss()

    os.makedirs(log_dir, exist_ok=True)
    
    for i in tqdm(range(5000)):
        optimizer.zero_grad()
        img_out = render(img, network)
        loss = mse_loss(img_out, img)
        loss.backward()
        optimizer.step()
        if i % 50 == 0:
            img_out_np = (img_out.detach().cpu().numpy()*255).clip(min=0, max=255).astype(np.uint8)
            nabla_np = render_nabla(img, network)
            cv2.imwrite("{}/rgb_{:04}.png".format(log_dir, i), img_out_np)
            cv2.imwrite("{}/nabla_{:04}.png".format(log_dir, i), nabla_np)
            cv2.imshow("result", img_out_np)
            cv2.imshow("nabla", nabla_np)
            cv2.waitKey(1)
            render_nabla(img, network)

if __name__ == "__main__":
    run()