from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import MNIST, ImageFolder
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import load_dataset
import os
import glob
import json
import argparse
from einops import rearrange, repeat, reduce, pack, unpack
import math
import random
# os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5,6,7"
import torch.nn.init as init



parser = argparse.ArgumentParser()
parser.add_argument('--lrate', default=1e-4, type=float)
parser.add_argument('--test_size', default=1.6, type=float)
parser.add_argument('--alpha', default=1500, type=int)
parser.add_argument('--beta', default=2.0, type=float)
parser.add_argument('--num_samples', default=5000, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--n_T', default=500, type=int)
parser.add_argument('--n_feat', default=256, type=int)
parser.add_argument('--n_sample', default=64, type=int)
parser.add_argument('--n_epoch', default=300, type=int)
parser.add_argument('--experiment', default="H32-fail1", type=str)
parser.add_argument('--remove_node', default="None", type=str)
parser.add_argument('--type_attention', default="", type=str)
parser.add_argument('--pixel_size', default=64, type=int)
parser.add_argument('--dataset', default="single-body_2d_3classes", type=str)
parser.add_argument('--scheduler', default="", type=str)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--std_rate', type=float, default=0.5)
parser.add_argument('--device', type=int, default=0)




def calculate_kaiming_std(fan_in, std_rate):
    return (1 / fan_in) ** std_rate  # Kaiming Normal's std computation



class ResidualConvBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, is_res: bool = False, std_rate=0.5
    ) -> None:
        super().__init__()
        '''
        standard ResNet style convolutional block
        '''
        self.same_channels = in_channels==out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

        self._initialize_weights(std_rate)

    def _initialize_weights(self, std_rate=0.5):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = calculate_kaiming_std(fan_in, std_rate)
                nn.init.uniform_(m.weight, -bound, bound)
                if m.bias is not None:
                    nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            if self.same_channels:
                out = x + x2
            else:
                out = x1 + x2 
            return out
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2


def l2norm(t):
    return F.normalize(t, dim = -1)

def exists(val):
    return val is not None

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class RearrangeToSequence(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        x = rearrange(x, 'b c ... -> b ... c')
        x, ps = pack([x], 'b * c')

        x = self.fn(x)

        x, = unpack(x, ps, 'b * c')
        x = rearrange(x, 'b ... c -> b c ...')
        return x

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5, fp16_eps = 1e-3, stable = False):
        super().__init__()
        self.eps = eps
        self.fp16_eps = fp16_eps
        self.stable = stable
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        eps = self.eps if x.dtype == torch.float32 else self.fp16_eps

        if self.stable:
            x = x / x.amax(dim = -1, keepdim = True).detach()

        var = torch.var(x, dim = -1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = -1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g


# Define the U-Net downsampling and upsampling components
class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels, type_attention,std_rate=0.5):
        super(UnetDown, self).__init__()
        '''
        process and downscale the image feature maps
        '''
        layers = [ResidualConvBlock(in_channels, out_channels, std_rate=std_rate), nn.MaxPool2d(2)]
        attention = nn.Identity()
        self.model = nn.Sequential(*[ResidualConvBlock(in_channels, out_channels, std_rate=std_rate), attention, nn.MaxPool2d(2)])

    def forward(self, x):
        return self.model(x)


class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels, type_attention, std_rate=0.5):
        super(UnetUp, self).__init__()
        '''
        process and upscale the image feature maps
        '''
        attention = nn.Identity()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels, std_rate=std_rate),
            attention, 
            ResidualConvBlock(out_channels, out_channels, std_rate=std_rate),
        ]
        self.model = nn.Sequential(*layers)

        self._initialize_weights(std_rate)

    def _initialize_weights(self, std_rate=0.5):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = calculate_kaiming_std(fan_in, std_rate)
                nn.init.uniform_(m.weight, -bound, bound)
                if m.bias is not None:
                    nn.init.uniform_(m.bias, -bound, bound)


    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)
        x = self.model(x)
        return x


class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim, std_rate=0.5):
        super(EmbedFC, self).__init__()
        '''
        generic one layer FC NN for embedding things  
        '''
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)


        self._initialize_weights(std_rate)

    def _initialize_weights(self, std_rate=0.5):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = calculate_kaiming_std(fan_in, std_rate)
                nn.init.uniform_(m.weight, -bound, bound)
                if m.bias is not None:
                    nn.init.uniform_(m.bias, -bound, bound)


    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)


class ContextUnet(nn.Module):
    def __init__(self, in_channels, n_feat = 256, n_classes=10, dataset="", type_attention="", std_rate=0.5):
        super(ContextUnet, self).__init__()

        self.in_channels = in_channels
        self.n_contexts = len(n_classes)
        self.n_feat = 2 * n_feat
        self.n_classes = n_classes

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True, std_rate=std_rate)

        self.down1 = UnetDown(n_feat, n_feat, type_attention, std_rate=std_rate)
        self.down2 = UnetDown(n_feat, 2 * n_feat, type_attention, std_rate=std_rate)

        n_conv = 16

        self.to_vec = nn.Sequential(nn.AvgPool2d(n_conv), nn.GELU())

        self.timeembed1 = EmbedFC(1, 2*n_feat, std_rate=std_rate)
        self.timeembed2 = EmbedFC(1, 1*n_feat, std_rate=std_rate)

        ### embedding shape
        self.dataset = dataset
        self.n_out1 = 2*n_feat 
        self.n_out2 = n_feat

        self.contextembed1 = nn.ModuleList([EmbedFC(self.n_classes[iclass], self.n_out1, std_rate=std_rate) for iclass in range(len(self.n_classes))])
        self.contextembed2 = nn.ModuleList([EmbedFC(self.n_classes[iclass], self.n_out2, std_rate=std_rate) for iclass in range(len(self.n_classes))])



        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, n_conv, n_conv), 
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )

        self.up1 = UnetUp(4 * n_feat, n_feat, type_attention, std_rate=std_rate)
        self.up2 = UnetUp(2 * n_feat, n_feat, type_attention, std_rate=std_rate)
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )

        self._initialize_weights(std_rate)

    def _initialize_weights(self, std_rate=0.5):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = calculate_kaiming_std(fan_in, std_rate)
                nn.init.uniform_(m.weight, -bound, bound)
                if m.bias is not None:
                    nn.init.uniform_(m.bias, -bound, bound)

            if isinstance(m, nn.ConvTranspose2d):
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = calculate_kaiming_std(fan_in, std_rate)
                nn.init.uniform_(m.weight, -bound, bound)
                if m.bias is not None:
                    nn.init.uniform_(m.bias, -bound, bound)


    def forward(self, x, c, t, context_mask=None):
        # x is (noisy) image, c is context label, t is timestep, 
        # print(f'x shape {x.shape}')

        x = self.init_conv(x)
        # print(f'x shape {x.shape}')
        down1 = self.down1(x)
        # print(f'down1 shape {down1.shape}')
        down2 = self.down2(down1)
        # print(f'down2 shape {down2.shape}')
        hiddenvec = self.to_vec(down2)

        # print(f'hiddenvec shape {hiddenvec.shape}')

        temb1 = self.timeembed1(t).view(-1, int(self.n_feat), 1, 1)
        temb2 = self.timeembed2(t).view(-1, int(self.n_feat/2), 1, 1)

        # embed context, time step
        cemb1 = 0
        cemb2 = 0
        for ic in range(len(self.n_classes)):
            tmpc = c[ic]
            if tmpc.dtype==torch.int64: 
                tmpc = nn.functional.one_hot(tmpc, num_classes=self.n_classes[ic]).type(torch.float)
            cemb1 += self.contextembed1[ic](tmpc).view(-1, int(self.n_out1/1.), 1, 1)
            cemb2 += self.contextembed2[ic](tmpc).view(-1, int(self.n_out2/1.), 1, 1)

        up1 = self.up0(hiddenvec)

        # print(f'up1 shape {up1.shape}')
        up2 = self.up1(cemb1*up1 + temb1, down2)

        # print(f'up2 shape {up2.shape}')
        up3 = self.up2(cemb2*up2 + temb2, down1)

        # print(f'up3 shape {up3.shape}')
        out = self.out(torch.cat((up3, x), 1))

        # print(f'out shape {out.shape}')
        return out


def ddpm_schedules(beta1, beta2, T):
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }


class DDPM(nn.Module):
    def __init__(self, nn_model, betas, n_T, device, drop_prob=0.1, n_classes=None, flag_weight=0):
        super(DDPM, self).__init__()
        self.nn_model = nn_model.to(device)
        self.n_classes = n_classes

        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.betas = torch.linspace(betas[0], betas[1], n_T).to(device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        self.n_T = n_T
        self.device = device
        self.flag_weight = flag_weight
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()

    def forward(self, x, c):
        """
        this method is used in training, so samples t and noise randomly
        """

        _ts = torch.randint(1, self.n_T+1, (x.shape[0],)).to(self.device)  # t ~ Uniform(0, n_T)
        noise = torch.randn_like(x)  # eps ~ N(0, 1)

        x_t = (
            self.sqrtab[_ts, None, None, None] * x
            + self.sqrtmab[_ts, None, None, None] * noise
        )  

        return self.loss_mse(noise, self.nn_model(x_t, c, _ts / self.n_T)) #, context_mask))

    def sample(self, n_sample, c_gen, size, device, guide_w = 0.0):

        x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1), sample initial noise
        _c_gen = [tmpc_gen[:n_sample].to(device) for tmpc_gen in c_gen.values()] 

        #context_mask = torch.zeros_like(_c_gen[0]).to(device)

        x_i_store = [] 
        print()
        for i in range(self.n_T, 0, -1):
            print(f'sampling timestep {i}',end='\r')
            t_is = torch.tensor([i / self.n_T]).to(device)
            t_is = t_is.repeat(n_sample,1,1,1)

            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0
            eps = self.nn_model(x_i, _c_gen, t_is) #, context_mask)
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )
            if i%20==0:
                x_i_store.append(x_i.detach().cpu().numpy())
        
        x_i_store = np.array(x_i_store)
        return x_i, x_i_store


    def ddim_step(self, x_t, t, noise_pred):
        """
        DDIM step to predict the next state of the image.
        """
        alpha_t = self.alphas_cumprod[t]
        alpha_t_1 = torch.where(t > 0, self.alphas_cumprod[t-1], torch.tensor(1.0).to(self.device))
        sigma_t = torch.sqrt((1 - alpha_t_1) / (1 - alpha_t) * (1 - alpha_t / alpha_t_1))
        alpha_t = alpha_t.view(-1,1,1,1)
        sigma_t = sigma_t.view(-1,1,1,1)
        alpha_t_1 = alpha_t_1.view(-1,1,1,1)
    
        x_0_pred = (x_t - sigma_t * noise_pred) / torch.sqrt(alpha_t)
        x_t_1 = torch.sqrt(alpha_t_1) * x_0_pred + sigma_t * torch.randn_like(x_t)
        return x_t_1
    
    def sample_ddim(self, n_sample, c_gen, size, device):
        """
        Sample using the DDIM scheduler.
        """
        x_t = torch.randn(n_sample, *size).to(device)  # Initialize with noise

        _c_gen = {k: v.to(device) for k, v in c_gen.items()}

        x_i_store = [] 
        for i in reversed(range(0, self.n_T)):
            print(f'sampling timestep {i}',end='\r')
            t = torch.full((n_sample,), i, device=device, dtype=torch.long)
            noise_pred = self.nn_model(x_t, _c_gen, t.float() / self.n_T)
            x_t = self.ddim_step(x_t, t, noise_pred)

            if i%20==0:
                x_i_store.append(x_t.detach().cpu().numpy())
        
        x_i_store = np.array(x_i_store)
        return x_t, x_i_store

    

def training(args):

    n_epoch = args.n_epoch 
    batch_size = args.batch_size 
    n_T = args.n_T 
    n_feat = args.n_feat 
    lrate = args.lrate 
    alpha = args.alpha
    beta = args.beta
    test_size = args.test_size
    dataset = args.dataset 
    num_samples = args.num_samples 
    pixel_size = args.pixel_size
    experiment = args.experiment 
    n_sample = args.n_sample 
    type_attention = args.type_attention 
    remove_node = args.remove_node 
    seed = args.seed
    scheduler = args.scheduler
    std_rate = args.std_rate
    in_channels = 3 if "celeba" in dataset else 4
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')


    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


    with open("config_category.json", 'r') as f:
         configs = json.load(f)[experiment]


    experiment_classes = {
        "H42-train1": [2, 3, 1, 1],
        "H22-train1": [2, 2],
        "default": [2, 3, 1]
    }
    n_classes = experiment_classes.get(experiment, experiment_classes["default"])

    if "celeba" in dataset:
        n_classes = [2,2,2]

    tf = transforms.Compose([transforms.Resize((pixel_size,pixel_size)), transforms.ToTensor()])


    save_dir = './output/'+dataset+'/'+experiment+'/'
    if not os.path.isdir(save_dir): os.makedirs(save_dir)
    save_dir = save_dir + str(num_samples) + "_" + str(test_size) + "_" + str(n_feat) + "_" + str(n_T) + "_" + str(n_epoch) \
                        + "_" + str(lrate) + "_" + remove_node + "_" + str(alpha) + "_" + str(beta) + "_" + str(seed)+f"_new_init_{str(args.std_rate)}"+f"seed_{args.seed}_pixel_64_each_sample_280"+"/" #+ str(type_attention) + "/"
    if not os.path.isdir(save_dir): os.makedirs(save_dir)

    ddpm = DDPM(nn_model=ContextUnet(in_channels=in_channels, n_feat=n_feat, n_classes=n_classes, dataset=dataset, type_attention=type_attention, std_rate=std_rate), 
                                     betas=(lrate, 0.02), n_T=n_T, device=device, drop_prob=0.1, n_classes=n_classes)
    ddpm.to(device)


    train_dataset = load_dataset.my_dataset(tf, num_samples, dataset, configs=configs["train"], training=True, alpha=alpha, remove_node=remove_node)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    test_dataloaders = {}
    log_dict = {'train_loss_per_batch': [],
                'test_loss_per_batch': {key: [] for key in configs["test"]}}
    output_configs = list(set(configs["test"] + configs["train"])) 
    for config in output_configs: 
        test_dataset = load_dataset.my_dataset(tf, n_sample, dataset, configs=config, training=False, test_size=test_size) 
        test_dataloaders[config] = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)

    for ep in range(n_epoch):
        print(f'epoch {ep}')

        ddpm.train()

        # linear lrate decay
        optim.param_groups[0]['lr'] = lrate*(1-ep/n_epoch)

        pbar = tqdm(train_dataloader)
        for x, c in pbar:
            optim.zero_grad()
            x = x.to(device)
            _c = [tmpc.to(device) for tmpc in c.values()]
            loss = ddpm(x, _c)
            log_dict['train_loss_per_batch'].append(loss.item())
            loss.backward()
            loss_ema = loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()
        

        ddpm.eval()
        with torch.no_grad():

            for test_config in configs["test"]: 
                for test_x, test_c in test_dataloaders[test_config]:
                    test_x = test_x.to(device)
                    _test_c = [tmptest_c.to(device) for tmptest_c in test_c.values()]
                    test_loss = ddpm(test_x, _test_c)
                    log_dict['test_loss_per_batch'][test_config].append(test_loss.item())

            if (ep + 1) % 100 == 0 or ep >= (n_epoch - 1): 
                for test_config in output_configs: 
                    x_real, c_gen = next(iter(test_dataloaders[test_config]))
                    x_real = x_real[:n_sample].to(device)
                    if scheduler=="DDIM": 
                        x_gen, x_gen_store = ddpm.sample_ddim(n_sample, c_gen, (in_channels, pixel_size, pixel_size), device)
                    else:
                        x_gen, x_gen_store = ddpm.sample(n_sample, c_gen, (in_channels, pixel_size, pixel_size), device, guide_w=0.0)
                    np.savez_compressed(save_dir + f"image_"+test_config+"_ep"+str(ep)+".npz", x_gen=x_gen.detach().cpu().numpy()) 
                    print('saved image at ' + save_dir + f"image_"+test_config+"_ep"+str(ep)+".png")

                    if ep + 1 == n_epoch: 
                        np.savez_compressed(save_dir + f"gen_store_"+test_config+"_ep"+str(ep)+".npz", x_gen_store=x_gen_store)
                        print('saved image file at ' + save_dir + f"gen_store_"+test_config+"_ep"+str(ep)+".npz")


            if (ep + 1) == n_epoch:
                with open(save_dir + f"training_log_"+str(ep)+".json", "w") as outfile:
                    json.dump(log_dict, outfile)



if __name__ == "__main__":
    args = parser.parse_args()
    training(args)


