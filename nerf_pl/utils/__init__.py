from .run_nerf_helpers_dsnerf import *
import torch
# optimizer
from torch.optim import SGD, Adam
import torch_optimizer as optim
# scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from .warmup_scheduler import GradualWarmupScheduler
from .visualization import *
import os

def get_parameters(models):
    """Get all model parameters recursively."""
    parameters = []
    if isinstance(models, list):
        for model in models:
            parameters += get_parameters(model)
    elif isinstance(models, dict):
        for model in models.values():
            parameters += get_parameters(model)
    else: # models is actually a single pytorch model
        parameters += list(models.parameters())
    return parameters

def get_optimizer(hparams, models):
    eps = 1e-8
    parameters = get_parameters(models)
    if hparams.optimizer == 'sgd':
        optimizer = SGD(parameters, lr=hparams.lr, 
                        momentum=hparams.momentum, weight_decay=hparams.weight_decay)
    elif hparams.optimizer == 'adam':
        optimizer = Adam(parameters, lr=hparams.lr, eps=eps, 
                         weight_decay=hparams.weight_decay)
    elif hparams.optimizer == 'radam':
        optimizer = optim.RAdam(parameters, lr=hparams.lr, eps=eps, 
                                weight_decay=hparams.weight_decay)
    elif hparams.optimizer == 'ranger':
        optimizer = optim.Ranger(parameters, lr=hparams.lr, eps=eps, 
                                 weight_decay=hparams.weight_decay)
    else:
        raise ValueError('optimizer not recognized!')

    return optimizer

def get_scheduler(hparams, optimizer):
    eps = 1e-8
    if hparams.lr_scheduler == 'steplr':
        scheduler = MultiStepLR(optimizer, milestones=hparams.decay_step, 
                                gamma=hparams.decay_gamma)
    elif hparams.lr_scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=hparams.num_epochs, eta_min=eps)
    elif hparams.lr_scheduler == 'poly':
        scheduler = LambdaLR(optimizer, 
                             lambda epoch: (1-epoch/hparams.num_epochs)**hparams.poly_exp)
    else:
        raise ValueError('scheduler not recognized!')

    if hparams.warmup_epochs > 0 and hparams.optimizer not in ['radam', 'ranger']:
        scheduler = GradualWarmupScheduler(optimizer, multiplier=hparams.warmup_multiplier, 
                                           total_epoch=hparams.warmup_epochs, after_scheduler=scheduler)

    return scheduler

def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def extract_model_state_dict(ckpt_path, model_name='model', prefixes_to_ignore=[]):
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
    checkpoint_ = {}
    if 'state_dict' in checkpoint: # if it's a pytorch-lightning checkpoint
        checkpoint = checkpoint['state_dict']
    for k, v in checkpoint.items():
        print("huh2")
        if not k.startswith(model_name):
            continue
        k = k[len(model_name)+1:]
        for prefix in prefixes_to_ignore:
            print("huh")
            if k.startswith(prefix):
                print('ignore', k)
                break
        else:
            print("huh3")
            checkpoint_[k] = v
    return checkpoint_

def load_ckpt(model, ckpt_path, model_name='model', prefixes_to_ignore=[]):
    if not ckpt_path:
        return
    model_dict = model.state_dict()
    checkpoint_ = extract_model_state_dict(ckpt_path, model_name, prefixes_to_ignore)
    model_dict.update(checkpoint_)
    model.load_state_dict(model_dict)

def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn

    def ret(inputs):
        return torch.cat([fn(inputs[i:i + chunk]) for i in range(0, inputs.shape[0], chunk)], 0)

    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*16):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs

def load_ckpt_other():
    embed_fn, input_ch = get_embedder(10, 0)
    args = {
        "N_importance": 128,
        "N_iters": 300000,
        "N_rand": 2048,
        "N_samples": 64,
        "alpha_model_path": None,
        "basedir": "../DSNeRF/logs/release",
        "chunk": 16384,
        "colmap_depth": True,
        "config": "configs/250frames.txt",
        "datadir": "../Hierarchical-Localization/outputs/larger_reconstruction/larger_reconstruction_250/",
        "dataset_type": "llff",
        "debug": False,
        "depth_lambda": 0.8,
        "depth_loss": True,
        "depth_rays_prop": 0.5,
        "depth_with_rgb": False,
        "expname": "250_frames_tensorboard_most_depth",
        "factor": 1,
        "ft_path": None,
        "half_res": False,
        "i_embed": 0,
        "i_img": 500,
        "i_print": 100,
        "i_testset": 10000,
        "i_video": 10000,
        "i_weights": 1000,
        "lindisp": False,
        "llffhold": 8,
        "lrate": 0.0005,
        "lrate_decay": 250,
        "multires": 10,
        "multires_views": 4,
        "netchunk": 16384,
        "netdepth": 8,
        "netdepth_fine": 8,
        "netwidth": 256,
        "netwidth_fine": 256,
        "no_batching": False,
        "no_coarse": False,
        "no_ndc": True,
        "no_reload": False,
        "normalize_depth": False,
        "perturb": 1.0,
        "precrop_frac": 0.5,
        "precrop_iters": 0,
        "raw_noise_std": 1.0,
        "relative_loss": False,
        "render_factor": 0,
        "render_mypath": False,
        "render_only": False,
        "render_test": False,
        "render_test_ray": False,
        "render_train": False,
        "shape": "greek",
        "sigma_lambda": 0.1,
        "sigma_loss": False,
        "spherify": False,
        "test_scene": None,
        "testskip": 8,
        "train_scene": None,
        "use_viewdirs": True,
        "weighted_loss": False,
        "white_bkgd": False
    }

    input_ch_views = 0
    embeddirs_fn = None
    if True:
        embeddirs_fn, input_ch_views = get_embedder(4, 0)
    output_ch = 5 if args["N_importance"] > 0 else 4
    skips = [4]
    if args["alpha_model_path"] is None:
        model = NeRF(D=args["netdepth"], W=args["netwidth"],
                     input_ch=input_ch, output_ch=output_ch, skips=skips,
                     input_ch_views=input_ch_views, use_viewdirs=args["use_viewdirs"]).to(device)
        grad_vars = list(model.parameters())
    else:
        alpha_model = NeRF(D=args["netdepth_fine"], W=args["netwidth_fine"],
                           input_ch=input_ch, output_ch=output_ch, skips=skips,
                           input_ch_views=input_ch_views, use_viewdirs=args["use_viewdirs"]).to(device)
        print('Alpha model reloading from', args["alpha_model_path"])
        ckpt = torch.load(args["alpha_model_path"])
        alpha_model.load_state_dict(ckpt['network_fine_state_dict'])
        if not args["no_coarse"]:
            model = NeRF_RGB(D=args["netdepth"], W=args["netwidth"],
                             input_ch=input_ch, output_ch=output_ch, skips=skips,
                             input_ch_views=input_ch_views, use_viewdirs=args["use_viewdirs"],
                             alpha_model=alpha_model).to(
                device)
            grad_vars = list(model.parameters())
        else:
            model = None
            grad_vars = []

    model_fine = None
    if args["N_importance"] > 0:
        if args["alpha_model_path"] is None:
            model_fine = NeRF(D=args["netdepth_fine"], W=args["netwidth_fine"],
                              input_ch=input_ch, output_ch=output_ch, skips=skips,
                              input_ch_views=input_ch_views, use_viewdirs=args["use_viewdirs"]).to(device)
        else:
            model_fine = NeRF_RGB(D=args["netdepth_fine"], W=args["netwidth_fine"],
                                  input_ch=input_ch, output_ch=output_ch, skips=skips,
                                  input_ch_views=input_ch_views, use_viewdirs=args["use_viewdirs"],
                                  alpha_model=alpha_model).to(device)
        grad_vars += list(model_fine.parameters())

    network_query_fn = lambda inputs, viewdirs, network_fn: run_network(inputs, viewdirs, network_fn,
                                                                        embed_fn=embed_fn,
                                                                        embeddirs_fn=embeddirs_fn,
                                                                        netchunk=args["netchunk"])

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args["lrate"], betas=(0.9, 0.999))

    start = 0
    basedir = args["basedir"]
    expname = args["expname"]

    ##########################

    # Load checkpoints
    if args["ft_path"] is not None and args["ft_path"] != 'None':
        ckpts = [args["ft_path"]]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if
                 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args["no_reload"]:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])
    if not ckpt_path:
        return
    optimizer = torch.optim.Adam(params=grad_vars, lr=args["lrate"], betas=(0.9, 0.999))
    ckpt = torch.load(ckpt_path)

    start = ckpt['global_step']
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])

    # Load model
    model.load_state_dict(ckpt['network_fn_state_dict'])
    if model_fine is not None:
        model_fine.load_state_dict(ckpt['network_fine_state_dict'])
    return model, model_fine
