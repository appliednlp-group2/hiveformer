import random
from typing import List, Tuple, Dict, Optional, Any
import os
from pathlib import Path
import torch
from torch import nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter  # type: ignore
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
import numpy as np
from tqdm import tqdm, trange
from filelock import FileLock
import tap
from network import Hiveformer
from utils import (
    LossAndMetrics,
    load_instructions,
    RLBenchEnv,
    count_parameters,
    load_episodes,
    get_max_episode_length,
    Actioner,
)
from dataset import RLBenchDataset


class Arguments(tap.Tap):
    accumulate_grad_batches: int = 1
    cameras: Tuple[str, ...] = ("wrist", "left_shoulder", "right_shoulder")
    checkpoint_dir: Optional[Path] = None
    checkpoint_period: int = 10
    device: str = "cuda"
    xp: Path = Path(__file__).parent / "xp"
    valset: Optional[Tuple[Path, ...]] = None
    name: str = "hiveformer"
    arch: str = "mct"
    num_workers: int = 5
    max_tries: int = 10
    max_episodes_per_taskvar: int = 100
    instructions: Optional[Path] = None
    cache_size: int = 100
    seed: int = 2

    tasks: Tuple[str, ...]
    variations: Tuple[int, ...] = (0,)

    # Train
    batch_size: int = 32
    lr: float = 0.001
    val_freq: int = 200
    val_batch_size: int = 100
    train_iters: int = 100_000
    jitter: bool = False

    # tests
    headless: bool = False
    output: Path = Path(__file__).parent / "records.txt"

    # model
    depth: int = 4
    dim_feedforward: int = 64
    hidden_dim: int = 64
    instr_size: int = 512
    mask_obs_prob: float = 0.0
    num_layers: int = 1
    num_words: int = 53
    num_demos: int = 100
    num_repeat: int = 1
    steps: Tuple[int, ...] = (50, 100, 150, 200, 250, 300, 350, 400, 450, 500)


def get_model(args: Arguments) -> Tuple[optim.Optimizer, Hiveformer]:
    device = torch.device(args.device)

    max_episode_length = get_max_episode_length(args.tasks, args.variations)
    model = Hiveformer(
        depth=args.depth,
        dim_feedforward=args.dim_feedforward,
        hidden_dim=args.hidden_dim,
        instr_size=args.instr_size,
        mask_obs_prob=args.mask_obs_prob,
        max_episode_length=max_episode_length,
        num_layers=args.num_layers,
        num_words=args.num_words
    ).to(device)

    optimizer_grouped_parameters = [
        {"params": [], "weight_decay": 0.0, "lr": args.lr},
        {"params": [], "weight_decay": 5e-4, "lr": args.lr},
    ]
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    for name, param in model.named_parameters():
        if any(nd in name for nd in no_decay):
            optimizer_grouped_parameters[0]["params"].append(param)  # type: ignore
        else:
            optimizer_grouped_parameters[1]["params"].append(param)  # type: ignore
    optimizer: optim.Optimizer = optim.AdamW(optimizer_grouped_parameters)

    if args.checkpoint is not None:
        model_dict = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(model_dict["weight"])
        optimizer.load_state_dict(model_dict["optimizer"])

    print("Number of parameters:")
    model_params = count_parameters(model)
    print("- model", model_params)
    print("Total", model_params)

    return optimizer, model


if __name__ == "__main__":
    args = Arguments().parse_args()
    print(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device(args.device)

    output_list = []
    for _ in range(args.num_repeat):
        output = {0: 0}
        for step in args.steps: # [50, 100, 150, 200, 250, 350, 300, 350, 400, 450, 500, 550]:
            if step == 550:
                args.checkpoint = args.checkpoint_dir / f"mtl_2_0.001.pth"
            else:
                args.checkpoint = args.checkpoint_dir / f"model.step={step}-value=0.pth"
            optimizer, model = get_model(args)

            # evaluation
            model.eval()

            env = RLBenchEnv(
                data_path="",
                apply_rgb=True,
                apply_pc=True,
                apply_cameras=("left_shoulder", "right_shoulder", "wrist"),
                headless=args.headless,
            )

            instruction = load_instructions(args.instructions)
            if instruction is None:
                raise NotImplementedError()

            actioner = Actioner(model=model, instructions=instruction)
            max_eps_dict = load_episodes()["max_episode_length"]

            success_rate = env.evaluate(
                args.tasks[0],
                actioner=actioner,
                max_episodes=max_eps_dict.get(args.tasks[0], 6),
                variation=args.variations[0],
                num_demos=args.num_demos,
                demos=None,
                log_dir=None,
                max_tries=args.max_tries,
            )
            output[step] = success_rate
            print("[{}]Testing Success Rate {}: {:.04f}".format(step, args.tasks[0], success_rate))
        print(output)
        output_list += [output]
        
    import pickle
    with open(f'{args.checkpoint_dir}/success_rate.pkl', 'wb') as f:
        pickle.dump(output_list, f)
