"""
Precompute embeddings of instructions
"""
import re
import json
from pathlib import Path
import itertools
from typing import List, Tuple, List, Literal, Dict, Optional
from collections import defaultdict
import pickle
import tap
import transformers
from tqdm.auto import tqdm
import torch
from torch import nn
from utils import RLBenchEnv, task_file_to_task_class
from chatgpt import InstructionLLM
from contextlib import redirect_stdout
import io


Annotations = Dict[str, Dict[int, List[str]]]
TextEncoder = Literal["bert", "clip"]


class Arguments(tap.Tap):
    tasks: Tuple[str, ...]
    output: Path
    batch_size: int = 10
    encoder: TextEncoder = "clip"
    model_max_length: int = 53
    variations: Tuple[int, ...] = (0, )
    device: str = "cuda"
    annotations: Tuple[Path, ...] = ()
    zero: bool = False
    verbose: bool = False
    fixed: bool = False


def parse_int(s):
    return int(re.findall(r"\d+", s)[0])


def load_model(encoder: TextEncoder) -> transformers.PreTrainedModel:
    if encoder == "bert":
        model = transformers.BertModel.from_pretrained("bert-base-uncased")
    elif encoder == "clip":
        model = transformers.CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
    else:
        raise ValueError(f"Unexpected encoder {encoder}")
    if not isinstance(model, transformers.PreTrainedModel):
        raise ValueError(f"Unexpected encoder {encoder}")
    return model


def load_tokenizer(encoder: TextEncoder) -> transformers.PreTrainedTokenizer:
    if encoder == "bert":
        tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
    elif encoder == "clip":
        tokenizer = transformers.CLIPTokenizer.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
    else:
        raise ValueError(f"Unexpected encoder {encoder}")
    if not isinstance(tokenizer, transformers.PreTrainedTokenizer):
        raise ValueError(f"Unexpected encoder {encoder}")
    return tokenizer


def load_annotations(annotations: Tuple[Path, ...]) -> Annotations:
    data = []
    for annotation in annotations:
        with open(annotation) as fid:
            data += json.load(fid)

    items: Annotations = {}
    for item in data:
        task = item["fields"]["task"]
        variation = item["fields"]["variation"]
        instruction = item["fields"]["instruction"]

        if instruction == "":
            continue

        if task not in items:
            items[task] = {}

        if variation not in items[task]:
            items[task][variation] = []

        items[task][variation].append(instruction)

    # merge annotations for push_buttonsX (same variations)
    push_buttons = ("push_buttons", "push_buttons3")
    for task, task2 in itertools.product(push_buttons, push_buttons):
        items[task] = items.get(task, {})
        for variation, instrs in items.get(task2, {}).items():
            items[task][variation] = instrs + items[task].get(variation, [])

    # display statistics
    for task, values in items.items():
        print(task, ":", sorted(values.keys()))

    return items


if __name__ == "__main__":
    args = Arguments().parse_args()
    print(args)

    annotations = load_annotations(args.annotations)

    tokenizer = load_tokenizer(args.encoder)
    tokenizer.model_max_length = args.model_max_length

    model = load_model(args.encoder)
    model = model.to(args.device)

    env = RLBenchEnv(
        data_path="",
        apply_rgb=True,
        apply_pc=True,
        apply_cameras=("left_shoulder", "right_shoulder", "wrist"),
        headless=True,
    )

    instructions: Dict[str, Dict[int, torch.Tensor]] = {}
    tasks = set(args.tasks)
    if args.fixed:
        llm = InstructionLLM()

    for task in tqdm(tasks):
        task_type = task_file_to_task_class(task)
        task_inst = env.env.get_task(task_type)._task
        task_inst.init_task()

        instructions[task] = {}

        variations = [v for v in args.variations if v < task_inst.variation_count()]
        # print("Num available var", task_inst.variation_count())
        # print("Num var", len(variations))
        for variation in variations:
            # check instructions among annotations
            if task in annotations and variation in annotations[task]:
                instr: Optional[List[str]] = annotations[task][variation]
            # or, collect it from RLBench synthetic instructions
            else:
                instr = None
                for i in range(30):
                    try:
                        instr = task_inst.init_episode(variation)
                        if args.fixed:
                            instr = [llm.get_instruction(task, instr)]
                        # instr = ["1. Position the arm above the rubbish. 2. Use the arm's gripper to pick up the rubbish. 3. Move the arm to the bin. 4. Release the rubbish into the bin by opening the gripper."]
                        # instr = ["1. Move the robotic arm to the door handle. 2. Grasp the door handle with the gripper. 3. Pull the door towards the robot until it is closed."]
                        if args.verbose:
                            print(task, variation, instr)

                        tokens = tokenizer(instr, padding="max_length")["input_ids"]
                        lengths = [len(t) for t in tokens]
                        if any(l > args.model_max_length for l in lengths):
                            raise RuntimeError(f"Too long instructions: {lengths}")
                        print(task + ":")
                        print(instr)
                        break
                    except:
                        print(f"Cannot init episode {task}")
                if instr is None:
                    raise RuntimeError()

            tokens = torch.tensor(tokens).to(args.device)
            with torch.no_grad():
                pred = model(tokens).last_hidden_state
            instructions[task][variation] = pred.cpu()
            instructions[task]["raw"] = instr

    if args.zero:
        for instr_task in instructions.values():
            for variation, instr_var in instr_task.items():
                instr_task[variation].fill_(0)

    print("Instructions:", sum(len(inst) for inst in instructions.values()))

    for task in tqdm(tasks):
        path = args.output / task / "instructions.pkl"
        if args.fixed:
            path = args.output / task / "instructions_fixed.pkl"
        path.parent.mkdir(exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({task: instructions[task]}, f)
