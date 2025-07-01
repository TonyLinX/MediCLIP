import argparse
import os
import math
import yaml
import torch
import tifffile
from easydict import EasyDict
from tqdm import tqdm
import open_clip

from models.Necker import Necker
from models.Adapter import Adapter
from models.MapMaker import MapMaker
from models.CoOp import PromptMaker
from utils.misc_helper import map_func, set_seed

from MVTecAD2_public_code_utils import mvtec_ad_2_public_offline as mad2


@torch.no_grad()
def make_vision_tokens_info(model, model_cfg, layers_out):
    img = torch.ones(
        (1, 3, model_cfg["vision_cfg"]["image_size"], model_cfg["vision_cfg"]["image_size"])
    ).to(model.device)
    _, tokens = model.encode_image(img, layers_out)
    if len(tokens[0].shape) == 3:
        model.token_size = [int(math.sqrt(t.shape[1] - 1)) for t in tokens]
        model.token_c = [t.shape[-1] for t in tokens]
    else:
        model.token_size = [t.shape[2] for t in tokens]
        model.token_c = [t.shape[1] for t in tokens]
    model.embed_dim = model_cfg["embed_dim"]


@torch.no_grad()
def inference_on_split(
    split,
    model,
    necker,
    adapter,
    prompt_maker,
    map_maker,
    preprocess,
    output_dir,
    layers_out,
    objects,
    batch_size=8,
):
    for obj in objects:
        dataset = mad2.MVTecAD2(obj, split, transform=preprocess)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=2)
        for batch in tqdm(dataloader, desc=f"{obj}-{split}"):
            images = batch["sample"].to(model.device)
            _, image_tokens = model.encode_image(images, out_layers=layers_out)
            image_features = necker(image_tokens)
            vision_adapter_features = adapter(image_features)
            prompt_adapter_features = prompt_maker(vision_adapter_features)
            anomaly_map = map_maker(vision_adapter_features, prompt_adapter_features)
            anomaly_map = anomaly_map[:, 1, :, :]
            for idx in range(images.size(0)):
                out_path = os.path.join(output_dir, batch["rel_out_path_cont"][idx])
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                tifffile.imwrite(out_path, anomaly_map[idx].cpu().numpy().astype("float16"))


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(args.config_path) as f:
        args.config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))
    set_seed(seed=args.config.get("random_seed", 0))

    model, preprocess, model_cfg = open_clip.create_model_and_transforms(
        args.config.model_name, args.config.image_size, device=device
    )
    for p in model.parameters():
        p.requires_grad_(False)
    args.config.model_cfg = model_cfg

    make_vision_tokens_info(model, args.config.model_cfg, args.config.layers_out)

    necker = Necker(clip_model=model).to(device)
    adapter = Adapter(clip_model=model, target=args.config.model_cfg["embed_dim"]).to(device)
    prompt_maker = PromptMaker(
        prompts=args.config.prompts,
        clip_model=model,
        n_ctx=args.config.n_learnable_token,
        CSC=args.config.CSC,
        class_token_position=args.config.class_token_positions,
    ).to(device)
    map_maker = MapMaker(image_size=args.config.image_size).to(device)

    ckpt = torch.load(args.checkpoint_path, map_location=map_func)
    adapter.load_state_dict(ckpt["adapter_state_dict"])
    prompt_maker.prompt_learner.load_state_dict(ckpt["prompt_state_dict"])
    prompt_maker.prompt_learner.eval()
    adapter.eval()

    mad2.PATH_TO_MVTEC_AD_2_FOLDER = args.mvtec_root

    splits = ["test_public", "test_private", "test_private_mixed"]
    objects = args.objects.split(',') if args.objects else mad2.MVTEC_AD_2_OBJECTS
    for split in splits:
        inference_on_split(
            split,
            model,
            necker,
            adapter,
            prompt_maker,
            map_maker,
            preprocess,
            args.output_dir,
            args.config.layers_out,
            objects,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test MediCLIP on MVTec AD 2")
    parser.add_argument("--config_path", type=str, help="model config path")
    parser.add_argument("--checkpoint_path", type=str, help="trained checkpoint path")
    parser.add_argument("--mvtec_root", type=str, required=True, help="path to mvtec_ad_2 dataset")
    parser.add_argument("--output_dir", type=str, default="mvtec_results", help="directory to save results")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--objects",
        type=str,
        default="",
        help=(
            "Comma-separated list of MVTec AD 2 objects to evaluate. "
            "If not set, all objects are processed."
        ),
    )
    args = parser.parse_args()
    torch.multiprocessing.set_start_method("spawn")
    main(args)
    
"""_code_
python  test_mvtec.py --config_path config/fabric.yaml  --checkpoint_path results/2025_07_01_01_29_48/checkpoints_52.pkl --mvtec_root mvtec_ad_2 --objects fabric
"""
