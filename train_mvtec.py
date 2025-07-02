import os
import math
import argparse
import random
import yaml
import open_clip
import torch
from easydict import EasyDict
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

from models.Necker import Necker
from models.Adapter import Adapter
from models.MapMaker import MapMaker
from models.CoOp import PromptMaker
from utils.losses import FocalLoss, BinaryDiceLoss
from utils.misc_helper import (
    AverageMeter,
    create_logger,
    get_current_time,
    compute_segmentation_f1,
    set_seed,
)
from mvtec_ad_evaluation.pro_curve_util import compute_pro
from mvtec_ad_evaluation.generic_util import trapezoid

from medsyn.tasks import (
    CutPastePatchBlender,
    SmoothIntensityChangeTask,
    GaussIntensityChangeTask,
    SinkDeformationTask,
    SourceDeformationTask,
    IdentityTask,
)

from MVTecAD2_public_code_utils import mvtec_ad_2_public_offline as mad2


class MVTecAD2SyntheticDataset(Dataset):
    """Dataset generating synthetic anomalies for MVTec AD 2 splits."""

    def __init__(self, obj, split, args, preprocess, k_shot=-1):
        super().__init__()
        mad2.PATH_TO_MVTEC_AD_2_FOLDER = args.mvtec_root
        base_dataset = mad2.MVTecAD2(obj, split, transform=None)
        self.image_paths = base_dataset.image_paths
        if k_shot != -1 and k_shot < len(self.image_paths):
            self.image_paths = random.sample(self.image_paths, k_shot)
        self.args = args
        self.preprocess = preprocess
        self.augs, self.augs_pro = self._load_anomaly_syn()

    def _load_anomaly_syn(self):
        tasks = []
        probs = []
        for name, prob in self.args.config.anomaly_tasks.items():
            if name == 'CutpasteTask':
                support_images = [self._read_image(p) for p in self.image_paths]
                task = CutPastePatchBlender(support_images)
            elif name == 'SmoothIntensityTask':
                task = SmoothIntensityChangeTask(30.0)
            elif name == 'GaussIntensityChangeTask':
                task = GaussIntensityChangeTask()
            elif name == 'SinkTask':
                task = SinkDeformationTask()
            elif name == 'SourceTask':
                task = SourceDeformationTask()
            elif name == 'IdentityTask':
                task = IdentityTask()
            else:
                raise NotImplementedError(
                    "task must in [CutpasteTask, SmoothIntensityTask, GaussIntensityChangeTask, SinkTask, SourceTask, IdentityTask]"
                )
            tasks.append(task)
            probs.append(prob)
        assert abs(sum(probs) - 1.0) < 1e-5
        return tasks, probs

    def _read_image(self, path):
        img = Image.open(path).resize((self.args.config.image_size, self.args.config.image_size), Image.Resampling.BILINEAR).convert('L')
        return np.array(img).astype(np.uint8)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = self._read_image(self.image_paths[idx])
        aug = np.random.choice(self.augs, p=self.augs_pro)
        img, mask = aug(img)
        img = Image.fromarray(img.astype(np.uint8)).convert('RGB')
        img = self.preprocess(img)
        mask = torch.from_numpy(mask)
        return {'image': img, 'mask': mask}


def make_vision_tokens_info(model, model_cfg, layers_out):
    img = torch.ones((1, 3, model_cfg['vision_cfg']['image_size'], model_cfg['vision_cfg']['image_size'])).to(model.device)
    _, tokens = model.encode_image(img, layers_out)
    if len(tokens[0].shape) == 3:
        model.token_size = [int(math.sqrt(t.shape[1] - 1)) for t in tokens]
        model.token_c = [t.shape[-1] for t in tokens]
    else:
        model.token_size = [t.shape[2] for t in tokens]
        model.token_c = [t.shape[1] for t in tokens]
    model.embed_dim = model_cfg['embed_dim']


def train_one_epoch(args, dataloader, optimizer, epoch, start_iter, logger, model, necker, adapter, prompt_maker, map_maker):
    loss_meter = AverageMeter(args.config.print_freq_step)
    focal_criterion = FocalLoss()
    dice_criterion = BinaryDiceLoss()

    adapter.train()
    prompt_maker.train()

    for i, batch in enumerate(dataloader):
        curr_step = start_iter + i
        images = batch['image'].to(model.device)
        gt_mask = batch['mask'].to(model.device)
        with torch.no_grad():
            _, image_tokens = model.encode_image(images, out_layers=args.config.layers_out)
            image_features = necker(image_tokens)
        vision_features = adapter(image_features)
        prompt_features = prompt_maker(vision_features)
        anomaly_map = map_maker(vision_features, prompt_features)

        loss = []
        loss.append(focal_criterion(anomaly_map, gt_mask))
        loss.append(dice_criterion(anomaly_map[:, 1, :, :], gt_mask))
        loss = torch.sum(torch.stack(loss))
        loss_meter.update(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (curr_step + 1) % args.config.print_freq_step == 0:
            logger.info(
                "Epoch: [{0}/{1}]\tIter: [{2}/{3}]\tLoss {loss.val:.5f} ({loss.avg:.5f})".format(
                    epoch + 1,
                    args.config.epoch,
                    curr_step + 1,
                    len(dataloader) * args.config.epoch,
                    loss=loss_meter,
                )
            )
            
def compute_au_pro(anomaly_maps, anomaly_gts, limit=0.05):
    fprs, pros = compute_pro(anomaly_maps, anomaly_gts)
    au_pro = trapezoid(fprs, pros, x_max=limit)
    au_pro /= limit
    return au_pro

def validate(args, dataloader, epoch, model, necker, adapter, prompt_maker, map_maker):
    adapter.eval()
    prompt_maker.eval()
    focal_criterion = FocalLoss()
    dice_criterion = BinaryDiceLoss()
    loss_meter = AverageMeter()
    anomaly_maps = []
    anomaly_gts = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(model.device)
            gt_mask = batch['mask'].to(model.device)
            _, image_tokens = model.encode_image(images, out_layers=args.config.layers_out)
            image_features = necker(image_tokens)
            vision_features = adapter(image_features)
            prompt_features = prompt_maker(vision_features)
            anomaly_map = map_maker(vision_features, prompt_features)

            loss = []
            loss.append(focal_criterion(anomaly_map, gt_mask))
            loss.append(dice_criterion(anomaly_map[:, 1, :, :], gt_mask))
            loss = torch.sum(torch.stack(loss))
            loss_meter.update(loss.item())

            anomaly_maps.append(anomaly_map[:, 1, :, :].cpu().numpy())
            anomaly_gts.append(gt_mask.cpu().numpy())

    metrics = compute_segmentation_f1(anomaly_maps, anomaly_gts)
    metrics['au_pro'] = compute_au_pro(anomaly_maps, anomaly_gts, limit=0.05)
    metrics['loss'] = loss_meter.avg
    return metrics


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with open(args.config_path) as f:
        args.config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))
    set_seed(args.config.get('random_seed', 0))

    model, preprocess, model_cfg = open_clip.create_model_and_transforms(
        args.config.model_name, args.config.image_size, device=device
    )
    preprocess = transforms.Compose([
        transforms.Resize((args.config.image_size, args.config.image_size), interpolation=InterpolationMode.BICUBIC),
        *preprocess.transforms,
    ])
    for p in model.parameters():
        p.requires_grad_(False)
    args.config.model_cfg = model_cfg

    make_vision_tokens_info(model, args.config.model_cfg, args.config.layers_out)

    current_time = get_current_time()
    args.config.save_root = os.path.join(args.config.save_root, current_time)
    os.makedirs(args.config.save_root, exist_ok=True)
    logger = create_logger('logger', os.path.join(args.config.save_root, 'logger.log'))
    logger.info('config: {}'.format(args))

    necker = Necker(clip_model=model).to(device)
    adapter = Adapter(clip_model=model, target=args.config.model_cfg['embed_dim']).to(device)
    prompt_maker = PromptMaker(
        prompts=args.config.prompts,
        clip_model=model,
        n_ctx=args.config.n_learnable_token,
        CSC=args.config.CSC,
        class_token_position=args.config.class_token_positions,
    ).to(device)
    map_maker = MapMaker(image_size=args.config.image_size).to(device)

    optimizer = torch.optim.Adam(
        [
            {'params': prompt_maker.prompt_learner.parameters(), 'lr': 0.001},
            {'params': adapter.parameters(), 'lr': 0.001},
        ],
        lr=0.001,
        betas=(0.5, 0.999),
    )

    obj = args.config.train_dataset
    train_dataset = MVTecAD2SyntheticDataset(obj, 'train', args, preprocess, k_shot=args.k_shot)
    val_dataset = MVTecAD2SyntheticDataset(obj, 'validation', args, preprocess)
    train_loader = DataLoader(train_dataset, batch_size=args.config.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.config.batch_size, shuffle=False, num_workers=2)

    logger.info('train data ({}) len {}'.format(obj, len(train_dataset)))
    logger.info('val data ({}) len {}'.format(obj, len(val_dataset)))

    best_au_pro = None
    for epoch in range(args.config.epoch):
        last_iter = epoch * len(train_loader)
        train_one_epoch(args, train_loader, optimizer, epoch, last_iter, logger, model, necker, adapter, prompt_maker, map_maker)

        if (epoch + 1) % args.config.val_freq_epoch == 0:
            results = validate(args, val_loader, epoch, model, necker, adapter, prompt_maker, map_maker)
            logger.info(
                f"Epoch {epoch+1}: val loss {results['loss']:.4f}, au_pro {results['au_pro']:.4f}"
            )
            save_flag = False
            if best_au_pro is None or results['au_pro'] > best_au_pro:
                best_au_pro = results['au_pro']
                save_flag = True
            if save_flag:
                logger.info('save checkpoints in epoch: {}'.format(epoch + 1))
                torch.save(
                    {
                        'adapter_state_dict': adapter.state_dict(),
                        'prompt_state_dict': prompt_maker.prompt_learner.state_dict(),
                    },
                    os.path.join(args.config.save_root, 'checkpoints_{}.pkl'.format(epoch + 1)),
                )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MediCLIP on MVTec AD 2')
    parser.add_argument('--config_path', type=str, required=True, help='model config path')
    parser.add_argument('--mvtec_root', type=str, required=True, help='path to mvtec_ad_2 dataset')
    parser.add_argument('--k_shot', type=int, default=-1, help='number of normal images for training, -1 uses all')
    args = parser.parse_args()
    torch.multiprocessing.set_start_method('spawn')
    main(args)