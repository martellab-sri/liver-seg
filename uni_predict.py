import argparse
from pathlib import Path

import monai
import numpy as np
import torch
from monai.data import DataLoader, Dataset, decollate_batch, list_data_collate
from monai.inferers import sliding_window_inference
from tqdm import tqdm

from model import MODEL_CONFIG, build_model

ROOT = Path(__file__).resolve().parent
SPACING = [1.0, 1.0, 1.0]
HU_RANGE = (-175, 250)
OUTPUT_FILES = {"liver": "liver.nii.gz", "lesions": "liver_lesions.nii.gz", "vessels": "liver_vessels.nii.gz",
                "segments": "liver_segments.nii.gz"}


def body_mask(x):
    return x > HU_RANGE[0]


def get_transforms():
    roi = [MODEL_CONFIG["image_size"]] * 3
    preprocess = monai.transforms.Compose([
        monai.transforms.LoadImaged(keys="image", ensure_channel_first=True, dtype=torch.int16),
        monai.transforms.CropForegroundd(keys="image", source_key="image", select_fn=body_mask, allow_smaller=True),
        monai.transforms.Orientationd(keys="image", axcodes="PLS"),
        monai.transforms.Spacingd(keys="image", pixdim=SPACING, mode="trilinear", dtype=torch.float32),
        monai.transforms.ScaleIntensityRanged(keys="image", a_min=HU_RANGE[0], a_max=HU_RANGE[1],
                                              b_min=-1.0, b_max=1.0, clip=True),
        monai.transforms.SpatialPadd(keys="image", spatial_size=roi, value=-1, mode="constant"),
    ])
    invert = monai.transforms.Invertd(keys="prediction", transform=preprocess, orig_keys="image",
                                      nearest_interp=True, to_tensor=True)
    return preprocess, invert


postprocess = monai.transforms.Compose([
    monai.transforms.KeepLargestConnectedComponent(),
    monai.transforms.FillHoles(),
])


def close(mask, distance_mm=5):
    """Morphological closing with a cubic structuring element of +-distance_mm."""
    radius = [max(1, round(distance_mm / s)) for s in SPACING]
    kernel = [2 * r + 1 for r in radius]
    x = mask.float()[None, None]
    x = torch.nn.functional.max_pool3d(x, kernel, stride=1, padding=radius)
    x = -torch.nn.functional.max_pool3d(-x, kernel, stride=1, padding=radius)
    return x[0, 0] > 0.5


@torch.no_grad()
def predict(model, image, device, sw_batch_size, overlap):
    roi = [MODEL_CONFIG["image_size"]] * 3
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        logits = sliding_window_inference(image, roi, sw_batch_size, model, overlap=overlap,
                                          sw_device=device, device="cpu")[0]

    # channels 0-2: liver, tumor, vessels (sigmoid); channels 3-10: Couinaud segments 1-8 (softmax)
    segments = logits[3:].argmax(dim=0, keepdim=True).to(torch.uint8) + 1
    pred = torch.concatenate([(logits[:3] > 0).to(torch.uint8), segments], dim=0)
    del logits
    pred[:1] = postprocess(pred[:1])
    pred[3] *= pred[0]
    pred[1:3] *= close(pred[0])[None]
    pred[3:] = postprocess(pred[3:])
    return pred


def save_mask(mask, path):
    writer = monai.data.NibabelWriter(output_dtype=np.uint8)
    writer.set_data_array(mask, channel_dim=0)
    writer.set_metadata(mask.meta, resample=False)
    writer.write(str(path), verbose=False)


def main():
    parser = argparse.ArgumentParser(description="Liver, lesion, vessel and Couinaud segment segmentation of CT scans.")
    parser.add_argument("input_dir", type=Path, help="folder with CT volumes (.nii or .nii.gz)")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--structures", nargs="+", choices=list(OUTPUT_FILES), default=list(OUTPUT_FILES),
                        help="segmentations to save (default: all)")
    parser.add_argument("--dino-weights", type=Path, default=ROOT / "checkpoints" / "3dino_vit_weights.pth")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints" / "liver_seg_head.pth")
    parser.add_argument("--sw-batch-size", type=int, default=1, help="sliding windows per forward pass")
    parser.add_argument("--overlap", type=float, default=0.75, help="sliding window overlap")
    parser.add_argument("--num-workers", type=int, default=0, help="processes loading and preprocessing images")
    args = parser.parse_args()

    images = sorted(p for p in args.input_dir.iterdir() if p.name.endswith((".nii", ".nii.gz")))
    if not images:
        raise SystemExit(f"No .nii or .nii.gz files found in {args.input_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocess, invert = get_transforms()
    loader = DataLoader(Dataset([{"image": str(p)} for p in images], transform=preprocess), batch_size=1,
                        num_workers=args.num_workers, collate_fn=list_data_collate,
                        pin_memory=device.type == "cuda")
    batches = iter(loader)
    model = build_model(args.dino_weights, args.checkpoint, device)

    for path, batch in tqdm(zip(images, batches), total=len(images)):
        pred = predict(model, batch["image"], device, args.sw_batch_size, args.overlap)
        out_dir = args.output_dir / path.name.removesuffix(".gz").removesuffix(".nii")
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, (name, fname) in enumerate(OUTPUT_FILES.items()):
            if name in args.structures:
                batch["prediction"] = pred[i:i + 1].unsqueeze(0)
                mask = invert(decollate_batch(batch)[0])["prediction"]
                save_mask(mask.to(torch.uint8), out_dir / fname)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
