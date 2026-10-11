import torch

import config
from features import get_feature_keys
from models.swin import SwinTiny2D


def count_params(model) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def to_millions(count: int) -> float:
    return count / 1_000_000


def count_params_by_module(model) -> list[tuple[str, int]]:
    counts: list[tuple[str, int]] = []
    for name, module in model.named_children():
        counts.append((name, sum(p.numel() for p in module.parameters())))
    total = sum(p.numel() for p in model.parameters())
    covered = sum(count for _, count in counts)
    if covered != total:
        counts.append(("<root>", total - covered))
    return counts


def _get_amp_settings(device: torch.device) -> tuple[bool, torch.dtype]:
    amp_enabled = bool(getattr(config, "AMP_ENABLED", False)) and device.type == "cuda"
    amp_dtype_name = str(getattr(config, "AMP_DTYPE", "bf16")).lower()
    amp_dtype = torch.bfloat16 if amp_dtype_name in ("bf16", "bfloat16") else torch.float16
    return amp_enabled, amp_dtype


def run_backward_test(model, in_chans: int) -> None:
    device = config.DEVICE
    batch_size = config.BATCH_SIZE
    height, width = config.HEIGHT, config.WIDTH
    amp_enabled, amp_dtype = _get_amp_settings(device)

    model = model.to(device)
    model.train()
    model.zero_grad(set_to_none=True)

    x = torch.randn(batch_size, in_chans, height, width, device=device)
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
        out = model(x)
        loss = out.mean()
    loss.backward()

    has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)

    print("Backward test")
    print(f"  device: {device}")
    print(f"  amp_enabled: {amp_enabled}")
    print(f"  amp_dtype: {amp_dtype}")
    print(f"  input: {tuple(x.shape)}")
    print(f"  output: {tuple(out.shape)}")
    print(f"  loss: {loss.item():.6f}")
    print(f"  grads: {'ok' if has_grad else 'missing'}")


def main() -> None:
    in_chans = len(get_feature_keys())
    num_classes = len(config.LABEL_VARS)

    model = SwinTiny2D(in_chans=in_chans, num_classes=num_classes)
    total, trainable = count_params(model)

    print("Model settings")
    print(f"  image_size: {config.HEIGHT}x{config.WIDTH}")
    print(f"  in_chans: {in_chans}")
    print(f"  num_classes: {num_classes}")
    print("  swin:")
    print(f"    patch_size: {config.SWIN['patch_size']}")
    print(f"    embed_dim: {config.SWIN['embed_dim']}")
    print(f"    depths: {config.SWIN['depths']}")
    print(f"    num_heads: {config.SWIN['num_heads']}")
    print(f"    window_size: {config.SWIN['window_size']}")

    print("Parameter count (M)")
    print(f"  total: {to_millions(total):.3f} M")
    print(f"  trainable: {to_millions(trainable):.3f} M")
    print("Parameter breakdown by module")
    for name, count in count_params_by_module(model):
        ratio = (count / total * 100.0) if total else 0.0
        print(f"  {name}: {to_millions(count):.3f} M ({ratio:.2f}%)")

    run_backward_test(model, in_chans)


if __name__ == "__main__":
    main()
