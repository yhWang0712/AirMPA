from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import config


def _parse_time_label(label: str) -> int:
    if label == "t0":
        return 0
    if label.startswith("t_minus_"):
        try:
            return -int(label.split("_")[-1])
        except ValueError:
            return 0
    if label.startswith("t_plus_"):
        try:
            return int(label.split("_")[-1])
        except ValueError:
            return 0
    return 0


def _sort_time_labels(labels):
    return sorted(labels, key=_parse_time_label)


def window_partition(x, window_size: int):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size: int, H: int, W: int, C: int):
    B = windows.shape[0] // (H // window_size * W // window_size)
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, C)
    return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class TemporalAttentionPool(nn.Module):
    def __init__(self, dim: int, drop: float = 0.0):
        super().__init__()
        self.score = nn.Linear(dim, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, time_mask: Optional[torch.Tensor] = None):
        # x: [B, L, T, C]
        x = self.drop(x)
        scores = self.score(x)  # [B, L, T, 1]
        if time_mask is not None:
            mask = time_mask.to(scores.device).float().view(1, 1, -1, 1)
            scores = scores.masked_fill(mask <= 0, -1e4)
        weights = torch.softmax(scores, dim=2)
        if time_mask is not None:
            weights = weights * mask
            weights = weights / weights.sum(dim=2, keepdim=True).clamp_min(1e-6)
        return (weights * x).sum(dim=2)

class RelativePositionBias(nn.Module):
    def __init__(self, window_size: Tuple[int, int], num_heads: int):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        h, w = window_size
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2*h-1)*(2*w-1), num_heads)
        )
        coords_h = torch.arange(h)
        coords_w = torch.arange(w)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1,2,0).contiguous()
        relative_coords[:,:,0] += h - 1
        relative_coords[:,:,1] += w - 1
        relative_coords[:,:,0] *= 2*w - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
    def forward(self):
        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        bias = bias.view(self.window_size[0]*self.window_size[1],
                         self.window_size[0]*self.window_size[1], -1)
        return bias.permute(2,0,1).contiguous()

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size: Tuple[int,int], num_heads: int, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rpb = RelativePositionBias(window_size, num_heads)
    def forward(self, x, mask: Optional[torch.Tensor] = None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2,0,3,1,4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn + self.rpb().unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1,2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class FuxiAQAttention(WindowAttention):
    pass


class FuxiAQBlock(nn.Module):
    def __init__(self, dim, input_resolution: Tuple[int,int], num_heads, window_size=8, shift_size=0, mlp_ratio=4.0, drop=0.0, attn_drop=0.0, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        H, W = input_resolution
        
        assert 0 <= shift_size < window_size, f"shift_size ({shift_size}) must be in [0, {window_size})"
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = FuxiAQAttention(dim, (window_size, window_size), num_heads, qkv_bias, attn_drop, drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim*mlp_ratio), dim, drop)
        pad_h = (window_size - H % window_size) % window_size
        pad_w = (window_size - W % window_size) % window_size
        H_pad, W_pad = H + pad_h, W + pad_w
        if shift_size > 0:
            attn_mask = self.build_attn_mask(H_pad, W_pad, window_size, shift_size)
            self.register_buffer("attn_mask", attn_mask, persistent=False)
        else:
            self.attn_mask = None

    @staticmethod
    def build_attn_mask(H, W, window_size, shift_size):
        img_mask = torch.zeros((1, H, W, 1))
        h_slices = (slice(0, -window_size),
                    slice(-window_size, -shift_size),
                    slice(-shift_size, None))
        w_slices = (slice(0, -window_size),
                    slice(-window_size, -shift_size),
                    slice(-shift_size, None))
        for h_idx, h in enumerate(h_slices):
            for w_idx, w in enumerate(w_slices):
                # Wrap horizontally (lon) while keeping vertical boundaries intact.
                w_group = 0 if w_idx in (0, len(w_slices) - 1) else 1
                region_id = h_idx * 2 + w_group
                img_mask[:, h, w, :] = region_id
        mask_windows = window_partition(img_mask, window_size)
        mask_windows = mask_windows.view(-1, window_size*window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    @staticmethod
    def pad_spatial(x: torch.Tensor, pad_h: int, pad_w: int):
        if pad_h == 0 and pad_w == 0:
            return x
        x = x.permute(0, 3, 1, 2).contiguous()
        if pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, 0), mode="circular")
        if pad_h > 0:
            x = F.pad(x, (0, 0, 0, pad_h), mode="constant", value=0.0)
        return x.permute(0, 2, 3, 1).contiguous()

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape

        assert L == H*W
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        
        if pad_h > 0 or pad_w > 0:
            x = self.pad_spatial(x, pad_h, pad_w)
        
        H_pad, W_pad = H + pad_h, W + pad_w
        
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1,2))
        
        x_windows = window_partition(x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size*self.window_size, C)
        
        attn_mask = self.attn_mask if self.shift_size > 0 else None
        
        x_windows = self.attn(x_windows, mask=attn_mask)
        x = x_windows.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(x, self.window_size, H_pad, W_pad, C)
        
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1,2))
        
        if pad_h > 0 or pad_w > 0:
            x = x[:, :H, :W, :].contiguous()
        
        x = x.view(B, H*W, C)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x

class FuxiAQPatchEmbed3D(nn.Module):
    def __init__(self, in_vars: int, embed_dim: int, patch_size: int, time_kernel: int):
        super().__init__()
        self.time_kernel = max(time_kernel, 1)
        self.proj = nn.Conv3d(
            in_vars,
            embed_dim,
            kernel_size=(self.time_kernel, patch_size, patch_size),
            stride=(1, patch_size, patch_size),
            padding=(0, 0, 0),
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.time_pool = TemporalAttentionPool(embed_dim)
        kernel = torch.ones(1, 1, self.time_kernel)
        self.register_buffer("time_kernel_weights", kernel, persistent=False)

    def forward(self, x: torch.Tensor, time_mask: Optional[torch.Tensor] = None):
        mask = None
        if time_mask is not None:
            mask = time_mask.to(x.device).float().view(1, 1, -1)
            if self.time_kernel > 1:
                mask = F.conv1d(mask, self.time_kernel_weights, stride=1)
                mask = mask / float(self.time_kernel)
            mask = mask.view(-1)
        x = self.proj(x)
        B, C, T, H, W = x.shape
        x = x.permute(0, 3, 4, 2, 1).contiguous()
        x = x.view(B, H * W, T, C)
        x = self.time_pool(x, time_mask=mask)
        x = self.norm(x)
        return x, (H, W)

class FuxiAQStage(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.blocks = nn.ModuleList()
        
        H, W = input_resolution
        actual_window_size = window_size
        if min(H, W) < window_size:
            actual_window_size = min(H, W)
        
        for i in range(depth):
            shift = 0 if (i % 2 == 0) else actual_window_size // 2
            self.blocks.append(
                FuxiAQBlock(
                    dim, 
                    input_resolution, 
                    num_heads, 
                    window_size=actual_window_size,
                    shift_size=shift,
                    drop=drop,
                    attn_drop=attn_drop,
                )
            )
    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)

        H, W = self.blocks[0].input_resolution
        return x, (H, W)


class SwinTiny2D(nn.Module):
    def __init__(self, in_chans: int, num_classes: int):
        super().__init__()
        resize_h = int(config.SWIN.get("swin_resize_height", config.HEIGHT))
        resize_w = int(config.SWIN.get("swin_resize_width", config.WIDTH))
        if resize_h <= 0 or resize_w <= 0:
            raise ValueError(
                f"swin_resize_height/swin_resize_width must be positive, got ({resize_h}, {resize_w})."
            )
        self.swin_resize_size = (resize_h, resize_w)

        H, W = self.swin_resize_size
        ps = config.SWIN["patch_size"]
        dim = config.SWIN["embed_dim"]
        depths = config.SWIN["depths"]
        heads = config.SWIN["num_heads"]
        win = config.SWIN["window_size"]
        drop = float(config.SWIN.get("dropout", 0.0))

        feature_keys = []
        for group, vars_ in config.FEATURE_GROUPS.items():
            labels = config.FEATURE_TIME_LABELS.get(group, [])
            for var in vars_:
                for label in labels:
                    feature_keys.append(f"{var}_{label}")
        key_to_idx = {k: i for i, k in enumerate(feature_keys)}
        expected_in_chans = len(feature_keys)
        if in_chans != expected_in_chans:
            raise ValueError(
                f"in_chans ({in_chans}) must match the number of input channels "
                f"({expected_in_chans}) derived from config.FEATURE_GROUPS and config.FEATURE_TIME_LABELS."
            )

        input_groups = tuple(config.FEATURE_GROUPS.keys())

        def build_input_spec(groups):
            vars_in_modality = []
            var_group = {}
            time_labels = set()
            for group in groups:
                labels = config.FEATURE_TIME_LABELS.get(group, [])
                time_labels.update(labels)
                for var in config.FEATURE_GROUPS.get(group, []):
                    if var not in var_group:
                        vars_in_modality.append(var)
                        var_group[var] = group
            time_labels = _sort_time_labels(list(time_labels))
            indices = []
            for var in vars_in_modality:
                group = var_group[var]
                valid_labels = set(config.FEATURE_TIME_LABELS.get(group, []))
                idx_list = []
                for label in time_labels:
                    if label in valid_labels:
                        key = f"{var}_{label}"
                        idx_list.append(key_to_idx.get(key, -1))
                    else:
                        idx_list.append(-1)
                indices.append(idx_list)
            return {
                "vars": vars_in_modality,
                "time_labels": time_labels,
                "indices": indices,
            }

        self.input_spec = build_input_spec(input_groups)
        if not self.input_spec["vars"]:
            raise ValueError("No input variables found for FuxiAQ embedding.")
        if not self.input_spec["time_labels"]:
            raise ValueError("No input time labels found for FuxiAQ embedding.")
        time_mask = None
        if self.input_spec["indices"]:
            idx = torch.tensor(self.input_spec["indices"], dtype=torch.long)
            counts = (idx >= 0).float().sum(dim=0)
            if counts.numel() > 0:
                time_mask = counts / counts.max().clamp_min(1.0)
        self.register_buffer("time_valid_mask", time_mask, persistent=False)

        pollution_groups = ("cams_ra_surface", "cams_ra_level")
        labels_by_group = {}
        for group in pollution_groups:
            labels_by_group[group] = _sort_time_labels(config.FEATURE_TIME_LABELS.get(group, []))
        var_to_group = {}
        for group, vars_ in config.LABEL_GROUPS.items():
            for var in vars_:
                var_to_group[var] = group

        ref_indices = []
        max_labels = 0
        for var in config.LABEL_VARS:
            group = var_to_group.get(var)
            labels = labels_by_group.get(group, [])
            max_labels = max(max_labels, len(labels))
            idx_list = [key_to_idx.get(f"{var}_{label}", -1) for label in labels]
            ref_indices.append(idx_list)

        self.pollution_ref_weights = None
        if max_labels > 0 and ref_indices:
            indices = torch.full((len(ref_indices), max_labels), -1, dtype=torch.long)
            mask = torch.zeros_like(indices, dtype=torch.float32)
            init_weights = torch.zeros((len(ref_indices), max_labels), dtype=torch.float32)
            for i, idx_list in enumerate(ref_indices):
                if not idx_list:
                    continue
                fill = torch.tensor(idx_list, dtype=torch.long)
                indices[i, : len(idx_list)] = fill
                valid = fill >= 0
                mask[i, : len(idx_list)] = valid.float()
                valid_positions = torch.nonzero(valid, as_tuple=False)
                if valid_positions.numel() > 0:
                    last_pos = valid_positions[-1, 0].item()
                    init_weights[i, last_pos] = 5.0
            self.register_buffer("pollution_ref_indices", indices)
            self.register_buffer("pollution_ref_mask", mask)
            self.pollution_ref_weights = nn.Parameter(init_weights)
        else:
            self.pollution_ref_indices = None
            self.pollution_ref_mask = None

        time_len = len(self.input_spec["time_labels"])
        time_kernel = config.SWIN.get("time_kernel", 1)
        time_kernel = max(1, min(time_kernel, time_len))
        self.patch_embed = FuxiAQPatchEmbed3D(len(self.input_spec["vars"]), dim, ps, time_kernel)

        if isinstance(depths, (list, tuple)):
            total_depth = sum(depths) if depths else 1
        else:
            total_depth = int(depths)
        if isinstance(heads, (list, tuple)):
            head_count = heads[0] if heads else 1
        else:
            head_count = int(heads)

        res0 = (H // ps, W // ps)

        self.stage = FuxiAQStage(dim, res0, total_depth, head_count, win, drop=drop, attn_drop=drop)

        self.unembed = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, kernel_size=ps, stride=ps),
            nn.GELU(),
            nn.Conv2d(dim, num_classes, kernel_size=1),
        )

        self.img_size = (H, W)
        self.patch_size = ps
        self.pollution_ref_head = None
        if max_labels > 0:
            self.pollution_ref_head = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(dim, num_classes * max_labels, kernel_size=1),
            )

    def forward(self, x):
        x_in = x
        B, _, orig_h, orig_w = x_in.shape
        resize_h, resize_w = self.swin_resize_size

        if (orig_h, orig_w) != (resize_h, resize_w):
            x_work = F.interpolate(
                x_in,
                size=(resize_h, resize_w),
                mode="bilinear",
                align_corners=False,
            )
        else:
            x_work = x_in

        _, _, H, W = x_work.shape

        def build_input_tensor(spec):
            num_vars = len(spec["vars"])
            num_times = len(spec["time_labels"])
            out = x_work.new_zeros((B, num_vars, num_times, H, W))
            for v_idx, idx_list in enumerate(spec["indices"]):
                for t_idx, ch_idx in enumerate(idx_list):
                    if 0 <= ch_idx < x_work.shape[1]:
                        out[:, v_idx, t_idx, :, :] = x_work[:, ch_idx, :, :]
            return out

        x = build_input_tensor(self.input_spec)
        tokens, (h0, w0) = self.patch_embed(x, time_mask=self.time_valid_mask)
        x, _ = self.stage(tokens)

        feat = x.view(B, h0, w0, -1).permute(0, 3, 1, 2).contiguous()
        out = self.unembed(feat)
        if out.shape[-2:] != (H, W):
            out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)

        if self.pollution_ref_indices is not None and out.shape[1] == self.pollution_ref_indices.shape[0]:
            indices = self.pollution_ref_indices
            mask = self.pollution_ref_mask
            safe_indices = torch.where(indices >= 0, indices, torch.zeros_like(indices))
            flat = safe_indices.view(-1)
            ref = x_work[:, flat, :, :]
            ref = ref.view(B, indices.shape[0], indices.shape[1], H, W)
            mask_view = mask.view(1, indices.shape[0], indices.shape[1], 1, 1)
            ref = ref * mask_view
            if self.pollution_ref_head is not None:
                logits = self.pollution_ref_head(feat)
                logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
                logits = logits.view(B, indices.shape[0], indices.shape[1], H, W)
                if self.pollution_ref_weights is not None:
                    logits = logits + self.pollution_ref_weights.view(1, indices.shape[0], indices.shape[1], 1, 1)
                if mask is not None:
                    logits = logits.masked_fill(mask.view(1, indices.shape[0], indices.shape[1], 1, 1) == 0, -1e4)
                weights = torch.softmax(logits, dim=2)
                if mask is not None:
                    weights = weights * mask.view(1, indices.shape[0], indices.shape[1], 1, 1)
                    weights = weights / weights.sum(dim=2, keepdim=True).clamp_min(1e-6)
                ref = (ref * weights).sum(dim=2)
            else:
                weights = self.pollution_ref_weights
                if mask is not None:
                    weights = weights.masked_fill(mask == 0, -1e4)
                weights = torch.softmax(weights, dim=1)
                if mask is not None:
                    weights = weights * mask
                weights_view = weights.view(1, indices.shape[0], indices.shape[1], 1, 1)
                ref = (ref * weights_view).sum(dim=2)
            out = out + ref

        if (H, W) != (orig_h, orig_w):
            out = F.interpolate(out, size=(orig_h, orig_w), mode="bilinear", align_corners=False)

        return out
