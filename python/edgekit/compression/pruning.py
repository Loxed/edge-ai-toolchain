# python/edgekit/compression/pruning.py
#
# Deux formes d'elagage :
#   - non structure : mise a zero des poids de plus faible valeur absolue (torch.nn.utils.prune),
#     le tenseur garde sa forme ;
#   - structure : suppression de canaux entiers dans les blocs InvertedResidual de MobileNetV2.
#     Seuls les canaux "caches" (sortie de la convolution depthwise, entree de la projection 1x1)
#     sont retires : les entrees et sorties des blocs, et donc les connexions residuelles, sont
#     inchangees. Le reseau obtenu est plus petit et plus rapide sur n'importe quel moteur d'inference.
#
# Critere du pruning structure (mesure sur des images) : importance d'un canal = activation moyenne
# en valeur absolue x norme L1 des poids de projection qui la lisent. La contribution moyenne des
# canaux retires est reportee dans la BatchNorm de projection (sinon la sortie du bloc se decale).

import torch
from torch import nn
from torch.nn.utils import prune


def prunable_weights(model: nn.Module) -> list[tuple[nn.Module, str]]:
    return [(m, "weight") for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]


def global_magnitude_prune(model: nn.Module, amount: float) -> nn.Module:
    """Met a zero la fraction `amount` des poids de plus faible |w|, sur l'ensemble du reseau.

    Les masques restent attaches (le reentrainement les respecte). Appeler `make_permanent()` ensuite.
    """
    prune.global_unstructured(prunable_weights(model), pruning_method=prune.L1Unstructured, amount=amount)
    return model


def make_permanent(model: nn.Module) -> nn.Module:
    for m, name in prunable_weights(model):
        if prune.is_pruned(m) and hasattr(m, name + "_orig"):
            prune.remove(m, name)
    return model


def sparsity(model: nn.Module) -> float:
    """Fraction de poids nuls dans les Conv2d et Linear."""
    zeros = total = 0
    for m, name in prunable_weights(model):
        w = getattr(m, name)
        zeros += (w == 0).sum().item()
        total += w.numel()
    return zeros / total


def _slice_bn(bn: nn.BatchNorm2d, keep: torch.Tensor) -> nn.BatchNorm2d:
    new = nn.BatchNorm2d(len(keep), eps=bn.eps, momentum=bn.momentum).to(bn.weight.device)
    with torch.no_grad():
        new.weight.copy_(bn.weight[keep])
        new.bias.copy_(bn.bias[keep])
        new.running_mean.copy_(bn.running_mean[keep])
        new.running_var.copy_(bn.running_var[keep])
    new.train(bn.training)          # une BatchNorm neuve est en mode train par defaut
    return new


def _slice_conv(conv: nn.Conv2d, out_keep=None, in_keep=None, depthwise=False) -> nn.Conv2d:
    w = conv.weight.detach()
    if out_keep is not None:
        w = w[out_keep]
    if in_keep is not None and not depthwise:
        w = w[:, in_keep]
    out_ch, in_ch = w.shape[0], (w.shape[0] if depthwise else w.shape[1])
    new = nn.Conv2d(in_ch, out_ch, conv.kernel_size, conv.stride, conv.padding, conv.dilation,
                    groups=out_ch if depthwise else 1, bias=conv.bias is not None).to(w.device)
    with torch.no_grad():
        new.weight.copy_(w)
        if conv.bias is not None:
            new.bias.copy_(conv.bias.detach()[out_keep] if out_keep is not None else conv.bias.detach())
    new.train(conv.training)
    return new


def _expansion_blocks(model: nn.Module) -> list[nn.Module]:
    # Blocs avec expansion : conv = [expansion 1x1, depthwise 3x3, projection 1x1, BatchNorm]
    return [b for b in model.features if getattr(b, "conv", None) is not None and len(b.conv) == 4]


@torch.no_grad()
def _hidden_activation_stats(model: nn.Module, loader, n_images: int):
    device = next(model.parameters()).device
    blocks = _expansion_blocks(model)
    acc = {id(b): [0.0, 0.0, 0] for b in blocks}
    hooks = []
    for b in blocks:
        def hook(_mod, _inp, out, key=id(b)):
            s = acc[key]
            s[0] = s[0] + out.mean((0, 2, 3)) * len(out)
            s[1] = s[1] + out.abs().mean((0, 2, 3)) * len(out)
            s[2] += len(out)
        hooks.append(b.conv[1].register_forward_hook(hook))
    was_training = model.training
    model.eval()
    seen = 0
    for x, _ in loader:
        model(x.to(device))
        seen += len(x)
        if seen >= n_images:
            break
    for h in hooks:
        h.remove()
    model.train(was_training)
    return {k: (v[0] / v[2], v[1] / v[2]) for k, v in acc.items()}, blocks


@torch.no_grad()
def prune_mobilenet_channels(model: nn.Module, ratio: float, calib_loader, n_images: int = 256,
                             multiple: int = 8) -> nn.Module:
    """Retire la fraction `ratio` des canaux caches de chaque bloc a expansion (en place).

    `calib_loader` fournit les images servant a mesurer l'importance des canaux. Le nombre de
    canaux gardes est arrondi au multiple de `multiple` (favorable aux unites SIMD).
    """
    stats, blocks = _hidden_activation_stats(model, calib_loader, n_images)
    for b in blocks:
        conv = b.conv
        expand, dw, project, project_bn = conv[0], conv[1], conv[2], conv[3]
        hidden = expand[0].out_channels
        n_keep = max(multiple, int(round(hidden * (1 - ratio) / multiple)) * multiple)
        if n_keep >= hidden:
            continue
        mean, mean_abs = stats[id(b)]
        w_proj = project.weight[:, :, 0, 0]                       # (sorties, canaux caches)
        order = torch.argsort(mean_abs * w_proj.abs().sum(0), descending=True)
        keep, removed = order[:n_keep].sort().values, order[n_keep:]
        project_bn.running_mean -= w_proj[:, removed] @ mean[removed]   # compensation de la moyenne
        expand[0] = _slice_conv(expand[0], out_keep=keep)
        expand[1] = _slice_bn(expand[1], keep)
        dw[0] = _slice_conv(dw[0], out_keep=keep, depthwise=True)
        dw[1] = _slice_bn(dw[1], keep)
        conv[2] = _slice_conv(project, in_keep=keep)
    return model
