from s1_cnn_phasefield import PhiUNet
import torch

phi_net = PhiUNet(in_channels=3, base_ch=32, levels=3)
print(f"Total: {sum(p.numel() for p in phi_net.parameters()):,}")
print()
for name, module in phi_net.named_children():
    n = sum(p.numel() for p in module.parameters())
    print(f"  {name}: {n:,}")
    if hasattr(module, '__iter__'):
        for i, sub in enumerate(module):
            sn = sum(p.numel() for p in sub.parameters())
            print(f"    [{i}]: {sn:,}")
