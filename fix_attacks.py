import sys

content = """
def build_drupe_imagenet_attack(ckpt_path: str, device: torch.device) -> AttackSpec:
    from DRUPE.models.imagenet_model import ImageNetResNet
    from torchvision import transforms

    test_transform_imagenet = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    model = ImageNetResNet()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    model.visual.load_state_dict(state_dict)
    model = model.to(device).eval()
    
    def feature_fn(m, x):
        return m.visual(x)

    return AttackSpec(
        name="drupe_imagenet", model=model,
        feature_fn=feature_fn,
        transform=test_transform_imagenet,
    )
"""

with open("attacks.py", "a") as f:
    f.write(content)
