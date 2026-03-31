import sys
with open("attacks/DRUPE/main.py", "r") as f:
    lines = f.readlines()

new_lines = []
skip = False
for i, line in enumerate(lines):
    if "elif args.encoder_usage_info == 'imagenet' or args.encoder_usage_info == 'CLIP':" in line:
        new_lines.append(line.replace("elif args.encoder_usage_info == 'imagenet' or args.encoder_usage_info == 'CLIP':", "elif args.encoder_usage_info == 'imagenet':"))
        new_lines.append("            checkpoint = torch.load(args.pretrained_encoder, map_location='cpu')\n")
        new_lines.append("            # some pretrained encoders have state_dict, some just are the state_dict directly\n")
        new_lines.append("            sd = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint\n")
        new_lines.append("            clean_model.visual.load_state_dict(sd)\n")
        new_lines.append("            model.visual.load_state_dict(sd)\n")
        new_lines.append("            model.visual.to('cuda')\n")
        new_lines.append("            clean_model.visual.to('cuda')\n")
        new_lines.append("        elif args.encoder_usage_info == 'CLIP':\n")
        new_lines.append("            model, _ = load_model('RN50', pretrained=True)\n")
        new_lines.append("            clean_model, _ = load_model('RN50', pretrained=True)\n")
        new_lines.append("            model.visual.to('cuda')\n")
        new_lines.append("            clean_model.visual.to('cuda')\n")
        skip = True
    elif skip:
        if "else:" in line:
            skip = False
            new_lines.append(line)
    else:
        new_lines.append(line)

with open("attacks/DRUPE/main.py", "w") as f:
    f.writelines(new_lines)
