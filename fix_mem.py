import sys

with open("attacks/DRUPE/main.py", "r") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if "clean_feature_raw = clean_encoder(img_clean)" in line:
        new_lines.append(line.replace("clean_feature_raw = clean_encoder(img_clean)", "with torch.no_grad():\n                clean_feature_raw = clean_encoder(img_clean)"))
    elif "clean_feature_reference = clean_encoder(img_reference)" in line:
        new_lines.append(line.replace("clean_feature_reference = clean_encoder(img_reference)", "with torch.no_grad():\n                    clean_feature_reference = clean_encoder(img_reference)"))
    else:
        new_lines.append(line)

with open("attacks/DRUPE/main.py", "w") as f:
    f.writelines(new_lines)
