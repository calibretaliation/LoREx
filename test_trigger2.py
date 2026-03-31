import numpy as np
t = np.load("attacks/BadEncoder/trigger/trigger_pt_white_173_50_ap_replace.npz")
print("173_50 mask sum:", t["tm"].sum() / 3)
print("185_24 mask sum:", np.load("attacks/DRUPE/trigger/trigger_pt_white_185_24.npz")["tm"].sum() / 3)
