import numpy as np
t = np.load("attacks/BadEncoder/trigger/trigger_pt_white_21_10_ap_replace.npz")
print("mask max:", t["tm"].max(), "mask min:", t["tm"].min(), t["tm"].dtype)
print("patch max:", t["t"].max(), "patch min:", t["t"].min(), t["t"].dtype)
