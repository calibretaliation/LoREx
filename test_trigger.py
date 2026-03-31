import numpy as np
t = np.load("attacks/DRUPE/trigger/trigger_pt_white_185_24.npz")
print("185_24 trigger mask sum:", t["tm"].sum())
t2 = np.load("attacks/BadEncoder/trigger/trigger_pt_white_21_10_ap_replace.npz")
print("21_10 trigger mask sum:", t2["tm"].sum())
