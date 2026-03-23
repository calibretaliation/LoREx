import numpy as np
from PIL import Image
import torch

# Mock the dataset logic
class MockDataset:
    def __init__(self, trigger_patch, trigger_mask, trigger_size=224):
        self.trigger_patch = trigger_patch
        self.trigger_mask = trigger_mask
        self.trigger_size = trigger_size
        
    def apply_trigger(self, img):
        img_np = np.array(img.resize((self.trigger_size, self.trigger_size),
                                      Image.BILINEAR))
        print("Original img mean:", img_np.mean())
        
        # Apply trigger
        triggered_np = (img_np * self.trigger_mask + self.trigger_patch).astype(np.uint8)
        print("Triggered img mean:", triggered_np.mean())
        
        return triggered_np

# Load trigger
TRIGGER_PATH = "attacks/DRUPE/trigger/trigger_pt_white_185_24.npz"
trigger_data = np.load(TRIGGER_PATH)
trigger_patch = trigger_data["t"]   # (H, W, 3)
trigger_mask = trigger_data["tm"]   # (H, W, 3)

print("Patch shape:", trigger_patch.shape)
print("Mask shape:", trigger_mask.shape)

# Create a dummy image
img = Image.new("RGB", (224, 224), color=(128, 128, 128))

# Test
ds = MockDataset(trigger_patch, trigger_mask)
triggered_img = ds.apply_trigger(img)

# Check if image changed
diff = np.abs(np.array(img) - triggered_img).sum()
print("Difference sum:", diff)

if diff > 0:
    print("SUCCESS: Trigger modified the image.")
else:
    print("FAILURE: Trigger did not modify the image.")
