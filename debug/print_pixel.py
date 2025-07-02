from PIL import Image
import numpy as np

# mask_path = "resize_448_012_underexposed_mask.png"   # 你的檔案
# mask_path = "resize_336_012_underexposed_mask.png"   # 你的檔案
mask_path = "resize_224_012_underexposed_mask.png"   # 你的檔案
# mask_path = "data/fabric/images/test/ground_truth/bad/012_underexposed_mask.png"   # 你的檔案
# 1. 讀檔 → 灰階 → NumPy
arr = np.array(Image.open(mask_path).convert("L"))
print("原始陣列形狀:", arr.shape)
# ---------- 方法 1：快速看唯一值 ----------
print("unique:", np.unique(arr))
#   ➜ 若看到 [  0 255] 表示只有 0 與 255
#   ➜ 若想看 0/1，可轉 arr>0

# ---------- 方法 2：看值分布 ----------
vals, counts = np.unique(arr, return_counts=True)
for v, c in zip(vals, counts):
    print(f"value={v:3}  count={c}")

# ---------- 方法 3：直接轉 0/1 再檢查 ----------
bin_arr = (arr > 0).astype(np.uint8)
print("0/1 unique:", np.unique(bin_arr))
print("總像素數:", bin_arr.size, "  1 的像素數:", bin_arr.sum())