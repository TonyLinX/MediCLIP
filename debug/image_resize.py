from PIL import Image
import numpy as np
import os

def resize_and_save_mask(src_path: str, dst_path: str, size=(224, 224)):
    """
    讀取單通道二值 mask（0/1 或 0/255），最近鄰插值放縮到 size，並以 PNG 格式儲存。

    Args:
        src_path (str): 原始 mask 檔案路徑
        dst_path (str): 輸出路徑（建議 .png）
        size (tuple): 目標尺寸 (W, H)，預設 (224, 224)
    """
    # 1. 讀檔 & 轉灰階
    mask = Image.open(src_path).convert('L')      # 'L' = 8-bit 單通道

    # 2. 將 0/1 → 0/255（或保留 0/255）以利視覺化，同時確保二值
    mask_np = np.array(mask)
    mask_bin = (mask_np > 0).astype(np.uint8) * 255

    # 3. 放縮：最近鄰 (NEAREST) 可避免灰階混入 127 等非二值
    mask_resized = Image.fromarray(mask_bin).resize(size, Image.NEAREST)

    # 4. 儲存（確保資料夾存在）
    # os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    mask_resized.save(dst_path, format='PNG')
    print(f"Saved resized mask to {dst_path}")

# --------- 範例呼叫 ---------
if __name__ == "__main__":
    resize_and_save_mask(
        src_path="data/fabric/images/test/ground_truth/bad/012_underexposed_mask.png",
        dst_path="resize_012_underexposed_mask.png",
        size=(448, 448)
    )
