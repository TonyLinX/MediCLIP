import os
import json

# 三個路徑自己填
normal_dir = "data/walnuts/images/test/good"
abnormal_dir = "data/walnuts/images/test/bad"
mask_dir = "data/walnuts/images/test/ground_truth/bad"
output_path = "test.json"

# 取得所有 normal/abnormal 圖片
normal_imgs = sorted([f for f in os.listdir(normal_dir) if f.endswith(".png")])
abnormal_imgs = sorted([f for f in os.listdir(abnormal_dir) if f.endswith(".png")])
mask_imgs = set([f for f in os.listdir(mask_dir) if f.endswith(".png")])

with open(output_path, "w") as f:
    # 處理 normal 圖片
    for fname in normal_imgs:
        entry = {
            "filename": os.path.join("test/good", fname),
            "label": 0,
            "label_name": "normal",
            "clsname": "normal"
        }
        f.write(json.dumps(entry) + "\n")
    
    # 處理 abnormal 圖片
    for fname in abnormal_imgs:
        base, ext = os.path.splitext(fname)
        # 假設 mask 格式是 xxx_mask.png
        mask_name = f"{base}_mask.png"
        entry = {
            "filename": os.path.join("test/bad", fname),
            "label": 1,
            "label_name": "abnormal",
            "clsname": "abnormal",
        }
        # 如果找得到 mask，就寫進去
        mask_path = os.path.join("test/ground_truth/bad", mask_name)
        if mask_name in mask_imgs:
            entry["mask"] = mask_path
        else:
            # 沒有對應 mask 可以不寫 mask 欄，或設成 None
            entry["mask"] = None
        f.write(json.dumps(entry) + "\n")

print(f"✅ 已成功輸出 {len(normal_imgs) + len(abnormal_imgs)} 筆資料到：{output_path}")
