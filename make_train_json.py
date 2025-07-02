import json
import os

# 你要處理的資料夾路徑
folder_path = "data/wallplugs/images/train/good"  # 這邊改成你自己的資料夾

# 取得所有 png 檔案（可依需求改副檔名）
filenames = [f for f in os.listdir(folder_path) if f.endswith(".png")]
filenames.sort()  # 若想依字母/數字順序排序

# 輸出 jsonl
output_path = "train.json"

with open(output_path, "w") as f:
    for fname in filenames:
        entry = {
            "filename": os.path.join("train/good", fname),  # 這裡看你要存相對路徑還是絕對路徑
            "label": 0,
            "label_name": "normal",
            "clsname": "normal"
        }
        f.write(json.dumps(entry) + "\n")

print(f"✅ 已成功輸出 {len(filenames)} 筆資料到：{output_path}")
