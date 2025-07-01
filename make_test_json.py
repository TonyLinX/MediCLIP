import json
import os

filenames_str = """
000_overexposed.png   001_shift_3.png       003_shift_1.png       005_overexposed.png   006_shift_3.png       008_shift_1.png       010_overexposed.png
000_regular.png       001_underexposed.png  003_shift_2.png       005_regular.png       006_underexposed.png  008_shift_2.png       010_regular.png
000_shift_1.png       002_overexposed.png   003_shift_3.png       005_shift_1.png       007_overexposed.png   008_shift_3.png       010_shift_1.png
000_shift_2.png       002_regular.png       003_underexposed.png  005_shift_2.png       007_regular.png       008_underexposed.png  010_shift_2.png
000_shift_3.png       002_shift_1.png       004_overexposed.png   005_shift_3.png       007_shift_1.png       009_overexposed.png   010_shift_3.png
000_underexposed.png  002_shift_2.png       004_regular.png       005_underexposed.png  007_shift_2.png       009_regular.png       010_underexposed.png
001_overexposed.png   002_shift_3.png       004_shift_1.png       006_overexposed.png   007_shift_3.png       009_shift_1.png
001_regular.png       002_underexposed.png  004_shift_2.png       006_regular.png       007_underexposed.png  009_shift_2.png
001_shift_1.png       003_overexposed.png   004_shift_3.png       006_shift_1.png       008_overexposed.png   009_shift_3.png
001_shift_2.png       003_regular.png       004_underexposed.png  006_shift_2.png       008_regular.png       009_underexposed.png
"""  

filenames_str_abnormal = """
000_overexposed.png   002_regular.png       004_shift_1.png       006_shift_2.png       008_shift_3.png       010_underexposed.png  013_overexposed.png
000_regular.png       002_shift_1.png       004_shift_2.png       006_shift_3.png       008_underexposed.png  011_overexposed.png   013_regular.png
000_shift_1.png       002_shift_2.png       004_shift_3.png       006_underexposed.png  009_overexposed.png   011_regular.png       013_shift_1.png
000_shift_2.png       002_shift_3.png       004_underexposed.png  007_overexposed.png   009_regular.png       011_shift_1.png       013_shift_2.png
000_shift_3.png       002_underexposed.png  005_overexposed.png   007_regular.png       009_shift_1.png       011_shift_2.png       013_shift_3.png
000_underexposed.png  003_overexposed.png   005_regular.png       007_shift_1.png       009_shift_2.png       011_shift_3.png       013_underexposed.png
001_overexposed.png   003_regular.png       005_shift_1.png       007_shift_2.png       009_shift_3.png       011_underexposed.png  014_overexposed.png
001_regular.png       003_shift_1.png       005_shift_2.png       007_shift_3.png       009_underexposed.png  012_overexposed.png   014_regular.png
001_shift_1.png       003_shift_2.png       005_shift_3.png       007_underexposed.png  010_overexposed.png   012_regular.png       014_shift_1.png
001_shift_2.png       003_shift_3.png       005_underexposed.png  008_overexposed.png   010_regular.png       012_shift_1.png       014_shift_2.png
001_shift_3.png       003_underexposed.png  006_overexposed.png   008_regular.png       010_shift_1.png       012_shift_2.png       014_shift_3.png
001_underexposed.png  004_overexposed.png   006_regular.png       008_shift_1.png       010_shift_2.png       012_shift_3.png       014_underexposed.png
002_overexposed.png   004_regular.png       006_shift_1.png       008_shift_2.png       010_shift_3.png       012_underexposed.png
""" 

filenames_str_mask = """
000_overexposed_mask.png   003_overexposed_mask.png   006_overexposed_mask.png   009_overexposed_mask.png   012_overexposed_mask.png
000_regular_mask.png       003_regular_mask.png       006_regular_mask.png       009_regular_mask.png       012_regular_mask.png
000_shift_1_mask.png       003_shift_1_mask.png       006_shift_1_mask.png       009_shift_1_mask.png       012_shift_1_mask.png
000_shift_2_mask.png       003_shift_2_mask.png       006_shift_2_mask.png       009_shift_2_mask.png       012_shift_2_mask.png
000_shift_3_mask.png       003_shift_3_mask.png       006_shift_3_mask.png       009_shift_3_mask.png       012_shift_3_mask.png
000_underexposed_mask.png  003_underexposed_mask.png  006_underexposed_mask.png  009_underexposed_mask.png  012_underexposed_mask.png
001_overexposed_mask.png   004_overexposed_mask.png   007_overexposed_mask.png   010_overexposed_mask.png   013_overexposed_mask.png
001_regular_mask.png       004_regular_mask.png       007_regular_mask.png       010_regular_mask.png       013_regular_mask.png
001_shift_1_mask.png       004_shift_1_mask.png       007_shift_1_mask.png       010_shift_1_mask.png       013_shift_1_mask.png
001_shift_2_mask.png       004_shift_2_mask.png       007_shift_2_mask.png       010_shift_2_mask.png       013_shift_2_mask.png
001_shift_3_mask.png       004_shift_3_mask.png       007_shift_3_mask.png       010_shift_3_mask.png       013_shift_3_mask.png
001_underexposed_mask.png  004_underexposed_mask.png  007_underexposed_mask.png  010_underexposed_mask.png  013_underexposed_mask.png
002_overexposed_mask.png   005_overexposed_mask.png   008_overexposed_mask.png   011_overexposed_mask.png   014_overexposed_mask.png
002_regular_mask.png       005_regular_mask.png       008_regular_mask.png       011_regular_mask.png       014_regular_mask.png
002_shift_1_mask.png       005_shift_1_mask.png       008_shift_1_mask.png       011_shift_1_mask.png       014_shift_1_mask.png
002_shift_2_mask.png       005_shift_2_mask.png       008_shift_2_mask.png       011_shift_2_mask.png       014_shift_2_mask.png
002_shift_3_mask.png       005_shift_3_mask.png       008_shift_3_mask.png       011_shift_3_mask.png       014_shift_3_mask.png
002_underexposed_mask.png  005_underexposed_mask.png  008_underexposed_mask.png  011_underexposed_mask.png  014_underexposed_mask.png
"""
# 切割成 list
filenames = filenames_str.split()
filenames_ab = filenames_str_abnormal.split()
filenames__mask = filenames_str_mask.split()

# 輸出檔案名稱
output_path = "test.jsonl"

# 寫入 JSONL 檔
with open(output_path, "w") as f:
    for fname in filenames:
        entry = {
            "filename": f"test/normal/{fname}",
            "label": 0,
            "label_name": "normal",
            "clsname": "normal"
        }
        f.write(json.dumps(entry) + "\n")
    
    for fname in filenames_ab:
        base, _ = os.path.splitext(fname)
        mask_name = f"{base}_mask.png"  # 可改成 .png_mask 看你要哪一種格式
        entry = {
            "filename": f"test/abnormal/{fname}",
            "label": 1,
            "label_name": "abnormal",
            "clsname": "abnormal",
            "mask": f"test/ground_truth/bad/{mask_name}"
        }
       
        f.write(json.dumps(entry) + "\n")

print(f"✅ 已成功輸出 {len(filenames) + len(filenames_ab)} 筆資料到：{output_path}")
