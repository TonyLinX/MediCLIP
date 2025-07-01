import json

# 原始檔案名（可貼更完整版本）
filenames_str = """
000_regular.png  039_regular.png  078_regular.png  117_regular.png  156_regular.png  195_regular.png  234_regular.png  273_regular.png  312_regular.png  351_regular.png
001_regular.png  040_regular.png  079_regular.png  118_regular.png  157_regular.png  196_regular.png  235_regular.png  274_regular.png  313_regular.png  352_regular.png
002_regular.png  041_regular.png  080_regular.png  119_regular.png  158_regular.png  197_regular.png  236_regular.png  275_regular.png  314_regular.png  353_regular.png
003_regular.png  042_regular.png  081_regular.png  120_regular.png  159_regular.png  198_regular.png  237_regular.png  276_regular.png  315_regular.png  354_regular.png
004_regular.png  043_regular.png  082_regular.png  121_regular.png  160_regular.png  199_regular.png  238_regular.png  277_regular.png  316_regular.png  355_regular.png
005_regular.png  044_regular.png  083_regular.png  122_regular.png  161_regular.png  200_regular.png  239_regular.png  278_regular.png  317_regular.png  356_regular.png
006_regular.png  045_regular.png  084_regular.png  123_regular.png  162_regular.png  201_regular.png  240_regular.png  279_regular.png  318_regular.png  357_regular.png
007_regular.png  046_regular.png  085_regular.png  124_regular.png  163_regular.png  202_regular.png  241_regular.png  280_regular.png  319_regular.png  358_regular.png
008_regular.png  047_regular.png  086_regular.png  125_regular.png  164_regular.png  203_regular.png  242_regular.png  281_regular.png  320_regular.png  359_regular.png
009_regular.png  048_regular.png  087_regular.png  126_regular.png  165_regular.png  204_regular.png  243_regular.png  282_regular.png  321_regular.png  360_regular.png
010_regular.png  049_regular.png  088_regular.png  127_regular.png  166_regular.png  205_regular.png  244_regular.png  283_regular.png  322_regular.png  361_regular.png
011_regular.png  050_regular.png  089_regular.png  128_regular.png  167_regular.png  206_regular.png  245_regular.png  284_regular.png  323_regular.png  362_regular.png
012_regular.png  051_regular.png  090_regular.png  129_regular.png  168_regular.png  207_regular.png  246_regular.png  285_regular.png  324_regular.png  363_regular.png
013_regular.png  052_regular.png  091_regular.png  130_regular.png  169_regular.png  208_regular.png  247_regular.png  286_regular.png  325_regular.png  364_regular.png
014_regular.png  053_regular.png  092_regular.png  131_regular.png  170_regular.png  209_regular.png  248_regular.png  287_regular.png  326_regular.png  365_regular.png
015_regular.png  054_regular.png  093_regular.png  132_regular.png  171_regular.png  210_regular.png  249_regular.png  288_regular.png  327_regular.png  366_regular.png
016_regular.png  055_regular.png  094_regular.png  133_regular.png  172_regular.png  211_regular.png  250_regular.png  289_regular.png  328_regular.png  367_regular.png
017_regular.png  056_regular.png  095_regular.png  134_regular.png  173_regular.png  212_regular.png  251_regular.png  290_regular.png  329_regular.png  368_regular.png
018_regular.png  057_regular.png  096_regular.png  135_regular.png  174_regular.png  213_regular.png  252_regular.png  291_regular.png  330_regular.png  369_regular.png
019_regular.png  058_regular.png  097_regular.png  136_regular.png  175_regular.png  214_regular.png  253_regular.png  292_regular.png  331_regular.png  370_regular.png
020_regular.png  059_regular.png  098_regular.png  137_regular.png  176_regular.png  215_regular.png  254_regular.png  293_regular.png  332_regular.png  371_regular.png
021_regular.png  060_regular.png  099_regular.png  138_regular.png  177_regular.png  216_regular.png  255_regular.png  294_regular.png  333_regular.png  372_regular.png
022_regular.png  061_regular.png  100_regular.png  139_regular.png  178_regular.png  217_regular.png  256_regular.png  295_regular.png  334_regular.png  373_regular.png
023_regular.png  062_regular.png  101_regular.png  140_regular.png  179_regular.png  218_regular.png  257_regular.png  296_regular.png  335_regular.png  374_regular.png
024_regular.png  063_regular.png  102_regular.png  141_regular.png  180_regular.png  219_regular.png  258_regular.png  297_regular.png  336_regular.png  375_regular.png
025_regular.png  064_regular.png  103_regular.png  142_regular.png  181_regular.png  220_regular.png  259_regular.png  298_regular.png  337_regular.png  376_regular.png
026_regular.png  065_regular.png  104_regular.png  143_regular.png  182_regular.png  221_regular.png  260_regular.png  299_regular.png  338_regular.png  377_regular.png
027_regular.png  066_regular.png  105_regular.png  144_regular.png  183_regular.png  222_regular.png  261_regular.png  300_regular.png  339_regular.png  378_regular.png
028_regular.png  067_regular.png  106_regular.png  145_regular.png  184_regular.png  223_regular.png  262_regular.png  301_regular.png  340_regular.png  379_regular.png
029_regular.png  068_regular.png  107_regular.png  146_regular.png  185_regular.png  224_regular.png  263_regular.png  302_regular.png  341_regular.png  380_regular.png
030_regular.png  069_regular.png  108_regular.png  147_regular.png  186_regular.png  225_regular.png  264_regular.png  303_regular.png  342_regular.png  381_regular.png
031_regular.png  070_regular.png  109_regular.png  148_regular.png  187_regular.png  226_regular.png  265_regular.png  304_regular.png  343_regular.png  382_regular.png
032_regular.png  071_regular.png  110_regular.png  149_regular.png  188_regular.png  227_regular.png  266_regular.png  305_regular.png  344_regular.png  383_regular.png
033_regular.png  072_regular.png  111_regular.png  150_regular.png  189_regular.png  228_regular.png  267_regular.png  306_regular.png  345_regular.png  384_regular.png
034_regular.png  073_regular.png  112_regular.png  151_regular.png  190_regular.png  229_regular.png  268_regular.png  307_regular.png  346_regular.png  385_regular.png
035_regular.png  074_regular.png  113_regular.png  152_regular.png  191_regular.png  230_regular.png  269_regular.png  308_regular.png  347_regular.png  386_regular.png
036_regular.png  075_regular.png  114_regular.png  153_regular.png  192_regular.png  231_regular.png  270_regular.png  309_regular.png  348_regular.png
037_regular.png  076_regular.png  115_regular.png  154_regular.png  193_regular.png  232_regular.png  271_regular.png  310_regular.png  349_regular.png
038_regular.png  077_regular.png  116_regular.png  155_regular.png  194_regular.png  233_regular.png  272_regular.png  311_regular.png  350_regular.png
"""  # ← 請把完整檔名貼進來

# 切割成 list
filenames = filenames_str.split()

# 輸出檔案名稱
output_path = "train.jsonl"

# 寫入 JSONL 檔
with open(output_path, "w") as f:
    for fname in filenames:
        entry = {
            "filename": f"train/good/{fname}",
            "label": 0,
            "label_name": "normal",
            "clsname": "normal"
        }
        f.write(json.dumps(entry) + "\n")

print(f"✅ 已成功輸出 {len(filenames)} 筆資料到：{output_path}")
