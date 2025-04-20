import os
import yaml
import subprocess
import copy

# 設定主目錄路徑
BASE_DIR = "/mnt/c/Users/anywhere4090/Desktop/ggmfg/Dynamic-noise-AD-master"
CONFIG_FILE = os.path.join(BASE_DIR, "config.yaml")
MAIN_SCRIPT = os.path.join(BASE_DIR, "main.py")

# 使用固定的類別清單（根據你最新提供的 mvtec3d 子資料夾）
categories = [

    "peach",
    "potato",
    "rope",
    "tire",
]

def main():
    # 讀取原始配置文件
    with open(CONFIG_FILE, 'r') as f:
        config = yaml.safe_load(f)
    
    # 循環處理每個類別
    for category in categories:
        print("=" * 50)
        print(f"開始處理類別: {category}")
        print("=" * 50)
        
        # 創建該類別的配置
        category_config = copy.deepcopy(config)
        category_config['data']['category'] = category
        
        # 保存臨時配置文件
        tmp_config_path = f"/tmp/config_{category}.yaml"
        with open(tmp_config_path, 'w') as f:
            yaml.dump(category_config, f, default_flow_style=False)
        
        # 執行命令
        try:
            subprocess.run(["python", MAIN_SCRIPT, "--config", tmp_config_path], check=True)
            print(f"類別 {category} 處理完成!")
        except subprocess.CalledProcessError as e:
            print(f"處理類別 {category} 時出錯: {e}")
        
        print()
    
    print("所有類別處理完成!")

if __name__ == "__main__":
    main()
