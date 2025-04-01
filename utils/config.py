import toml

def load_config(config_file):
    """用于加载 toml 配置文件，并添加设备信息。

    :param config_file: toml 配置文件的路径
    :return: 包含配置信息的字典
    """
    try:
        with open(config_file, "r") as f:
            # 使用 toml 加载配置
            config = toml.load(f)
        return config
    except FileNotFoundError:
        print(f"配置文件 {config_file} 未找到，请检查文件路径。")
    except Exception as e:
        print(f"解析 toml 文件时出错: {e}")
