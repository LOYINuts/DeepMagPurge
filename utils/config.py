import yaml


def load_config(config_file):
    """用于加载 YAML 配置文件，并添加设备信息。

    :param config_file: YAML 配置文件的路径
    :return: 包含配置信息的字典
    """
    try:
        with open(config_file, "r") as f:
            # 使用 yaml.safe_load 方法加载 YAML 文件内容
            config = yaml.safe_load(f)
        # 检查是否有可用的 CUDA 设备，若有则使用 CUDA，否则使用 CPU
        return config
    except FileNotFoundError:
        print(f"配置文件 {config_file} 未找到，请检查文件路径。")
    except yaml.YAMLError as e:
        print(f"解析 YAML 文件时出错: {e}")
