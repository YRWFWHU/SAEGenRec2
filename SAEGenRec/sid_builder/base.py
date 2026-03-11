"""SIDMethod 抽象基类：定义统一的 train + generate 接口。"""

from abc import ABC, abstractmethod


class SIDMethod(ABC):
    """SID 生成方法的统一抽象接口。

    子类必须实现 train() 和 generate()，并设置类属性：
    - name: 方法名（如 'rqvae'）
    - default_k: 默认 SID token 数
    - token_format: token 前缀格式（'auto' 表示按方法自动选择）
    """

    name: str = ""
    default_k: int = 3
    token_format: str = "auto"  # 'auto' | 单字符如 'v' | 'f'

    @abstractmethod
    def train(self, embedding_path: str, output_dir: str, **config) -> str:
        """训练 SID 模型。

        Args:
            embedding_path: 输入 .npy 嵌入文件路径
            output_dir: 模型 checkpoint 输出目录
            **config: 方法特定的训练超参数

        Returns:
            checkpoint_path: 训练好的模型 checkpoint 路径
        """
        ...

    @abstractmethod
    def generate(
        self,
        checkpoint: str,
        embedding_path: str,
        output_path: str,
        k: int = None,
        token_format: str = "auto",
    ) -> str:
        """从训练好的模型生成 .index.json。

        Args:
            checkpoint: 模型 checkpoint 路径
            embedding_path: 输入 .npy 嵌入文件路径
            output_path: 输出 .index.json 路径
            k: SID token 数（None 时使用 default_k）
            token_format: token 前缀格式（'auto' 时按方法默认）

        Returns:
            output_path: 生成的 .index.json 路径
        """
        ...
