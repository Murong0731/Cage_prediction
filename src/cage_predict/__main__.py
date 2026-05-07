"""允许通过 `python -m cage_predict` 方式启动命令行工具。

这是 Python 包的标准入口点，当用户执行 python -m cage_predict 时，
Python 解释器会自动执行本文件中的 main() 函数。
"""

from .cli import main

main()
