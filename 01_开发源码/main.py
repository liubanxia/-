from datetime import datetime


PROJECT_NAME = "Project Phoenix"
VERSION = "V1.0.0"


def main() -> None:
    """Project Phoenix 基础启动入口。"""
    print("=" * 50)
    print(f"{PROJECT_NAME} 启动成功")
    print(f"当前版本：{VERSION}")
    print("运行模式：开发模式")
    print("支持范围：普通 CT 平扫、常规 DR")
    print(f"启动时间：{datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 50)


if __name__ == "__main__":
    main()