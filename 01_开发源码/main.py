from config.settings import *
from utils.logger import setup_logger

logger = setup_logger()

logger.info("Project Phoenix 启动")

print("=" * 50)
print(PROJECT_NAME)
print("版本：", VERSION)
print("开发模式：", DEBUG)
print("支持设备：", SUPPORTED_MODALITIES)
print("=" * 50)