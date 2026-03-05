"""
数据库初始化脚本

运行此脚本以创建所有必要的数据库表
"""
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.persistence.database import init_database
from src.config import settings
from loguru import logger


def main():
    """初始化数据库"""
    logger.info("开始初始化数据库...")
    logger.info(f"数据库 URL: {settings.database_url}")

    try:
        db_manager = init_database(settings.database_url)
        logger.success("数据库初始化成功！")
        logger.info("已创建以下表：")
        logger.info("  - sessions (Session 状态)")
        logger.info("  - conversation_turns (对话历史)")
        logger.info("  - memory_snapshots (三层记忆)")
        logger.info("  - feature_history (特征演化)")
        logger.info("  - relationship_snapshots (关系预测)")
        logger.info("  - logic_tree_nodes (逻辑树)")

    except Exception as e:
        logger.error(f"数据库初始化失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
