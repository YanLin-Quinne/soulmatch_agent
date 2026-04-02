"""
数据库连接管理 - 支持 SQLite 和 PostgreSQL
"""
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, Text, DateTime, ForeignKey, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import os
from typing import Optional

Base = declarative_base()


class Session(Base):
    """Session 表 - 持久化 session 状态"""
    __tablename__ = 'sessions'

    session_id = Column(String(255), primary_key=True)
    user_id = Column(String(255), nullable=False, index=True)
    bot_id = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)
    last_active = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, index=True)
    state = Column(Text)  # JSON serialized AgentContext

    # Relationships
    turns = relationship("ConversationTurn", back_populates="session", cascade="all, delete-orphan")
    memories = relationship("MemorySnapshot", back_populates="session", cascade="all, delete-orphan")
    features = relationship("FeatureHistory", back_populates="session", cascade="all, delete-orphan")
    relationships = relationship("RelationshipSnapshot", back_populates="session", cascade="all, delete-orphan")
    logic_trees = relationship("LogicTreeNode", back_populates="session", cascade="all, delete-orphan")


class ConversationTurn(Base):
    """对话轮次表"""
    __tablename__ = 'conversation_turns'

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(255), ForeignKey('sessions.session_id'), nullable=False)
    turn_number = Column(Integer, nullable=False)
    speaker = Column(String(50), nullable=False)  # 'user' or 'bot'
    message = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

    session = relationship("Session", back_populates="turns")


class MemorySnapshot(Base):
    """三层记忆快照表"""
    __tablename__ = 'memory_snapshots'

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(255), ForeignKey('sessions.session_id'), nullable=False)
    layer = Column(String(50), nullable=False)  # 'working', 'episodic', 'semantic'
    turn_range_start = Column(Integer)
    turn_range_end = Column(Integer)
    content = Column(Text, nullable=False)  # JSON serialized memory item
    embedding = Column(LargeBinary)  # Optional: for semantic search
    variance = Column(Float)  # 新增：方差指标
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("Session", back_populates="memories")


class FeatureHistory(Base):
    """特征演化历史表 - 追踪 Bayesian 更新"""
    __tablename__ = 'feature_history'

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(255), ForeignKey('sessions.session_id'), nullable=False)
    turn_number = Column(Integer, nullable=False)
    features = Column(Text, nullable=False)  # JSON: {feature_name: value}
    confidences = Column(Text, nullable=False)  # JSON: {feature_name: confidence}
    bayesian_updates = Column(Text)  # JSON: prior → posterior trace
    timestamp = Column(DateTime, default=datetime.utcnow)

    session = relationship("Session", back_populates="features")


class RelationshipSnapshot(Base):
    """关系预测快照表"""
    __tablename__ = 'relationship_snapshots'

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(255), ForeignKey('sessions.session_id'), nullable=False)
    turn_number = Column(Integer, nullable=False)
    rel_status = Column(String(100))
    rel_type = Column(String(100))
    sentiment = Column(String(100))
    trust_score = Column(Float)
    can_advance = Column(Boolean)
    social_votes = Column(Text)  # JSON: agent votes
    timestamp = Column(DateTime, default=datetime.utcnow)

    session = relationship("Session", back_populates="relationships")


class LogicTreeNode(Base):
    """逻辑树节点表 - 三段论推理结构"""
    __tablename__ = 'logic_tree_nodes'

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(255), ForeignKey('sessions.session_id'), nullable=False)
    turn_number = Column(Integer, nullable=False)
    node_type = Column(String(50), nullable=False)  # 'major_premise', 'minor_premise', 'conclusion'
    parent_id = Column(Integer, ForeignKey('logic_tree_nodes.id'))
    content = Column(Text, nullable=False)
    confidence = Column(Float, nullable=False)
    evidence = Column(Text)  # JSON: supporting evidence
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("Session", back_populates="logic_trees")
    children = relationship("LogicTreeNode", backref="parent", remote_side=[id])


class DatabaseManager:
    """数据库管理器 - 支持 SQLite 和 PostgreSQL"""

    def __init__(self, database_url: Optional[str] = None):
        """
        初始化数据库连接

        Args:
            database_url: 数据库 URL
                - SQLite: sqlite:///./ai_you.db
                - PostgreSQL: postgresql://user:pass@localhost/ai_you
        """
        if database_url is None:
            # 默认使用 SQLite
            db_path = os.getenv("DATABASE_PATH", "./ai_you.db")
            database_url = f"sqlite:///{db_path}"

        self.engine = create_engine(
            database_url,
            echo=os.getenv("SQL_ECHO", "false").lower() == "true",
            pool_pre_ping=True  # 自动重连
        )
        self.SessionLocal = sessionmaker(bind=self.engine)

    def create_tables(self):
        """创建所有表"""
        Base.metadata.create_all(self.engine)

    def drop_tables(self):
        """删除所有表（谨慎使用）"""
        Base.metadata.drop_all(self.engine)

    def get_session(self):
        """获取数据库 session"""
        return self.SessionLocal()


# 全局数据库管理器实例
_db_manager: Optional[DatabaseManager] = None


def get_db_manager() -> DatabaseManager:
    """获取全局数据库管理器"""
    global _db_manager
    if _db_manager is None:
        database_url = os.getenv("DATABASE_URL")
        _db_manager = DatabaseManager(database_url)
        _db_manager.create_tables()
    return _db_manager


def init_database(database_url: Optional[str] = None):
    """初始化数据库"""
    global _db_manager
    _db_manager = DatabaseManager(database_url)
    _db_manager.create_tables()
    return _db_manager
