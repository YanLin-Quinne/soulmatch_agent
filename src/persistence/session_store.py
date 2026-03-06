"""
Session 持久化存储 - CRUD 操作
"""
import json
from typing import Optional, List, Dict, Any
from datetime import datetime
from sqlalchemy.orm import Session as DBSession

from .database import (
    get_db_manager,
    Session,
    ConversationTurn,
    MemorySnapshot,
    FeatureHistory,
    RelationshipSnapshot,
    LogicTreeNode,
)


class SessionStore:
    """Session 持久化存储"""

    def __init__(self):
        self.db_manager = get_db_manager()

    def create_session(
        self,
        session_id: str,
        user_id: str,
        bot_id: Optional[str] = None,
        state: Optional[Dict[str, Any]] = None
    ) -> Session:
        """创建新 session"""
        db = self.db_manager.get_session()
        try:
            session = Session(
                session_id=session_id,
                user_id=user_id,
                bot_id=bot_id,
                state=json.dumps(state) if state else None,
                created_at=datetime.utcnow(),
                last_active=datetime.utcnow()
            )
            db.add(session)
            db.commit()
            db.refresh(session)
            return session
        finally:
            db.close()

    def get_session(self, session_id: str) -> Optional[Session]:
        """获取 session"""
        db = self.db_manager.get_session()
        try:
            return db.query(Session).filter(Session.session_id == session_id).first()
        finally:
            db.close()

    def update_session_state(self, session_id: str, state: Dict[str, Any]):
        """更新 session 状态"""
        db = self.db_manager.get_session()
        try:
            session = db.query(Session).filter(Session.session_id == session_id).first()
            if session:
                session.state = json.dumps(state)
                session.last_active = datetime.utcnow()
                db.commit()
        finally:
            db.close()

    def delete_session(self, session_id: str):
        """删除 session（级联删除所有关联数据）"""
        db = self.db_manager.get_session()
        try:
            session = db.query(Session).filter(Session.session_id == session_id).first()
            if session:
                db.delete(session)
                db.commit()
        finally:
            db.close()

    def aggregate_user_sessions(self, user_id: str) -> Dict[str, Any]:
        """聚合某用户的所有 session 数据"""
        db = self.db_manager.get_session()
        try:
            sessions = db.query(Session).filter(Session.user_id == user_id).all()
            if not sessions:
                return {}

            total_turns = sum(
                db.query(ConversationTurn).filter(
                    ConversationTurn.session_id == s.session_id
                ).count() for s in sessions
            )

            return {
                "user_id": user_id,
                "total_sessions": len(sessions),
                "total_turns": total_turns,
                "first_session": min(s.created_at for s in sessions).isoformat(),
                "last_session": max(s.last_active for s in sessions).isoformat(),
            }
        finally:
            db.close()

    def add_conversation_turn(
        self,
        session_id: str,
        turn_number: int,
        speaker: str,
        message: str
    ):
        """添加对话轮次"""
        db = self.db_manager.get_session()
        try:
            turn = ConversationTurn(
                session_id=session_id,
                turn_number=turn_number,
                speaker=speaker,
                message=message,
                timestamp=datetime.utcnow()
            )
            db.add(turn)
            db.commit()
        finally:
            db.close()

    def get_conversation_history(
        self,
        session_id: str,
        limit: Optional[int] = None
    ) -> List[ConversationTurn]:
        """获取对话历史"""
        db = self.db_manager.get_session()
        try:
            query = db.query(ConversationTurn).filter(
                ConversationTurn.session_id == session_id
            ).order_by(ConversationTurn.turn_number)

            if limit:
                query = query.limit(limit)

            return query.all()
        finally:
            db.close()

    def add_memory_snapshot(
        self,
        session_id: str,
        layer: str,
        content: Dict[str, Any],
        turn_range_start: Optional[int] = None,
        turn_range_end: Optional[int] = None,
        variance: Optional[float] = None
    ):
        """添加记忆快照"""
        db = self.db_manager.get_session()
        try:
            snapshot = MemorySnapshot(
                session_id=session_id,
                layer=layer,
                turn_range_start=turn_range_start,
                turn_range_end=turn_range_end,
                content=json.dumps(content),
                variance=variance,
                created_at=datetime.utcnow()
            )
            db.add(snapshot)
            db.commit()
        finally:
            db.close()

    def get_memory_snapshots(
        self,
        session_id: str,
        layer: Optional[str] = None
    ) -> List[MemorySnapshot]:
        """获取记忆快照"""
        db = self.db_manager.get_session()
        try:
            query = db.query(MemorySnapshot).filter(
                MemorySnapshot.session_id == session_id
            )

            if layer:
                query = query.filter(MemorySnapshot.layer == layer)

            return query.order_by(MemorySnapshot.created_at.desc()).all()
        finally:
            db.close()

    def add_feature_history(
        self,
        session_id: str,
        turn_number: int,
        features: Dict[str, Any],
        confidences: Dict[str, float],
        bayesian_updates: Optional[Dict[str, Any]] = None
    ):
        """添加特征历史"""
        db = self.db_manager.get_session()
        try:
            history = FeatureHistory(
                session_id=session_id,
                turn_number=turn_number,
                features=json.dumps(features),
                confidences=json.dumps(confidences),
                bayesian_updates=json.dumps(bayesian_updates) if bayesian_updates else None,
                timestamp=datetime.utcnow()
            )
            db.add(history)
            db.commit()
        finally:
            db.close()

    def get_feature_history(
        self,
        session_id: str,
        limit: Optional[int] = None
    ) -> List[FeatureHistory]:
        """获取特征历史"""
        db = self.db_manager.get_session()
        try:
            query = db.query(FeatureHistory).filter(
                FeatureHistory.session_id == session_id
            ).order_by(FeatureHistory.turn_number.desc())

            if limit:
                query = query.limit(limit)

            return query.all()
        finally:
            db.close()

    def add_relationship_snapshot(
        self,
        session_id: str,
        turn_number: int,
        rel_status: str,
        rel_type: str,
        sentiment: str,
        trust_score: float,
        can_advance: bool,
        social_votes: Optional[Dict[str, Any]] = None
    ):
        """添加关系快照"""
        db = self.db_manager.get_session()
        try:
            snapshot = RelationshipSnapshot(
                session_id=session_id,
                turn_number=turn_number,
                rel_status=rel_status,
                rel_type=rel_type,
                sentiment=sentiment,
                trust_score=trust_score,
                can_advance=can_advance,
                social_votes=json.dumps(social_votes) if social_votes else None,
                timestamp=datetime.utcnow()
            )
            db.add(snapshot)
            db.commit()
        finally:
            db.close()

    def get_relationship_history(
        self,
        session_id: str,
        limit: Optional[int] = None
    ) -> List[RelationshipSnapshot]:
        """获取关系历史"""
        db = self.db_manager.get_session()
        try:
            query = db.query(RelationshipSnapshot).filter(
                RelationshipSnapshot.session_id == session_id
            ).order_by(RelationshipSnapshot.turn_number.desc())

            if limit:
                query = query.limit(limit)

            return query.all()
        finally:
            db.close()

    def add_logic_tree_node(
        self,
        session_id: str,
        turn_number: int,
        node_type: str,
        content: str,
        confidence: float,
        evidence: Optional[List[str]] = None,
        parent_id: Optional[int] = None
    ) -> int:
        """添加逻辑树节点"""
        db = self.db_manager.get_session()
        try:
            node = LogicTreeNode(
                session_id=session_id,
                turn_number=turn_number,
                node_type=node_type,
                parent_id=parent_id,
                content=content,
                confidence=confidence,
                evidence=json.dumps(evidence) if evidence else None,
                created_at=datetime.utcnow()
            )
            db.add(node)
            db.commit()
            db.refresh(node)
            return node.id
        finally:
            db.close()

    def get_logic_tree(
        self,
        session_id: str,
        turn_number: Optional[int] = None
    ) -> List[LogicTreeNode]:
        """获取逻辑树"""
        db = self.db_manager.get_session()
        try:
            query = db.query(LogicTreeNode).filter(
                LogicTreeNode.session_id == session_id
            )

            if turn_number is not None:
                query = query.filter(LogicTreeNode.turn_number == turn_number)

            return query.order_by(LogicTreeNode.created_at).all()
        finally:
            db.close()
