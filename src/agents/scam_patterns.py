"""Scam pattern definitions and detection rules for romance scams (杀猪盘)"""

from enum import Enum
from typing import Dict, List, Set
from dataclasses import dataclass


class ScamPattern(str, Enum):
    """Types of scam patterns in romance fraud"""
    LOVE_BOMBING = "love_bombing"  # 过早表达强烈感情
    MONEY_REQUEST = "money_request"  # 提及金钱、转账、投资
    EXTERNAL_LINKS = "external_links"  # 外部网站链接、二维码
    URGENCY_PRESSURE = "urgency_pressure"  # 施压策略
    INCONSISTENCY = "inconsistency"  # 前后矛盾的信息
    TOO_GOOD = "too_good"  # 过于完美的承诺


@dataclass
class PatternRule:
    """A detection rule for a specific scam pattern"""
    pattern: ScamPattern
    keywords: Set[str]
    weight: float  # Contribution to risk score (0.0-1.0)
    description: str
    examples: List[str]


# ============================================================================
# PATTERN RULES LIBRARY
# ============================================================================

# 1. Love Bombing (爱情轰炸) - Early excessive affection
LOVE_BOMBING_KEYWORDS = {
    # Chinese keywords
    "命中注定", "真爱", "唯一", "灵魂伴侣", "完美", "天生一对",
    "离不开", "想你想得", "每分每秒", "余生", "白头偕老",
    "嫁给", "娶你", "结婚", "一辈子", "永远", "最爱",
    "世界上最", "无法自拔", "爱上", "深深爱着",
    
    # English keywords
    "soulmate", "destiny", "meant to be", "perfect match",
    "love of my life", "marry me", "forever", "eternally",
    "obsessed", "can't live without", "true love", "the one",
}

# 2. Money Request (金钱请求) - Financial requests or investment talk
MONEY_REQUEST_KEYWORDS = {
    # Direct money requests
    "转账", "打款", "汇款", "借钱", "借点", "周转", "应急",
    "钱包", "支付宝", "微信支付", "银行卡", "卡号",
    
    # Investment/crypto scams
    "投资", "理财", "收益", "赚钱", "盈利", "回报",
    "比特币", "虚拟币", "数字货币", "加密货币", "炒币",
    "平台", "账户", "充值", "提现", "操作",
    "稳赚", "保本", "高回报", "低风险", "内幕消息",
    
    # Gift/favor scams
    "礼物", "红包", "充话费", "买单", "帮忙付",
    "医药费", "住院", "手术", "急需", "困难",
    
    # English keywords
    "transfer money", "send money", "wire", "investment",
    "crypto", "bitcoin", "trading", "profit", "returns",
    "urgent payment", "hospital bill", "emergency funds",
}

# 3. External Links (外部链接) - Suspicious links/QR codes
EXTERNAL_LINKS_KEYWORDS = {
    "链接", "网址", "点击", "打开", "二维码", "扫码",
    "下载", "安装", "注册", "app", "应用",
    "平台", "网站", "加我", "添加我",
    
    # Common scam platforms
    "telegram", "whatsapp", "line", "qq群",
    
    # URL patterns (checked separately via regex)
    "http://", "https://", "www.", ".com", ".cn",
}

# 4. Urgency Pressure (紧迫施压) - Creating false urgency
URGENCY_PRESSURE_KEYWORDS = {
    "现在", "立刻", "马上", "赶紧", "快点", "抓紧",
    "今天", "必须", "限时", "截止", "错过", "机会",
    "最后", "仅剩", "只有", "不然", "否则",
    "来不及", "晚了", "太迟", "后悔",
    
    # English
    "right now", "immediately", "urgent", "hurry",
    "last chance", "limited time", "expires", "don't miss",
}

# 5. Inconsistency (信息矛盾) - Detection requires conversation history
# These keywords help identify profile inconsistencies
INCONSISTENCY_KEYWORDS = {
    "其实", "实际上", "说实话", "坦白说", "老实说",
    "之前", "忘了", "记错", "搞混", "弄错",
    "改口", "更正", "纠正", "不是",
    
    # Vague identity claims
    "海外", "国外", "出差", "外派", "驻外",
    "军人", "医生", "工程师", "商人", "老板",
    "石油", "钻井", "维和", "联合国",
}

# 6. Too Good to Be True (太完美) - Unrealistic promises
TOO_GOOD_KEYWORDS = {
    "保证", "一定", "肯定", "绝对", "100%",
    "零风险", "无风险", "稳赚不赔", "躺赚", "轻松",
    "月入", "日赚", "翻倍", "暴富", "发财",
    "简单", "容易", "不用", "什么都不用",
    
    # Unrealistic life claims
    "豪车", "豪宅", "私人飞机", "游艇", "别墅",
    "千万", "亿万", "富豪", "继承", "遗产",
    
    # English
    "guaranteed", "100%", "risk-free", "easy money",
    "get rich", "passive income", "no effort",
}


# ============================================================================
# PATTERN RULES CONFIGURATION
# ============================================================================

PATTERN_RULES: Dict[ScamPattern, PatternRule] = {
    ScamPattern.LOVE_BOMBING: PatternRule(
        pattern=ScamPattern.LOVE_BOMBING,
        keywords=LOVE_BOMBING_KEYWORDS,
        weight=0.35,  # High weight - strong indicator
        description="过早或过度的感情表达，快速建立深层情感联系",
        examples=[
            "你是我这辈子遇到的唯一真爱",
            "我们是命中注定的灵魂伴侣",
            "认识你三天，但感觉像认识了一辈子",
        ]
    ),
    
    ScamPattern.MONEY_REQUEST: PatternRule(
        pattern=ScamPattern.MONEY_REQUEST,
        keywords=MONEY_REQUEST_KEYWORDS,
        weight=0.45,  # Highest weight - direct scam indicator
        description="涉及金钱转账、投资理财或财务帮助请求",
        examples=[
            "帮我转2000块应急，明天就还你",
            "这个投资平台收益很高，我带你一起赚",
            "我住院了急需医药费，能帮帮我吗",
        ]
    ),
    
    ScamPattern.EXTERNAL_LINKS: PatternRule(
        pattern=ScamPattern.EXTERNAL_LINKS,
        keywords=EXTERNAL_LINKS_KEYWORDS,
        weight=0.30,  # Medium-high weight
        description="分享外部链接、二维码或要求转移到其他平台",
        examples=[
            "加我Telegram: @scammer123",
            "扫这个二维码注册投资平台",
            "点击这个链接看我的照片",
        ]
    ),
    
    ScamPattern.URGENCY_PRESSURE: PatternRule(
        pattern=ScamPattern.URGENCY_PRESSURE,
        keywords=URGENCY_PRESSURE_KEYWORDS,
        weight=0.25,  # Medium weight - often combined with others
        description="制造紧迫感，催促立即行动",
        examples=[
            "必须今天转账，不然机会就没了",
            "现在马上注册，限时优惠",
            "快点决定，我等不了太久",
        ]
    ),
    
    ScamPattern.INCONSISTENCY: PatternRule(
        pattern=ScamPattern.INCONSISTENCY,
        keywords=INCONSISTENCY_KEYWORDS,
        weight=0.20,  # Lower weight - needs context
        description="个人信息前后矛盾或含糊不清",
        examples=[
            "我在美国当军医（之前说是工程师）",
            "其实我不是30岁，我38了",
            "我在叙利亚维和部队工作（常见诈骗故事）",
        ]
    ),
    
    ScamPattern.TOO_GOOD: PatternRule(
        pattern=ScamPattern.TOO_GOOD,
        keywords=TOO_GOOD_KEYWORDS,
        weight=0.30,  # Medium-high weight
        description="过于完美的承诺或不切实际的描述",
        examples=[
            "这个投资100%稳赚不赔",
            "我有私人飞机，可以带你环游世界",
            "跟我做，月入10万很轻松",
        ]
    ),
}


# ============================================================================
# RISK THRESHOLDS
# ============================================================================

RISK_THRESHOLDS = {
    "low": 0.3,      # 低风险：有可疑迹象
    "medium": 0.5,   # 中风险：多个可疑信号
    "high": 0.7,     # 高风险：明显诈骗特征
    "critical": 0.9, # 极高风险：强烈诈骗信号
}


# ============================================================================
# COMPOSITE PATTERNS (Multi-turn detection)
# ============================================================================

# Dangerous pattern combinations that increase risk score
COMPOSITE_PATTERNS = {
    "fast_intimacy_money": {
        "patterns": [ScamPattern.LOVE_BOMBING, ScamPattern.MONEY_REQUEST],
        "turns_threshold": 5,  # If both appear within 5 turns
        "risk_multiplier": 1.5,  # 50% risk increase
        "description": "快速建立亲密关系后要钱 - 典型杀猪盘套路"
    },
    
    "urgency_money": {
        "patterns": [ScamPattern.URGENCY_PRESSURE, ScamPattern.MONEY_REQUEST],
        "turns_threshold": 3,
        "risk_multiplier": 1.3,
        "description": "紧迫施压 + 要钱 - 紧急诈骗"
    },
    
    "too_good_investment": {
        "patterns": [ScamPattern.TOO_GOOD, ScamPattern.MONEY_REQUEST],
        "turns_threshold": 10,
        "risk_multiplier": 1.4,
        "description": "不实承诺 + 投资邀请 - 投资诈骗"
    },
    
    "external_money": {
        "patterns": [ScamPattern.EXTERNAL_LINKS, ScamPattern.MONEY_REQUEST],
        "turns_threshold": 5,
        "risk_multiplier": 1.4,
        "description": "引导外部平台 + 要钱 - 平台诈骗"
    },
}


# ============================================================================
# WARNING MESSAGES
# ============================================================================

WARNING_MESSAGES = {
    "low": {
        "zh": "⚠️ 检测到轻微可疑信号。请保持警惕，不要透露个人信息。",
        "en": "⚠️ Minor suspicious signals detected. Stay cautious and don't share personal info."
    },
    
    "medium": {
        "zh": "🚨 检测到多个可疑特征。建议谨慎交流，不要涉及金钱往来。",
        "en": "🚨 Multiple suspicious patterns detected. Be very cautious and avoid financial discussions."
    },
    
    "high": {
        "zh": "🛑 高度可疑！这段对话有明显诈骗特征。强烈建议停止交流。",
        "en": "🛑 High risk! Clear scam indicators detected. Strongly recommend ending this conversation."
    },
    
    "critical": {
        "zh": "‼️ 极度危险！这几乎可以确认是诈骗。请立即停止交流并举报此用户。",
        "en": "‼️ CRITICAL DANGER! This is almost certainly a scam. Stop immediately and report this user."
    },
}


def get_pattern_description(pattern: ScamPattern, lang: str = "zh") -> str:
    """Get human-readable description of a scam pattern"""
    descriptions = {
        "zh": {
            ScamPattern.LOVE_BOMBING: "爱情轰炸 - 过早表达强烈感情",
            ScamPattern.MONEY_REQUEST: "金钱请求 - 涉及转账或投资",
            ScamPattern.EXTERNAL_LINKS: "外部链接 - 引导到其他平台",
            ScamPattern.URGENCY_PRESSURE: "紧迫施压 - 制造虚假紧迫感",
            ScamPattern.INCONSISTENCY: "信息矛盾 - 前后不一致",
            ScamPattern.TOO_GOOD: "过于完美 - 不切实际的承诺",
        },
        "en": {
            ScamPattern.LOVE_BOMBING: "Love Bombing - Excessive early affection",
            ScamPattern.MONEY_REQUEST: "Money Request - Financial asks or investment",
            ScamPattern.EXTERNAL_LINKS: "External Links - Moving to other platforms",
            ScamPattern.URGENCY_PRESSURE: "Urgency Pressure - Creating false urgency",
            ScamPattern.INCONSISTENCY: "Inconsistency - Contradictory information",
            ScamPattern.TOO_GOOD: "Too Good to Be True - Unrealistic promises",
        }
    }
    
    return descriptions.get(lang, descriptions["zh"]).get(pattern, str(pattern))
