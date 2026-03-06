#!/usr/bin/env python3
"""用 Claude Opus 4.6 推断旧金山 profiles 的心理特征"""

import json
import os
from pathlib import Path
from anthropic import Anthropic

INPUT_PATH = Path(__file__).parent.parent / "data/raw/sf_profiles_selected.json"
OUTPUT_PATH = Path(__file__).parent.parent / "data/processed/bot_personas_sf.json"

INFERENCE_PROMPT = """分析以下 OkCupid profile，推断心理特征。

Profile 信息：
- 年龄：{age}，性别：{sex}，职业：{job}，地点：{location}
- Essays（自述文本）：
{essays}

请推断以下特征（JSON 格式）：
{{
  "big_five": {{
    "openness": 0.0-1.0,
    "conscientiousness": 0.0-1.0,
    "extraversion": 0.0-1.0,
    "agreeableness": 0.0-1.0,
    "neuroticism": 0.0-1.0
  }},
  "mbti": "XXXX",
  "enneagram": "XwY",
  "communication_style": "direct/indirect/casual",
  "core_values": ["value1", "value2", "value3"],
  "interest_categories": {{
    "music": 0.0-1.0,
    "sports": 0.0-1.0,
    "travel": 0.0-1.0,
    "food": 0.0-1.0,
    "arts": 0.0-1.0,
    "tech": 0.0-1.0,
    "books": 0.0-1.0
  }},
  "relationship_goals": "serious/casual/unsure",
  "personality_summary": "简短描述（1-2 句话）"
}}

只返回 JSON，不要其他文字。"""

def truncate_essays(essays: dict, max_words: int = 1000) -> str:
    """截断 essays 到最多 max_words 词"""
    text = ""
    for key, value in essays.items():
        if value:
            text += f"\n{key}: {value}\n"
    words = text.split()
    if len(words) > max_words:
        return " ".join(words[:max_words]) + "..."
    return text

def infer_features(client: Anthropic, profile: dict) -> dict:
    """用 Claude 推断心理特征"""
    essays_text = truncate_essays(profile["essays"], max_words=800)

    prompt = INFERENCE_PROMPT.format(
        age=profile["age"],
        sex=profile["sex"],
        job=profile.get("job", "unknown"),
        location=profile.get("location", "unknown"),
        essays=essays_text
    )

    response = client.messages.create(
        model="claude-opus-4-20250514",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )

    result = response.content[0].text.strip()
    # 提取 JSON（可能被 markdown 包裹）
    if "```json" in result:
        result = result.split("```json")[1].split("```")[0].strip()
    elif "```" in result:
        result = result.split("```")[1].split("```")[0].strip()

    return json.loads(result)

def generate_system_prompt(profile: dict, features: dict) -> str:
    """生成 system_prompt"""
    sex_str = "male" if profile["sex"] == "m" else "female"
    age = profile["age"]
    job = profile.get("job", "unknown")
    location = profile.get("location", "unknown")

    # 从 essay0 提取简短自我介绍（前 200 词）
    essay0 = profile["essays"].get("essay0", "")
    bio_words = essay0.split()[:200]
    bio = " ".join(bio_words) if bio_words else "No bio available."

    prompt = f"""You are a {age}-year-old {sex_str} from {location}. Your job: {job}.

Personality: {features['mbti']}, {features['enneagram']}. {features['personality_summary']}

Communication style: {features['communication_style']}.

Bio excerpt: {bio}

Keep responses natural, concise, and human. Reflect your personality in conversation."""

    return prompt

def main():
    # 直接从 .env 读取
    env_path = Path(__file__).parent.parent / ".env"
    api_key = None
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                if line.startswith("ANTHROPIC_API_KEY="):
                    api_key = line.split("=", 1)[1].strip()
                    break

    if not api_key:
        print("错误：未找到 ANTHROPIC_API_KEY")
        return

    client = Anthropic(
        api_key=api_key,
        base_url="https://api.anthropic.com"
    )

    print(f"读取: {INPUT_PATH}")
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        profiles = json.load(f)

    print(f"处理 {len(profiles)} 个 profiles...")

    results = []
    for i, profile in enumerate(profiles):
        print(f"\n[{i+1}/{len(profiles)}] 处理 {profile['profile_id']}...")

        try:
            features = infer_features(client, profile)
            system_prompt = generate_system_prompt(profile, features)

            persona = {
                "profile_id": profile["profile_id"],
                "is_bot": True,
                "system_prompt": system_prompt,
                "original_profile": {
                    "age": profile["age"],
                    "sex": profile["sex"],
                    "orientation": profile.get("orientation"),
                    "status": profile.get("status"),
                    "job": profile.get("job"),
                    "location": profile.get("location"),
                    "education": profile.get("education"),
                    "ethnicity": profile.get("ethnicity")
                },
                "features": {
                    "communication_style": features["communication_style"],
                    "communication_confidence": 0.85,
                    "core_values": features["core_values"],
                    "values_confidence": 0.85,
                    "interest_categories": features["interest_categories"],
                    "openness": features["big_five"]["openness"],
                    "conscientiousness": features["big_five"]["conscientiousness"],
                    "extraversion": features["big_five"]["extraversion"],
                    "agreeableness": features["big_five"]["agreeableness"],
                    "neuroticism": features["big_five"]["neuroticism"],
                    "mbti": features["mbti"],
                    "enneagram": features["enneagram"],
                    "relationship_goals": features["relationship_goals"],
                    "goals_confidence": 0.85,
                    "personality_summary": features["personality_summary"]
                }
            }

            results.append(persona)
            print(f"  ✓ MBTI: {features['mbti']}, Enneagram: {features['enneagram']}")

        except Exception as e:
            print(f"  ✗ 错误: {e}")
            continue

    # 保存
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✓ 保存到: {OUTPUT_PATH}")
    print(f"成功处理: {len(results)}/{len(profiles)}")

if __name__ == "__main__":
    main()
