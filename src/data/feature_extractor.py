"""Feature extraction from profile essays using LLM"""

import json
from typing import Optional
from loguru import logger

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from src.data.schema import OkCupidProfile, ExtractedFeatures
from src.config import settings


class FeatureExtractor:
    """Extract personality features from profile essays using LLM"""
    
    def __init__(self, use_claude: bool = True):
        self.use_claude = use_claude and ANTHROPIC_AVAILABLE
        
        if self.use_claude:
            if not settings.anthropic_api_key:
                raise ValueError("ANTHROPIC_API_KEY not set in environment")
            self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
            logger.info("Using Claude for feature extraction")
        else:
            if not OPENAI_AVAILABLE:
                raise ImportError("openai package not installed")
            if not settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY not set in environment")
            self.client = openai.OpenAI(api_key=settings.openai_api_key)
            logger.info("Using GPT for feature extraction")
    
    def _build_prompt(self, profile: OkCupidProfile) -> str:
        """Build extraction prompt from profile"""
        
        # Combine all essays
        essays = []
        for i in range(10):
            essay = getattr(profile, f'essay{i}', None)
            if essay and essay.strip():
                essays.append(f"Essay {i}: {essay}")
        
        combined_essays = "\n\n".join(essays)
        
        prompt = f"""Analyze this dating profile and extract personality features in JSON format.

Profile Information:
- Age: {profile.age}
- Sex: {profile.sex}
- Orientation: {profile.orientation}
- Education: {profile.education}
- Job: {profile.job}

Profile Essays:
{combined_essays}

Extract the following features (respond ONLY with valid JSON):
{{
  "communication_style": "<direct/indirect/humorous/serious/casual/formal>",
  "communication_confidence": <0.0-1.0>,
  "core_values": ["<value1>", "<value2>", "<value3>"],
  "values_confidence": <0.0-1.0>,
  "interest_categories": {{
    "music": <0.0-1.0>,
    "sports": <0.0-1.0>,
    "travel": <0.0-1.0>,
    "food": <0.0-1.0>,
    "arts": <0.0-1.0>,
    "tech": <0.0-1.0>,
    "outdoors": <0.0-1.0>,
    "books": <0.0-1.0>
  }},
  "openness": <0.0-1.0>,
  "conscientiousness": <0.0-1.0>,
  "extraversion": <0.0-1.0>,
  "agreeableness": <0.0-1.0>,
  "neuroticism": <0.0-1.0>,
  "relationship_goals": "<casual/serious/friendship/unsure>",
  "goals_confidence": <0.0-1.0>,
  "personality_summary": "<brief 2-3 sentence summary>"
}}

Guidelines:
- communication_style: How they express themselves
- core_values: 3-5 most important life values (e.g., family, career, adventure, honesty)
- interest_categories: Strength of interest in each category (0=none, 1=very strong)
- Big Five traits: Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism (0-1 scale)
- relationship_goals: What they're looking for
- All confidence scores: How confident you are in the assessment (0-1)
- personality_summary: Concise overview of their personality

Respond with ONLY the JSON object, no explanation."""

        return prompt
    
    def extract_features(self, profile: OkCupidProfile) -> Optional[ExtractedFeatures]:
        """Extract features from a profile"""
        
        try:
            prompt = self._build_prompt(profile)
            
            if self.use_claude:
                response = self.client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1024,
                    temperature=0.3,
                    messages=[{
                        "role": "user",
                        "content": prompt
                    }]
                )
                content = response.content[0].text
            else:
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{
                        "role": "user",
                        "content": prompt
                    }],
                    temperature=0.3,
                    max_tokens=1024
                )
                content = response.choices[0].message.content
            
            # Parse JSON response
            # Remove markdown code blocks if present
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            features_dict = json.loads(content)
            features = ExtractedFeatures(**features_dict)
            
            logger.debug(f"Extracted features: {features.personality_summary}")
            return features
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Response content: {content}")
            return None
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return None
    
    def extract_batch(
        self, 
        profiles: list[OkCupidProfile],
        max_profiles: Optional[int] = None
    ) -> dict[int, ExtractedFeatures]:
        """Extract features from multiple profiles"""
        
        if max_profiles:
            profiles = profiles[:max_profiles]
        
        results = {}
        total = len(profiles)
        
        for idx, profile in enumerate(profiles):
            logger.info(f"Extracting features for profile {idx+1}/{total}")
            
            features = self.extract_features(profile)
            if features:
                results[idx] = features
            
            # Rate limiting
            if (idx + 1) % 10 == 0:
                logger.info(f"Processed {idx+1}/{total} profiles")
        
        logger.info(f"Successfully extracted features for {len(results)}/{total} profiles")
        return results
