"""Build persona profiles from OkCupid data and extracted features"""

import json
import random
from pathlib import Path
from typing import Optional
from loguru import logger
import numpy as np

from src.data.schema import OkCupidProfile, ExtractedFeatures, PersonaProfile
from src.data.preprocessor import OkCupidPreprocessor
from src.data.feature_extractor import FeatureExtractor


class PersonaBuilder:
    """Build complete persona profiles for agents"""
    
    def __init__(self):
        self.feature_extractor: Optional[FeatureExtractor] = None
    
    def initialize_extractor(self, use_claude: bool = True):
        """Initialize feature extractor (lazy loading)"""
        if self.feature_extractor is None:
            self.feature_extractor = FeatureExtractor(use_claude=use_claude)
    
    def build_system_prompt(
        self, 
        profile: OkCupidProfile, 
        features: ExtractedFeatures
    ) -> str:
        """Generate system prompt for role-playing this persona"""
        
        # Build demographic info
        demographics = []
        if profile.age:
            demographics.append(f"age {profile.age}")
        if profile.sex:
            demographics.append(f"{profile.sex}")
        if profile.location:
            demographics.append(f"from {profile.location}")
        
        demographic_str = ", ".join(demographics) if demographics else "person"
        
        # Build personality description
        personality_traits = []
        
        if features.extraversion:
            if features.extraversion > 0.6:
                personality_traits.append("outgoing and social")
            elif features.extraversion < 0.4:
                personality_traits.append("introverted and reflective")
        
        if features.openness and features.openness > 0.6:
            personality_traits.append("curious and open-minded")
        
        if features.agreeableness and features.agreeableness > 0.6:
            personality_traits.append("friendly and cooperative")
        
        personality_str = ", ".join(personality_traits) if personality_traits else "balanced"
        
        # Build interests
        top_interests = sorted(
            features.interest_categories.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:3]
        interests_str = ", ".join([interest for interest, _ in top_interests])
        
        # Build values
        values_str = ", ".join(features.core_values[:3]) if features.core_values else "authenticity"
        
        # Build communication style
        comm_style = features.communication_style
        
        prompt = f"""You are role-playing as a {demographic_str} on a dating platform.

PERSONALITY:
{features.personality_summary}

TRAITS:
- You are {personality_str}
- Communication style: {comm_style}
- Core values: {values_str}

INTERESTS:
- Primary interests: {interests_str}
- Your essays reveal passion for: {', '.join(features.core_values) if features.core_values else 'various topics'}

RELATIONSHIP GOALS:
- Looking for: {features.relationship_goals}

BACKGROUND:
- Education: {profile.education or 'not specified'}
- Occupation: {profile.job or 'not specified'}
- Lifestyle: drinks {profile.drinks or 'socially'}, {profile.diet or 'varied diet'}

ROLE-PLAYING GUIDELINES:
1. Stay in character - respond as this person would
2. Draw from the background and interests described
3. Match the communication style ({comm_style})
4. Be authentic and conversational
5. Show your personality through your responses
6. Ask questions that reflect your values and interests
7. Don't reveal you're an AI or break character

Remember: You are having a genuine conversation on a dating app. Be yourself (this persona), be engaging, and show interest in getting to know the other person."""

        return prompt
    
    def create_feature_vector(self, features: ExtractedFeatures) -> list[float]:
        """Create normalized feature vector for matching"""
        
        vector = []
        
        # Big Five traits (5 dimensions)
        vector.extend([
            features.openness or 0.5,
            features.conscientiousness or 0.5,
            features.extraversion or 0.5,
            features.agreeableness or 0.5,
            features.neuroticism or 0.5,
        ])
        
        # Communication style (one-hot encoding)
        comm_styles = ["direct", "indirect", "humorous", "serious", "casual", "formal"]
        comm_vector = [1.0 if features.communication_style == style else 0.0 
                      for style in comm_styles]
        vector.extend(comm_vector)
        
        # Interest categories (8 dimensions)
        interest_order = ["music", "sports", "travel", "food", "arts", "tech", "outdoors", "books"]
        for interest in interest_order:
            vector.append(features.interest_categories.get(interest, 0.0))
        
        # Relationship goals (one-hot encoding)
        goal_types = ["casual", "serious", "friendship", "unsure"]
        goal_vector = [1.0 if features.relationship_goals == goal else 0.0 
                      for goal in goal_types]
        vector.extend(goal_vector)
        
        # Normalize to unit vector
        vector_array = np.array(vector)
        norm = np.linalg.norm(vector_array)
        if norm > 0:
            vector_array = vector_array / norm
        
        return vector_array.tolist()
    
    def build_persona(
        self, 
        profile: OkCupidProfile, 
        features: ExtractedFeatures,
        profile_id: str,
        is_bot: bool = True
    ) -> PersonaProfile:
        """Build complete persona profile"""
        
        system_prompt = self.build_system_prompt(profile, features)
        feature_vector = self.create_feature_vector(features)
        
        persona = PersonaProfile(
            original_profile=profile,
            features=features,
            profile_id=profile_id,
            is_bot=is_bot,
            system_prompt=system_prompt,
            feature_vector=feature_vector
        )
        
        return persona
    
    def sample_bot_personas(
        self,
        profiles: list[OkCupidProfile],
        num_bots: int = 8,
        use_claude: bool = True
    ) -> list[PersonaProfile]:
        """Sample and build bot personas from profile pool"""
        
        logger.info(f"Sampling {num_bots} bot personas from {len(profiles)} profiles")
        
        # Initialize extractor
        self.initialize_extractor(use_claude=use_claude)
        
        # Stratified sample: ensure gender/orientation diversity
        male = [p for p in profiles if p.sex == 'm']
        female = [p for p in profiles if p.sex == 'f']
        
        # 4 male + 4 female for 8 bots, mix orientations
        n_per_gender = num_bots // 2
        sampled_profiles = (
            random.sample(male, min(n_per_gender, len(male))) +
            random.sample(female, min(n_per_gender, len(female)))
        )
        random.shuffle(sampled_profiles)
        
        personas = []
        for idx, profile in enumerate(sampled_profiles):
            logger.info(f"Building bot persona {idx+1}/{num_bots}")
            
            # Extract features
            features = self.feature_extractor.extract_features(profile)
            if not features:
                logger.warning(f"Failed to extract features for profile {idx}, skipping")
                continue
            
            # Build persona
            persona = self.build_persona(
                profile=profile,
                features=features,
                profile_id=f"bot_{idx}",
                is_bot=True
            )
            
            personas.append(persona)
        
        logger.info(f"Built {len(personas)} bot personas")
        return personas
    
    def save_personas(self, personas: list[PersonaProfile], output_path: Path):
        """Save personas to JSON file"""
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dict
        personas_dict = [persona.model_dump(mode='json') for persona in personas]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(personas_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(personas)} personas to {output_path}")
    
    def load_personas(self, input_path: Path) -> list[PersonaProfile]:
        """Load personas from JSON file"""
        
        with open(input_path, 'r', encoding='utf-8') as f:
            personas_dict = json.load(f)
        
        personas = [PersonaProfile(**p) for p in personas_dict]
        
        logger.info(f"Loaded {len(personas)} personas from {input_path}")
        return personas
