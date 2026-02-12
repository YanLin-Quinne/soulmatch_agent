"""Main script for data preprocessing pipeline"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger
from src.data.preprocessor import preprocess_pipeline
from src.data.persona_builder import PersonaBuilder
from src.config import settings


def main():
    """Run complete data preprocessing pipeline"""
    
    logger.info("=" * 60)
    logger.info("SoulMatch Data Preprocessing Pipeline")
    logger.info("=" * 60)
    
    # Paths
    raw_data_path = settings.data_dir / "raw" / "okcupid_profiles.csv"
    processed_data_path = settings.data_dir / "processed" / "okcupid_processed.parquet"
    personas_output_path = settings.data_dir / "processed" / "bot_personas.json"
    
    # Check if raw data exists
    if not raw_data_path.exists():
        logger.error(f"Raw data not found at {raw_data_path}")
        logger.info("Please run: python scripts/download_okcupid_data.py")
        return
    
    # Step 1: Preprocess data
    logger.info("\nStep 1: Preprocessing raw data...")
    df, stats = preprocess_pipeline(
        raw_data_path=raw_data_path,
        output_path=processed_data_path,
        min_essay_length=100
    )
    
    logger.info(f"\nPreprocessing Statistics:")
    logger.info(f"Total profiles: {stats['total_profiles']}")
    logger.info(f"Average essay length: {stats['avg_essay_length']:.1f}")
    
    # Step 2: Build personas (sample 8 bots)
    logger.info("\nStep 2: Building bot personas...")
    logger.info("Note: This will use LLM API (Claude/GPT) and may take time")
    
    # Load processed profiles
    from src.data.preprocessor import OkCupidPreprocessor
    preprocessor = OkCupidPreprocessor(raw_data_path)
    preprocessor.df = df
    profiles = preprocessor.to_profile_models()
    
    # Build personas
    builder = PersonaBuilder()
    
    # Check API key
    if not settings.anthropic_api_key and not settings.openai_api_key:
        logger.warning("No LLM API key found. Skipping persona building.")
        logger.info("Set ANTHROPIC_API_KEY or OPENAI_API_KEY in .env to build personas")
        return
    
    use_claude = bool(settings.anthropic_api_key)
    
    bot_personas = builder.sample_bot_personas(
        profiles=profiles,
        num_bots=8,
        use_claude=use_claude
    )
    
    # Save personas
    builder.save_personas(bot_personas, personas_output_path)
    
    logger.info("\n" + "=" * 60)
    logger.info("Preprocessing Complete!")
    logger.info(f"Processed data: {processed_data_path}")
    logger.info(f"Bot personas: {personas_output_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
