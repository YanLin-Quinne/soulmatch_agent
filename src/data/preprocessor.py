"""Data preprocessing for OkCupid profiles"""

import pandas as pd
from pathlib import Path
from typing import Optional
from loguru import logger

from src.data.schema import OkCupidProfile


class OkCupidPreprocessor:
    """Preprocess OkCupid CSV data"""
    
    def __init__(self, raw_data_path: Path):
        self.raw_data_path = raw_data_path
        self.df: Optional[pd.DataFrame] = None
        
    def load_data(self) -> pd.DataFrame:
        """Load raw CSV data"""
        logger.info(f"Loading data from {self.raw_data_path}")
        self.df = pd.read_csv(self.raw_data_path, low_memory=False)
        logger.info(f"Loaded {len(self.df)} profiles")
        logger.info(f"Columns: {list(self.df.columns)}")
        return self.df
    
    def clean_data(self) -> pd.DataFrame:
        """Clean and normalize data"""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        logger.info("Starting data cleaning...")
        
        # Remove duplicates
        original_len = len(self.df)
        self.df = self.df.drop_duplicates()
        logger.info(f"Removed {original_len - len(self.df)} duplicates")
        
        # Handle missing values in essays
        essay_cols = [f'essay{i}' for i in range(10)]
        for col in essay_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna("")
        
        # Clean income (convert to int, handle -1 as None)
        if 'income' in self.df.columns:
            self.df['income'] = pd.to_numeric(self.df['income'], errors='coerce')
            self.df.loc[self.df['income'] == -1, 'income'] = None
        
        # Clean age (remove outliers)
        if 'age' in self.df.columns:
            self.df = self.df[(self.df['age'] >= 18) & (self.df['age'] <= 80)]
        
        # Normalize text fields
        text_columns = ['status', 'sex', 'orientation', 'body_type', 'diet', 
                       'drinks', 'drugs', 'smokes', 'education', 'job', 
                       'religion', 'ethnicity', 'sign', 'location']
        
        for col in text_columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].str.lower().str.strip()
        
        logger.info(f"Cleaned data: {len(self.df)} profiles remaining")
        return self.df
    
    def filter_quality_profiles(self, min_essay_length: int = 100) -> pd.DataFrame:
        """Filter profiles with sufficient essay content"""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Combine all essays
        essay_cols = [f'essay{i}' for i in range(10)]
        self.df['combined_essays'] = self.df[essay_cols].fillna('').agg(' '.join, axis=1)
        
        # Filter by total essay length
        self.df['essay_length'] = self.df['combined_essays'].str.len()
        original_len = len(self.df)
        self.df = self.df[self.df['essay_length'] >= min_essay_length]
        
        logger.info(f"Filtered to {len(self.df)} profiles with >={min_essay_length} chars")
        logger.info(f"Removed {original_len - len(self.df)} profiles with insufficient content")
        
        return self.df
    
    def to_profile_models(self) -> list[OkCupidProfile]:
        """Convert DataFrame to OkCupidProfile models"""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        profiles = []
        for idx, row in self.df.iterrows():
            try:
                profile_dict = {}
                
                # Basic fields
                for field in OkCupidProfile.model_fields.keys():
                    if field in row.index:
                        value = row[field]
                        # Convert NaN to None
                        if pd.isna(value):
                            profile_dict[field] = None
                        else:
                            profile_dict[field] = value
                
                profile = OkCupidProfile(**profile_dict)
                profiles.append(profile)
            except Exception as e:
                logger.warning(f"Failed to parse profile at index {idx}: {e}")
                continue
        
        logger.info(f"Converted {len(profiles)} profiles to OkCupidProfile models")
        return profiles
    
    def save_processed(self, output_path: Path):
        """Save processed data"""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_parquet(output_path, index=False)
        logger.info(f"Saved processed data to {output_path}")
    
    def get_statistics(self) -> dict:
        """Get data statistics"""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        stats = {
            'total_profiles': len(self.df),
            'age_distribution': self.df['age'].describe().to_dict() if 'age' in self.df.columns else {},
            'sex_distribution': self.df['sex'].value_counts().to_dict() if 'sex' in self.df.columns else {},
            'orientation_distribution': self.df['orientation'].value_counts().to_dict() if 'orientation' in self.df.columns else {},
            'avg_essay_length': self.df['essay_length'].mean() if 'essay_length' in self.df.columns else 0,
        }
        
        return stats


def preprocess_pipeline(
    raw_data_path: Path,
    output_path: Path,
    min_essay_length: int = 100
) -> tuple[pd.DataFrame, dict]:
    """Complete preprocessing pipeline"""
    
    preprocessor = OkCupidPreprocessor(raw_data_path)
    
    # Load and clean
    preprocessor.load_data()
    preprocessor.clean_data()
    preprocessor.filter_quality_profiles(min_essay_length=min_essay_length)
    
    # Save processed data
    preprocessor.save_processed(output_path)
    
    # Get statistics
    stats = preprocessor.get_statistics()
    
    logger.info("Preprocessing complete!")
    logger.info(f"Statistics: {stats}")
    
    return preprocessor.df, stats
