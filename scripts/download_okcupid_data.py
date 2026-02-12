"""Download OkCupid profiles dataset from Kaggle

This script downloads the OkCupid profiles dataset using the Kaggle API.
Dataset contains ~59,946 user profiles with 22 dimensions and essay texts.

Requirements:
- Kaggle API credentials configured in .env (KAGGLE_USERNAME, KAGGLE_KEY)
- kaggle package installed (pip install kaggle)

Usage:
    python scripts/download_okcupid_data.py
"""

import os
import sys
import zipfile
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import settings


def setup_kaggle_credentials() -> bool:
    """Setup Kaggle API credentials from environment
    
    Returns:
        bool: True if credentials are configured, False otherwise
    """
    if not settings.kaggle_username or not settings.kaggle_key:
        print("❌ Error: Kaggle credentials not configured")
        print("\nPlease set the following in your .env file:")
        print("  KAGGLE_USERNAME=your_username")
        print("  KAGGLE_KEY=your_api_key")
        print("\nTo get your Kaggle API credentials:")
        print("  1. Go to https://www.kaggle.com/settings")
        print("  2. Scroll to 'API' section")
        print("  3. Click 'Create New Token'")
        print("  4. Add credentials to .env file")
        return False
    
    # Set environment variables for kaggle package
    os.environ["KAGGLE_USERNAME"] = settings.kaggle_username
    os.environ["KAGGLE_KEY"] = settings.kaggle_key
    return True


def download_dataset(dataset_name: str, output_dir: Path) -> Optional[Path]:
    """Download dataset from Kaggle
    
    Args:
        dataset_name: Kaggle dataset identifier (e.g., "andrewmvd/okcupid-profiles")
        output_dir: Directory to save the downloaded data
        
    Returns:
        Path to downloaded file if successful, None otherwise
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        print(f"🔍 Initializing Kaggle API...")
        api = KaggleApi()
        api.authenticate()
        
        print(f"📥 Downloading dataset: {dataset_name}")
        print(f"📂 Output directory: {output_dir}")
        
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Download dataset
        api.dataset_download_files(
            dataset_name,
            path=str(output_dir),
            unzip=False,
            quiet=False
        )
        
        # Find the downloaded zip file
        zip_files = list(output_dir.glob("*.zip"))
        if not zip_files:
            print("❌ Error: No zip file found after download")
            return None
            
        zip_path = zip_files[0]
        print(f"\n✅ Download complete: {zip_path}")
        return zip_path
        
    except ImportError:
        print("❌ Error: kaggle package not installed")
        print("Install with: pip install kaggle")
        return None
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        return None


def extract_csv_files(zip_path: Path, output_dir: Path) -> list[Path]:
    """Extract CSV files from zip archive
    
    Args:
        zip_path: Path to zip file
        output_dir: Directory to extract files to
        
    Returns:
        List of extracted CSV file paths
    """
    csv_files = []
    
    try:
        print(f"\n📦 Extracting files from {zip_path.name}...")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get list of CSV files in archive
            csv_members = [m for m in zip_ref.namelist() if m.endswith('.csv')]
            
            if not csv_members:
                print("⚠️  Warning: No CSV files found in archive")
                # Extract all files anyway
                zip_ref.extractall(output_dir)
            else:
                # Extract CSV files
                for member in csv_members:
                    print(f"  📄 Extracting: {member}")
                    zip_ref.extract(member, output_dir)
                    extracted_path = output_dir / member
                    csv_files.append(extracted_path)
        
        # Clean up zip file
        print(f"🧹 Removing zip file...")
        zip_path.unlink()
        
        return csv_files
        
    except zipfile.BadZipFile:
        print(f"❌ Error: {zip_path} is not a valid zip file")
        return []
    except Exception as e:
        print(f"❌ Error extracting files: {e}")
        return []


def verify_dataset(csv_path: Path) -> bool:
    """Verify the downloaded dataset
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        bool: True if dataset is valid, False otherwise
    """
    try:
        import pandas as pd
        
        print(f"\n🔍 Verifying dataset: {csv_path.name}")
        
        # Read first few rows
        df = pd.read_csv(csv_path, nrows=5)
        
        print(f"  ✅ Rows (sample): {len(df)}")
        print(f"  ✅ Columns: {len(df.columns)}")
        print(f"  📋 Column names: {', '.join(df.columns[:5])}...")
        
        # Get full row count
        df_full = pd.read_csv(csv_path)
        print(f"  ✅ Total rows: {len(df_full):,}")
        
        return True
        
    except ImportError:
        print("⚠️  Warning: pandas not installed, skipping verification")
        print("Install with: pip install pandas")
        return True
    except Exception as e:
        print(f"❌ Error verifying dataset: {e}")
        return False


def main():
    """Main execution function"""
    print("=" * 60)
    print("📊 OkCupid Dataset Downloader")
    print("=" * 60)
    
    # Check Kaggle credentials
    if not setup_kaggle_credentials():
        sys.exit(1)
    
    # Dataset configuration
    dataset_name = "andrewmvd/okcupid-profiles"
    output_dir = settings.data_dir / "raw"
    expected_csv = output_dir / "okcupid_profiles.csv"
    
    # Check if dataset already exists
    if expected_csv.exists():
        print(f"\n⚠️  Dataset already exists: {expected_csv}")
        response = input("Do you want to re-download? (y/N): ").strip().lower()
        if response != 'y':
            print("✅ Using existing dataset")
            verify_dataset(expected_csv)
            return
        print("🗑️  Removing existing dataset...")
        expected_csv.unlink()
    
    # Download dataset
    zip_path = download_dataset(dataset_name, output_dir)
    if not zip_path:
        sys.exit(1)
    
    # Extract files
    csv_files = extract_csv_files(zip_path, output_dir)
    
    if not csv_files:
        print("\n❌ No CSV files extracted")
        sys.exit(1)
    
    # Rename to standard name if needed
    if len(csv_files) == 1 and csv_files[0].name != "okcupid_profiles.csv":
        print(f"\n📝 Renaming {csv_files[0].name} -> okcupid_profiles.csv")
        csv_files[0].rename(expected_csv)
        csv_files[0] = expected_csv
    
    # Verify dataset
    for csv_file in csv_files:
        if not verify_dataset(csv_file):
            sys.exit(1)
    
    print("\n" + "=" * 60)
    print("✅ Dataset download complete!")
    print("=" * 60)
    print(f"📂 Location: {output_dir / 'okcupid_profiles.csv'}")
    print(f"📊 Ready for preprocessing")


if __name__ == "__main__":
    main()
