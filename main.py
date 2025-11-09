import sys
from pathlib import Path
import time
from datetime import datetime

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Import modules
from data_loader import DataLoader
from feature_extractor import FeatureExtractor
from troll_farm_detector import TrollFarmDetector
from bot_scorer import BotScorer
from visualizer import BotVisualizer


def print_banner():
    print()
    print("=" * 100)
    print("TWITTER BOT DETECTION ON 2024 US ELECTION TWEETS")
    print("=" * 100)
    print("\n")


def print_section_header(title, width=100):
    print("\n" + "-" * width)
    print(title)
    print("-" * width)


def check_environment():
    print_section_header("ENVIRONMENT CHECK")

    # Check for data directory
    data_dir = Path("data")
    if not data_dir.exists():
        print("ERROR: 'data/' directory not found!")
        return False

    # Check for data files
    data_files = list(data_dir.glob("october_chunk_*.csv.gz"))
    if len(data_files) == 0:
        print("ERROR: No data files found in 'data/' directory!")
        return False

    print(f"Found {len(data_files)} data file(s):")
    for f in data_files:
        print(f"   - {f.name}")

    # Check for output directory
    output_dir = Path("output")
    if not output_dir.exists():
        print("\n'output/' directory not found.")
        output_dir.mkdir(exist_ok=True)
        print(f"Created output directory: {output_dir}")

    # Check for src directory and modules
    src_dir = Path("src")
    if not src_dir.exists():
        print("\nERROR: 'src/' directory not found!")
        return False

    required_modules = [
        'data_loader.py',
        'feature_extractor.py',
        'bot_scorer.py',
        'visualizer.py'
    ]

    missing_modules = []
    for module in required_modules:
        if not (src_dir / module).exists():
            missing_modules.append(module)

    if missing_modules:
        print("\nERROR: Missing required modules:")
        for module in missing_modules:
            print(f"   - {module}")
        return False

    print("\nEnvironment check complete.\n")
    return True


def main():
    """
    1. Environment check
    2. Data loading
    3. Feature extraction
    4. Bot scoring
    5. Visualization
    6. Summary report
    """

    # Print banner
    print_banner()

    # Record start time
    start_time = time.time()

    if not check_environment():
        print("\nEnvironment check failed")
        sys.exit(1)

    try:
        print_section_header("LOADING DATA")
        loader = DataLoader(data_dir="data")
        raw_tweets = loader.load_csv_files()

        if raw_tweets is None or len(raw_tweets) == 0:
            print("\nERROR: No data was loaded.")
            sys.exit(1)

        user_data = loader.aggregate_by_user()
        loader.save_user_data("output/user_aggregated_data.csv")

        print_section_header("EXTRACTING ACCOUNT FEATURES")
        extractor = FeatureExtractor(user_data)
        features = extractor.extract_all_features()
        extractor.save_features("output/user_features.csv")

        print_section_header("CALCULATING FIRST ROUND OF SCORES")
        initial_scorer = BotScorer(features)
        initial_scores = initial_scorer.calculate_all_scores(is_initial=True)
        initial_scorer.save_scores("output/user_scores_initial.csv")

        # Troll farm detection
        detector = TrollFarmDetector(data_dir="data", output_dir="output")
        detector.load_data()
        detector.load_bot_scores("output/user_scores_initial.csv")
        detector.filter_original_tweets()
        detector.identify_troll_farms(min_accounts=10, min_bot_percentage=50)
        detector.save_results("output/troll_farm_members.csv")
        troll_farm_usernames = detector.get_troll_farm_usernames()

        print_section_header("RECALCULATING USER SCORES")
        scorer = BotScorer(features, troll_farm_members=troll_farm_usernames)
        scores = scorer.calculate_all_scores(is_initial=False)
        scorer.save_scores("output/user_scores.csv")
        scorer.save_flagged_accounts("output/flagged_bots.csv")

        visualizer = BotVisualizer(scores, features, output_dir="output")
        visualizer.create_all_visualizations()

        scorer.generate_summary_report()

        detector.generate_summary_report()


    except Exception as e:
        print("\n\nERROR: An unexpected error occurred!")

        # Print full error for debugging
        import traceback
        print("\n   Full error trace:")
        print("   " + "-" * 60)
        traceback.print_exc()
        sys.exit(1)
