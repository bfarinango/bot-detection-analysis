"""
TROLL FARM DETECTION MODULE
Detects coordinated troll farm activity by identifying groups of accounts that share identical message text.
"""

import pandas as pd
import numpy as np
import re
from collections import Counter, defaultdict
from pathlib import Path


class TrollFarmDetector:
    """
    Detects troll farms by analyzing coordinated messaging patterns.

    A troll farm is defined as:
    - 10+ accounts sharing identical message text (exact match)
    - >50% of accounts classified as "Suspicious", "Likely Bot", or "Definite Bot"
    """

    def __init__(self, data_dir="data", output_dir="output"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.raw_tweets = None
        self.bot_scores = None
        self.troll_farms = None
        self.troll_farm_members = None

    def load_data(self, file_patterns=["october_chunk_*.csv.gz"]):
        all_dataframes = []

        # Find all matching files
        data_files = []
        for pattern in file_patterns:
            data_files.extend(self.data_dir.glob(pattern))

        # Load each file
        for file_path in data_files:
            try:
                df = pd.read_csv(file_path, compression='gzip')
                all_dataframes.append(df)
            except Exception as e:
                print(f"   ERROR loading {file_path.name}: {e}")
                continue

        # Combine all dataframes
        if all_dataframes:
            self.raw_tweets = pd.concat(all_dataframes, ignore_index=True)
            return self.raw_tweets
        else:
            raise ValueError("No data files were successfully loaded!")

    def load_bot_scores(self, scores_path="output/user_scores.csv"):
        """Load user classification scores."""
        self.bot_scores = pd.read_csv(scores_path)
        return self.bot_scores

    def filter_original_tweets(self):
        if self.raw_tweets is None:
            raise ValueError("No tweet data loaded! Call load_data() first.")

        # Filter retweets (where retweetedTweet is not null or False)
        # Column contains False for original tweets and other values for retweets
        self.raw_tweets = self.raw_tweets[
            (self.raw_tweets['retweetedTweet'].isna()) |
            (self.raw_tweets['retweetedTweet'] == False) |
            (self.raw_tweets['retweetedTweet'] == 'False') |
            (self.raw_tweets['retweetedTweet'] == '') |
            (self.raw_tweets['retweetedTweet'] == 'nan') |
            (self.raw_tweets['retweetedTweet'] == '[]')
        ].copy()

        return self.raw_tweets

    def normalize_text(self, text):
        """
        Normalize tweet text for comparison.

        Normalization steps:
        1. Convert to lowercase
        2. Remove URLs (http/https links)
        3. Remove extra whitespace
        4. Strip leading/trailing whitespace

        Args:
            text (str): Raw tweet text

        Returns:
            str: Normalized text
        """
        if pd.isna(text) or text == '':
            return ''

        # Convert to string and lowercase
        text = str(text).lower()

        # Remove URLs (http/https)
        text = re.sub(r'http\S+|https\S+', '', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        # Strip leading/trailing whitespace
        text = text.strip()

        return text

    def identify_troll_farms(self, min_accounts=10, min_bot_percentage=50):
        """
        Identify troll farms by grouping accounts with identical messages.

        A troll farm is identified when:
        - At least min_accounts users share the same normalized message text
        - At least min_bot_percentage% of those users are classified as bots

        Args:
            min_accounts (int): Minimum accounts to constitute a troll farm (10)
            min_bot_percentage (float): Minimum % of bot accounts (50)

        Returns:
            pandas.DataFrame: Troll farm members with statistics
        """
        print("-" * 100)
        print("IDENTIFYING TROLL FARMS")
        print("-" * 100)

        if self.raw_tweets is None or self.bot_scores is None:
            raise ValueError("Data not loaded")

        print("Normalizing tweet text...")

        # Add normalized text column
        self.raw_tweets['normalized_text'] = self.raw_tweets['text'].apply(self.normalize_text)

        # Remove empty normalized texts
        self.raw_tweets = self.raw_tweets[self.raw_tweets['normalized_text'] != ''].copy()

        print(f"   - Normalized {len(self.raw_tweets):,} tweets\n")

        # Group by normalized text to find shared messages
        # For each message, track which users posted it and how often
        message_groups = defaultdict(lambda: defaultdict(int))

        print("Analyzing tweets for troll farm patterns...")
        for _, row in self.raw_tweets.iterrows():
            normalized_msg = row['normalized_text']
            username = row['username']
            message_groups[normalized_msg][username] += 1


        print("Analyzing message groups for troll farm patterns...")

        # Find groups with enough accounts
        potential_farms = []

        for message, users in message_groups.items():
            # Count users sharing message
            unique_users = list(users.keys())
            num_users = len(unique_users)

            if num_users >= min_accounts:
                # Merge with bot scores to check classifications
                users_df = pd.DataFrame({
                    'username': unique_users,
                    'message_frequency': [users[u] for u in unique_users]
                })

                users_df = users_df.merge(
                    self.bot_scores[['username', 'classification', 'total_score']],
                    on='username',
                    how='left'
                )

                # Count bot classifications
                bot_labels = ['Suspicious', 'Likely Bot', 'Definite Bot']
                bot_count = users_df['classification'].isin(bot_labels).sum()
                bot_percentage = (bot_count / num_users) * 100

                # Check if meets bot percentage threshold
                if bot_percentage >= min_bot_percentage:
                    potential_farms.append({
                        'shared_message': message,  # Use normalized text
                        'farm_size': num_users,
                        'farm_bot_count': bot_count,
                        'farm_bot_percentage': bot_percentage,
                        'users_data': users_df
                    })

        print(f"   - Identified {len(potential_farms)} troll farms")

        if len(potential_farms) == 0:
            self.troll_farms = []
            self.troll_farm_members = pd.DataFrame()
            return self.troll_farm_members

        # Assign troll farm IDs and compile members
        all_members = []

        for farm_id, farm in enumerate(potential_farms, start=1):
            farm_name = f"troll_farm_{farm_id}"

            # Add farm info to each user
            users_df = farm['users_data'].copy()
            users_df['troll_farm_id'] = farm_name
            users_df['shared_message'] = farm['shared_message']
            users_df['farm_size'] = farm['farm_size']
            users_df['farm_bot_percentage'] = farm['farm_bot_percentage']

            # Rename columns
            users_df = users_df.rename(columns={
                'classification': 'bot_label',
                'message_frequency': 'message_frequency'
            })

            all_members.append(users_df)

        # Combine all troll farm members
        self.troll_farm_members = pd.concat(all_members, ignore_index=True)

        # Sort by farm ID and bot score
        self.troll_farm_members = self.troll_farm_members.sort_values(
            ['troll_farm_id', 'total_score'],
            ascending=[True, False]
        )

        return self.troll_farm_members

    def save_results(self, output_path="output/troll_farm_members.csv"):
        # Save troll farm member list to CSV
        if self.troll_farm_members is None or len(self.troll_farm_members) == 0:
            return

        print(f"\nSaving troll farm members to {output_path}…\n")

        # Select and order columns for output
        output_df = self.troll_farm_members[[
            'username',
            'troll_farm_id',
            'shared_message',
            'bot_label',
            'farm_size',
            'farm_bot_percentage',
            'message_frequency',
            'total_score'
        ]].copy()

        # Round percentages for readability
        output_df['farm_bot_percentage'] = output_df['farm_bot_percentage'].round(1)

        output_df.to_csv(output_path, index=False)

    def get_troll_farm_usernames(self):
        # Get list of usernames identified as troll farm members
        if self.troll_farm_members is None or len(self.troll_farm_members) == 0:
            return set()

        return set(self.troll_farm_members['username'].unique())

    def generate_summary_report(self):
        if self.troll_farm_members is None or len(self.troll_farm_members) == 0:
            return

        print()
        print("-" * 100)
        print("TROLL FARM DETECTION REPORT")
        print("-" * 100)
        print()

        # Calculate statistics
        num_farms = self.troll_farm_members['troll_farm_id'].nunique()
        total_members = len(self.troll_farm_members)
        avg_farm_size = self.troll_farm_members.groupby('troll_farm_id')['farm_size'].first().mean()
        avg_bot_pct = self.troll_farm_members.groupby('troll_farm_id')['farm_bot_percentage'].first().mean()

        print("OVERVIEW")
        print("-" * 50)
        print(f"Total troll Farms Identified: {num_farms}")
        print(f"Total Accounts in Troll Farms: {total_members:,}")
        print(f"Average Farm Size: {avg_farm_size:.1f} accounts")
        print(f"Average Predicted Bot Percentage: {avg_bot_pct:.1f}%")
        print()

        print("DETECTION CRITERIA")
        print("-" * 50)
        print("- Minimum 10 accounts sharing identical normalized message")
        print("- At least 50% of accounts classified as bots")
        print("- Bot classifications: Suspicious, Likely Bot, Definite Bot")
        print("- Text normalization: lowercase, URL removal, whitespace trimming")
        print("- Original tweets only (retweets excluded)")
        print()

        print("TROLL FARM DETAILS")
        print("-" * 70)

        for farm_id in sorted(self.troll_farm_members['troll_farm_id'].unique()):
            farm_data = self.troll_farm_members[self.troll_farm_members['troll_farm_id'] == farm_id]
            size = farm_data['farm_size'].iloc[0]
            bot_pct = farm_data['farm_bot_percentage'].iloc[0]
            message = farm_data['shared_message'].iloc[0]

            # Classification breakdown
            class_counts = farm_data['bot_label'].value_counts()

            print(f"\n{farm_id.upper()}:")
            print(f"  Size: {size} accounts ({bot_pct:.1f}% were suspected bots)")
            print("  Classification Breakdown:")
            for label, count in class_counts.items():
                print(f"    - {label}: {count} accounts")
            print("  Shared Message:")
            truncated_message = message[:200] + ('...' if len(message) > 200 else '')
            print(f'    "{truncated_message}"')
            print("  Top 5 Most Active Accounts:")
            top_users = farm_data.nlargest(5, 'message_frequency')[['username', 'message_frequency', 'total_score', 'bot_label']]
            for _, user in top_users.iterrows():
                print(f"    - {user['username']}: {int(user['message_frequency'])} posts, score {user['total_score']:.1f}, {user['bot_label']}")




