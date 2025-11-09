"""
BOT SCORER MODULE
Applies a tiered scoring system to classify users as likely human, suspicious, likely bots, or definite bots.
"""

import pandas as pd
import numpy as np


class BotScorer:
    """
    Scoring System:
    - Total points: 185 pts (before verification discount)
    - 0-35: Likely Human
    - 36-60: Suspicious (human review recommended)
    - 61-85: Likely Bot (high confidence)
    - 86+: Definite Bot (very high confidence, flagged for review)

    Note: Thresholds were adjusted and based on analysis of known bot accounts.
    Strong indicators have increased weights (tier system).
    """

    def __init__(self, features_df, troll_farm_members=None):
        self.features = features_df.copy()
        self.scores = None
        self.troll_farm_members = troll_farm_members if troll_farm_members is not None else set()

    def calculate_all_scores(self, is_initial=False):
        # Initialize scoring DataFrame
        self.scores = pd.DataFrame()
        self.scores['username'] = self.features['username']
        self.scores['user_id'] = self.features['user_id']

        if not is_initial:
            print(f"Scoring {len(self.features):,} users...")

        tier1_scores = []
        tier2_scores = []
        tier3_scores = []

        for idx, row in self.features.iterrows():
            # Calculate tier scores
            t1 = self._calculate_tier1_score(row)
            t2 = self._calculate_tier2_score(row)
            t3 = self._calculate_tier3_score(row)

            tier1_scores.append(t1)
            tier2_scores.append(t2)
            tier3_scores.append(t3)

        # Add tier scores to dataframe
        self.scores['tier1_score'] = tier1_scores
        self.scores['tier2_score'] = tier2_scores
        self.scores['tier3_score'] = tier3_scores

        # Calculate total score
        self.scores['total_score_before_verification'] = (
            self.scores['tier1_score'] +
            self.scores['tier2_score'] +
            self.scores['tier3_score']
        )

        # give partial credit for verified accounts
        self.scores['is_verified'] = self.features['is_verified']
        self.scores['has_blue'] = self.features['has_blue']
        self.scores['is_verified_or_blue'] = (
            self.scores['is_verified'] | self.scores['has_blue']
        )

        # Verified accs get 25% point reduction
        self.scores['total_score'] = self.scores.apply(
            lambda row: row['total_score_before_verification'] * 0.75 if row['is_verified_or_blue']
                       else row['total_score_before_verification'],
            axis=1
        )

        # Classify users
        self.scores['classification'] = self.scores['total_score'].apply(self._classify_user)

        # Flag high-risk accounts (definite bot)
        self.scores['flagged_for_review'] = self.scores['total_score'] >= 86
        return self.scores

    def _calculate_tier1_score(self, feature_row):
        """
        Calculate Tier 1 score (15 pts each, max 135).
        Tier 1 is the most important bot indicator, so valued higher.
        """
        score = 0

        # Troll farm membership
        username = feature_row.get('username', '')
        if username in self.troll_farm_members:
            score += 15

        # Follower/following ratio
        ratio = feature_row.get('follower_following_ratio', 999)
        if ratio < 0.1:
            score += 15
        elif ratio <= 0.3:
            score += 10
        elif ratio <= 0.5:
            score += 5

        # Low engagement rate
        engagement = feature_row.get('avg_engagement_per_tweet', 0)
        if engagement < 0.5:
            score += 15
        elif engagement <= 2:
            score += 10
        elif engagement <= 5:
            score += 5

        # Favorites-to-tweets ratio
        fav_ratio = feature_row.get('favorites_tweets_ratio', 0)
        if fav_ratio < 0.1:
            score += 15
        elif fav_ratio <= 0.3:
            score += 10
        elif fav_ratio <= 0.5:
            score += 5

        # Account age
        created_year = feature_row.get('account_created_year', 0)
        same_month = feature_row.get('created_same_month_as_dataset', False)

        if same_month:
            score += 15
        elif created_year == 2024:
            score += 7.5  # Created in 2024 but not Oct
        # No points for 2022-2023 or before 2022

        # Account activity metrics
        lifetime_tweets = feature_row.get('lifetime_tweets_per_day', 0)
        if lifetime_tweets > 100:
            score += 15
        elif lifetime_tweets >= 50:
            score += 10
        elif lifetime_tweets >= 25:
            score += 5

        # Reply to post ratio
        reply_ratio = feature_row.get('reply_ratio_pct', 0)
        if reply_ratio >= 80:
            score += 15
        elif reply_ratio >= 66:
            score += 10
        elif reply_ratio >= 50:
            score += 5

        # Text similarity
        similarity = feature_row.get('avg_text_similarity_pct', 0)
        if similarity > 80:
            score += 15
        elif similarity >= 60:
            score += 10
        elif similarity >= 40:
            score += 5

        # Topic diversity
        concentration = feature_row.get('topic_concentration_pct', 0)
        if concentration == 100:
            score += 15
        elif concentration >= 80:
            score += 10
        elif concentration > 70:
            score += 5

        return score

    def _calculate_tier2_score(self, feature_row):
        """
        Calculate Tier 2 score (10 pts each, max 30).

        Tier 2 includes medium-importance bot indicators.
        """
        score = 0

        # High posting frequency
        tweets_per_day = feature_row.get('posting_freq_tweets_per_day', 0)
        if tweets_per_day > 50:
            score += 10
        elif tweets_per_day >= 30:
            score += 7
        elif tweets_per_day >= 20:
            score += 3

        # Alphanumeric username
        is_bot_username = feature_row.get('bot_username_pattern', False)
        if is_bot_username:
            score += 10

        # Default profile image
        is_default = feature_row.get('has_default_profile_image', False)
        if is_default:
            score += 10

        return score

    def _calculate_tier3_score(self, feature_row):
        """
        Calculate Tier 3 score (5 pts each, max 20).
        Tier 3 includes lower-importance bot indicators.
        """
        score = 0

        # Excessive hashtags
        avg_hashtags = feature_row.get('avg_hashtags_per_tweet', 0)
        has_long = feature_row.get('has_long_hashtag', False)

        if avg_hashtags > 3:
            score += 5
        elif has_long:
            score += 3

        # Retweet-to-original post ratio
        retweet_ratio = feature_row.get('retweet_ratio_pct', 0)
        if retweet_ratio > 80:
            score += 5
        elif retweet_ratio >= 60:
            score += 3

        # porifle completeness
        empty_desc = feature_row.get('has_empty_description', False)
        no_location = feature_row.get('has_no_location', False)

        profile_score = 0
        if empty_desc:
            profile_score += 2
        if no_location:
            profile_score += 2
        score += min(profile_score, 5)

        # Mention spam
        avg_mentions = feature_row.get('avg_mentions_per_tweet', 0)
        if avg_mentions > 3:
            score += 5

        return score


    def _classify_user(self, total_score):
        if total_score <= 35:
            return "Likely Human"
        elif total_score <= 60:
            return "Suspicious"
        elif total_score <= 85:
            return "Likely Bot"
        else:
            return "Definite Bot"

    def get_flagged_accounts(self):
        # Returns high-risk accounts flagged for detailed review (score >= 86
        flagged = self.scores[self.scores['flagged_for_review'] == True].copy()
        flagged = flagged.sort_values('total_score', ascending=False)

        return flagged

    def get_detailed_breakdown(self, username):
        # Get feature row
        feature_row = self.features[self.features['username'] == username]
        if len(feature_row) == 0:
            return {"error": f"Username '{username}' not found"}

        feature_row = feature_row.iloc[0]

        # Get score row
        score_row = self.scores[self.scores['username'] == username].iloc[0]

        # Build breakdown of user's scores
        breakdown = {
            'username': username,
            'total_score': score_row['total_score'],
            'classification': score_row['classification'],
            'flagged': score_row['flagged_for_review'],
            'tier_scores': {
                'tier1': score_row['tier1_score'],
                'tier2': score_row['tier2_score'],
                'tier3': score_row['tier3_score'],
            },
            'verification': {
                'is_verified': score_row['is_verified'],
                'has_blue': score_row['has_blue'],
            },
            'key_features': {
                'posting_freq_tweets_per_day': feature_row['posting_freq_tweets_per_day'],
                'follower_following_ratio': feature_row['follower_following_ratio'],
                'avg_engagement_per_tweet': feature_row['avg_engagement_per_tweet'],
                'account_age_days': feature_row.get('account_age_days', 'N/A'),
                'bot_username_pattern': feature_row['bot_username_pattern'],
                'topic_concentration_pct': feature_row['topic_concentration_pct'],
            }
        }

        return breakdown

    def save_scores(self, output_path="output/user_scores.csv"):
        if self.scores is None:
            raise ValueError("No scores to save")

        if "initial" not in output_path:
            print(f"\nSaving {len(self.scores):,} user scores to {output_path}...")
        self.scores.to_csv(output_path, index=False)

    def save_flagged_accounts(self, output_path="output/flagged_bots.csv"):

        flagged = self.get_flagged_accounts()

        if len(flagged) == 0:
            return

        print(f"\nSaving {len(flagged):,} flagged accounts to {output_path}...\n\n")

        # Merge with features for detailed info
        flagged_detailed = flagged.merge(
            self.features[['username', 'posting_freq_tweets_per_day', 'follower_following_ratio',
                          'avg_engagement_per_tweet', 'account_age_days',
                          'dominant_keyword']],
            on='username',
            how='left'
        )

        flagged_detailed.to_csv(output_path, index=False)

    def generate_summary_report(self):
        if self.scores is None:
            raise ValueError("No scores available")

        # Calculate statistics
        total_users = len(self.scores)
        classification_counts = self.scores['classification'].value_counts()
        flagged_count = self.scores['flagged_for_review'].sum()
        avg_score = self.scores['total_score'].mean()
        median_score = self.scores['total_score'].median()

        # Get top 10 most suspicious accounts
        top_bots = self.scores.nlargest(10, 'total_score')[['username', 'total_score', 'classification']]

        # Tier score statistics
        avg_tier1 = self.scores['tier1_score'].mean()
        avg_tier2 = self.scores['tier2_score'].mean()
        avg_tier3 = self.scores['tier3_score'].mean()

        # Print report to console
        print("\n")
        print("-" * 100)
        print("2024 US ELECTION TWITTER BOT DETECTION ANALYSIS SUMMARY REPORT")
        print("-" * 100)
        print()

        print("DATASET OVERVIEW")
        print("-" * 50)
        print(f"Total Users Analyzed: {total_users:,}")
        print(f"Average Bot Score: {avg_score:.2f}")
        print(f"Median Bot Score: {median_score:.2f}")
        print()

        print("CLASSIFICATION BREAKDOWN")
        print("-" * 50)
        for classification in ["Likely Human", "Suspicious", "Likely Bot", "Definite Bot"]:
            count = classification_counts.get(classification, 0)
            percentage = (count / total_users * 100)
            print(f"{classification:20s}: {count:6,} users ({percentage:5.2f}%)")
        print()

        print("TOP 10 MOST SUSPICIOUS ACCOUNTS")
        print("-" * 50)
        print(f"{'Username':<35s} {'Score':>5s} {'Classification':<20s}")
        print("-" * 50)
        for _, row in top_bots.iterrows():
            print(f"{row['username']:<35s} {row['total_score']:>5.1f} {row['classification']:<20s}")
        print()

        print("METHODOLOGY")
        print("-" * 50)
        print("Scoring System:")
        print("  - 0-35 points: Likely Human")
        print("  - 36-60 points: Suspicious (human review recommended)")
        print("  - 61-85 points: Likely Bot (high confidence)")
        print("  - 86+ points: Definite Bot (very high confidence)")
        print()
        print("Special Rules:")
        print("  - Verified/Blue accounts receive 25% score reduction")
        print("  - Accounts scoring ≥ 86 flagged for detailed manual review")
        print()

