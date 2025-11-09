"""
FEATURE EXTRACTOR MODULE
Calculates all 16 bot detection indicators for each user.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import warnings

warnings.filterwarnings('ignore')


class FeatureExtractor:
    """
    Extracts bot detection features from user-level data.
    Calculates 16 bot indicators for each user in the dataset.
    """

    def __init__(self, user_data):
        self.user_data = user_data.copy()
        self.features = None

    def extract_all_features(self):
        # Initialize a list to store features for each user
        all_features = []

        print("Extracting features...")
        for idx, user_row in self.user_data.iterrows():
            # Calculate all features for this user
            user_features = self._extract_user_features(user_row)
            all_features.append(user_features)

        # Convert list of feature dicts to a DataFrame
        self.features = pd.DataFrame(all_features)
        return self.features

    def _extract_user_features(self, user_row):
        # Basic user information
        features = {
            'username': user_row['username'],
            'user_id': user_row['user_id'],
            'followersCount': user_row.get('followersCount', 0),
            'friendsCount': user_row.get('friendsCount', 0),
        }

        # Calculate each feature category
        features.update(self._tier1_features(user_row))
        features.update(self._tier2_features(user_row))
        features.update(self._tier3_features(user_row))

        return features

    def _tier1_features(self, user_row):
        features = {}

        # Follower/Following Ratio
        features.update(self._calc_follower_ratio(user_row))

        # Low Engagement Rate
        features.update(self._calc_engagement_rate(user_row))

        # Favorites-to-Tweets Ratio
        features.update(self._calc_favorites_ratio(user_row))

        # Account Age
        features.update(self._calc_account_age(user_row))

        # Account Activity Metrics
        features.update(self._calc_account_activity(user_row))

        return features

    def _calc_posting_frequency(self, user_row):
        epochs = user_row['epoch']
        if not epochs or len(epochs) == 0:
            return {'posting_freq_tweets_per_day': 0}

        # Calculate time span in days
        timestamps = [e for e in epochs if pd.notna(e)]
        if len(timestamps) < 2:
            # If only 1 tweet, assume 1 day
            tweets_per_day = len(timestamps)
        else:
            min_time = min(timestamps)
            max_time = max(timestamps)
            days = (max_time - min_time) / 86400
            days = max(days, 1)
            tweets_per_day = len(timestamps) / days

        return {'posting_freq_tweets_per_day': round(tweets_per_day, 2)}

    def _calc_follower_ratio(self, user_row):
        followers = user_row.get('followersCount', 0)
        following = user_row.get('friendsCount', 0)

        if following == 0:
            # Set high ratio to avoid / 0
            ratio = 999
        else:
            ratio = followers / following

        return {'follower_following_ratio': round(ratio, 3)}

    def _calc_engagement_rate(self, user_row):
        replies = user_row['replyCount']
        retweets = user_row['retweetCount']
        likes = user_row['likeCount']

        # Calculate total engagement per tweet
        total_engagement = []
        for r, rt, l in zip(replies, retweets, likes):
            # Handle None/NaN values
            r = r if pd.notna(r) else 0
            rt = rt if pd.notna(rt) else 0
            l = l if pd.notna(l) else 0
            total_engagement.append(r + rt + l)

        avg_engagement = np.mean(total_engagement) if total_engagement else 0

        return {'avg_engagement_per_tweet': round(avg_engagement, 2)}

    def _calc_favorites_ratio(self, user_row):
        favorites = user_row.get('favouritesCount', 0)
        statuses = user_row.get('statusesCount', 0)

        if statuses == 0:
            ratio = 0
        else:
            ratio = favorites / statuses

        return {'favorites_tweets_ratio': round(ratio, 3)}

    def _calc_account_age(self, user_row):
        created = user_row.get('created', None)
        epochs = user_row['epoch']

        # Get first tweet date
        first_tweet_epoch = min([e for e in epochs if pd.notna(e)]) if epochs else None

        if created is None or pd.isna(created) or not first_tweet_epoch:
            return {
                'account_created_date': None,
                'account_age_days': None,
                'first_tweet_date': None,
            }

        try:
            # Handle datetime objects and string formats
            if isinstance(created, datetime):
                account_created = created
            elif isinstance(created, (pd.Timestamp, pd.DatetimeIndex)):
                account_created = created.to_pydatetime()
            else:
                # Parse as string (format: "Fri Oct 31 12:00:00 +0000 2020")
                account_created = datetime.strptime(str(created), "%a %b %d %H:%M:%S %z %Y")

            # Timezone aware
            if account_created.tzinfo is None:
                account_created = pytz.UTC.localize(account_created)

            first_tweet_date = datetime.fromtimestamp(first_tweet_epoch, tz=pytz.UTC)

            # Calculate age in days
            age_days = (first_tweet_date - account_created).days

            # Check if created in same month as first tweet
            same_month = (account_created.year == first_tweet_date.year and
                          account_created.month == first_tweet_date.month)

            return {
                'account_created_date': account_created.strftime("%Y-%m-%d"),
                'account_created_year': account_created.year,
                'account_age_days': age_days,
                'first_tweet_date': first_tweet_date.strftime("%Y-%m-%d"),
                'created_same_month_as_dataset': same_month,
            }
        except:
            return {
                'account_created_date': None,
                'account_age_days': None,
                'first_tweet_date': None,
            }

    def _calc_account_activity(self, user_row):
        statuses = user_row.get('statusesCount', 0)
        created = user_row.get('created', None)

        if created is None or pd.isna(created) or statuses == 0:
            return {'lifetime_tweets_per_day': 0}

        try:
            # Handle both datetime objects and string formats
            if isinstance(created, datetime):
                account_created = created
            elif isinstance(created, (pd.Timestamp, pd.DatetimeIndex)):
                account_created = created.to_pydatetime()
            else:
                # Parse as string
                account_created = datetime.strptime(str(created), "%a %b %d %H:%M:%S %z %Y")

            # Timezone aware
            if account_created.tzinfo is None:
                account_created = pytz.UTC.localize(account_created)

            now = datetime.now(pytz.UTC)
            days_since_creation = (now - account_created).days
            days_since_creation = max(days_since_creation, 1)  # At least 1 day

            tweets_per_day = statuses / days_since_creation

            return {'lifetime_tweets_per_day': round(tweets_per_day, 2)}
        except Exception as e:
            return {'lifetime_tweets_per_day': 0}


    def _tier2_features(self, user_row):
        features = {}

        # High Posting Frequency
        features.update(self._calc_posting_frequency(user_row))

        # Reply-to-Post Ratio
        features.update(self._calc_reply_ratio(user_row))

        # Text Similarity
        features.update(self._calc_text_similarity(user_row))

        # Alphanumeric Username
        features.update(self._calc_username_pattern(user_row))

        # Default Profile Image
        features.update(self._calc_default_profile(user_row))

        # Topic Diversity
        features.update(self._calc_topic_diversity(user_row))

        return features

    def _calc_reply_ratio(self, user_row):
        reply_ids = user_row['in_reply_to_status_id_str']
        if not reply_ids:
            return {'reply_ratio_pct': 0}

        reply_count = sum(1 for r in reply_ids if pd.notna(r))
        total = len(reply_ids)

        reply_pct = (reply_count / total * 100) if total > 0 else 0

        return {'reply_ratio_pct': round(reply_pct, 2)}

    def _calc_text_similarity(self, user_row):
        texts = user_row['text']

        if not texts or len(texts) < 2:
            return {'avg_text_similarity_pct': 0}

        # Filter out empty/null texts
        texts = [str(t) for t in texts if pd.notna(t) and len(str(t)) > 0]

        if len(texts) < 2:
            return {'avg_text_similarity_pct': 0}

        try:
            # Use TF-IDF to calculate text similarity
            # Limit to first 50 tweets
            sample_texts = texts[:50]

            vectorizer = TfidfVectorizer(max_features=100)
            tfidf_matrix = vectorizer.fit_transform(sample_texts)

            # Calculate pairwise cosine similarity
            similarity_matrix = cosine_similarity(tfidf_matrix)

            # Get average similarity
            n = len(similarity_matrix)
            total_similarity = (similarity_matrix.sum() - n) / (n * (n - 1)) if n > 1 else 0

            similarity_pct = total_similarity * 100

            return {'avg_text_similarity_pct': round(similarity_pct, 2)}
        except:
            return {'avg_text_similarity_pct': 0}

    def _calc_username_pattern(self, user_row):
        username = user_row['username']

        # Ends with 6+ digits
        ends_with_digits = bool(re.search(r'\d{6,}$', username))

        # Random mix of letters and numbers
        # Count transitions between letters and numbers
        transitions = 0
        for i in range(len(username) - 1):
            if username[i].isalpha() != username[i + 1].isalpha():
                transitions += 1

        # If many transitions and contains numbers, likely random
        random_mix = transitions >= 3 and any(c.isdigit() for c in username)

        is_bot_username = ends_with_digits or random_mix

        return {
            'bot_username_pattern': is_bot_username,
            'username_ends_with_digits': ends_with_digits,
        }

    def _calc_default_profile(self, user_row):
        profile_url = user_row.get('profileImageUrl', '')

        # Common default image patterns
        default_patterns = [
            'default_profile',
            'default_avatar',
            'egg',
            'placeholder',
        ]

        is_default = any(pattern in profile_url.lower() for pattern in default_patterns)

        return {'has_default_profile_image': is_default}

    def _calc_topic_diversity(self, user_row):
        texts = user_row['text']

        if not texts or len(texts) < 5:
            return {'topic_concentration_pct': 0, 'dominant_keyword': None}

        # Political keywords to check
        political_keywords = [
            'trump', 'biden', 'harris', 'election', 'vote', 'ballot',
            'democrat', 'republican', 'liberal', 'conservative',
            'maga', 'gop', 'dnc', 'rnc'
        ]

        # Count occurrences of each keyword
        keyword_counts = {}
        for keyword in political_keywords:
            count = sum(1 for text in texts if keyword in str(text).lower())
            if count > 0:
                keyword_counts[keyword] = count

        if not keyword_counts:
            return {'topic_concentration_pct': 0, 'dominant_keyword': None}

        # Find most common keyword
        dominant_keyword = max(keyword_counts, key=keyword_counts.get)
        dominant_count = keyword_counts[dominant_keyword]

        concentration_pct = (dominant_count / len(texts) * 100)

        return {
            'topic_concentration_pct': round(concentration_pct, 2),
            'dominant_keyword': dominant_keyword,
        }

    def _tier3_features(self, user_row):
        features = {}

        # Excessive Hashtags
        features.update(self._calc_hashtag_usage(user_row))

        # Retweet-to-Original Ratio
        features.update(self._calc_retweet_ratio(user_row))

        # Profile Completeness
        features.update(self._calc_profile_completeness(user_row))

        # Mention Spam
        features.update(self._calc_mention_spam(user_row))

        # Add verification status
        features['is_verified'] = user_row.get('verified', False)
        features['has_blue'] = user_row.get('blue', False)

        return features

    def _calc_hashtag_usage(self, user_row):
        hashtags = user_row['hashtags']

        if not hashtags:
            return {'avg_hashtags_per_tweet': 0, 'has_long_hashtag': False}

        # Count hashtags per tweet
        hashtag_counts = []
        has_long = False

        for hashtag_list in hashtags:
            if pd.notna(hashtag_list) and hashtag_list != '[]':
                try:
                    # Parse hashtag list
                    if isinstance(hashtag_list, str):
                        # Count commas as proxy for hashtag count
                        count = hashtag_list.count(',') + 1 if hashtag_list != '[]' else 0
                    else:
                        count = len(hashtag_list)

                    hashtag_counts.append(count)

                    # Check for long hashtags
                    if isinstance(hashtag_list, str) and len(hashtag_list) > 30:
                        has_long = True
                except:
                    hashtag_counts.append(0)
            else:
                hashtag_counts.append(0)

        avg_hashtags = np.mean(hashtag_counts) if hashtag_counts else 0

        return {
            'avg_hashtags_per_tweet': round(avg_hashtags, 2),
            'has_long_hashtag': has_long,
        }

    def _calc_retweet_ratio(self, user_row):
        retweets = user_row['retweetedTweet']

        if not retweets:
            return {'retweet_ratio_pct': 0}

        retweet_count = sum(1 for rt in retweets if rt == True or rt == 'True')
        total = len(retweets)

        retweet_pct = (retweet_count / total * 100) if total > 0 else 0

        return {'retweet_ratio_pct': round(retweet_pct, 2)}

    def _calc_profile_completeness(self, user_row):
        # Get scalar values, not Series
        description = user_row.get('rawDescription', '')
        location = user_row.get('location', '')

        # Convert to string if pandas Series/list/etc
        if isinstance(description, (list, pd.Series)):
            description = str(description) if len(description) > 0 else ''
        if isinstance(location, (list, pd.Series)):
            location = str(location) if len(location) > 0 else ''

        # Safe boolean evaluation
        empty_description = (pd.isna(description) or description == '' or description == 'PW')
        no_location = (pd.isna(location) or location == '' or location == 'PW')

        return {
            'has_empty_description': empty_description,
            'has_no_location': no_location,
        }

    def _calc_mention_spam(self, user_row):
        mentions = user_row['mentionedUsers']

        if not mentions:
            return {'avg_mentions_per_tweet': 0}

        # Count mentions per tweet
        mention_counts = []
        for mention_list in mentions:
            if pd.notna(mention_list) and mention_list != '[]':
                try:
                    # Count mentions
                    if isinstance(mention_list, str):
                        count = mention_list.count('id_str')  # Each mention has 'id_str'
                    else:
                        count = len(mention_list)
                    mention_counts.append(count)
                except:
                    mention_counts.append(0)
            else:
                mention_counts.append(0)

        avg_mentions = np.mean(mention_counts) if mention_counts else 0

        return {'avg_mentions_per_tweet': round(avg_mentions, 2)}

    def save_features(self, output_path="output/user_features.csv"):
        # Save extracted features to CSV
        if self.features is None:
            raise ValueError("No features to save")

        print(f"Saving features to {output_path}...")
        self.features.to_csv(output_path, index=False)
        print(f"   - Saved {len(self.features):,} user feature records\n")

    def get_features(self):
        # Get the extracted features DataFrame
        if self.features is None:
            raise ValueError("No features available")
        return self.features
