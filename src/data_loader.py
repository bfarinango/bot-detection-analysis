"""
DATA LOADER MODULE
This script loads and processes data from CSV files. 
It's the first step in the pipeline and organizes tweets by user.
"""

import pandas as pd
import json
import ast
from pathlib import Path


class DataLoader:
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.raw_tweets = None
        self.user_data = None

    def load_csv_files(self, file_patterns=["october_chunk_*.csv.gz"]):

        all_dataframes = []

        # Find all files matching the pattern
        data_files = []
        for pattern in file_patterns:
            data_files.extend(self.data_dir.glob(pattern))

        # Load each file one at a time
        for file_path in data_files:
            try:
                df = pd.read_csv(file_path, compression='gzip')
                all_dataframes.append(df)

            except Exception as e:
                print(f" ERROR loading {file_path.name}: {e}")
                continue

        # Combine all dataframes into one big dataframe
        if all_dataframes:
            self.raw_tweets = pd.concat(all_dataframes, ignore_index=True)
            print(f"Total tweets loaded: {len(self.raw_tweets):,}")
            print(f"Unique users: {self.raw_tweets['username'].nunique():,}\n")
            return self.raw_tweets
        else:
            raise ValueError("No data files were successfully loaded!")

    def parse_user_dict(self, user_string):
        """
        Parse the 'user' field which is stored as a string representation of a dict

        Returns:
            dict: Parsed user information, or empty dict if parsing fails
        """
        # Handle missing/null values
        if pd.isna(user_string) or user_string == 'PW':
            return {}

        try:
            # Replace datetime.datetime() patterns with None for safe parsing
            import re
            from datetime import datetime, timezone

            # Remove datetime objects
            cleaned = re.sub(r'datetime\.datetime\([^)]+\)', 'None', str(user_string))

            # Try parsing the cleaned string
            try:
                user_dict = ast.literal_eval(cleaned)
            except:
                user_dict = {}

            # Extract the actual datetime if present in original string
            datetime_match = re.search(r'datetime\.datetime\((\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)', str(user_string))
            if datetime_match and 'created' in user_dict:
                year, month, day, hour, minute, second = map(int, datetime_match.groups())
                user_dict['created'] = datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)

            return user_dict

        except Exception as e:
            # If all parsing fails, return empty dict
            return {}

    def extract_user_fields(self, row):
        # Parse the user dictionary
        user_dict = self.parse_user_dict(row.get('user', '{}'))

        # Extract fields with safe defaults (use .get() to avoid KeyError)
        return {
            'user_id': user_dict.get('id_str', user_dict.get('id', 'unknown')),
            'username': row.get('username', 'unknown'),
            'created': user_dict.get('created', None),  # Account creation date
            'followersCount': user_dict.get('followersCount', 0),
            'friendsCount': user_dict.get('friendsCount', 0),  # Following count
            'statusesCount': user_dict.get('statusesCount', 0),  # Total tweets
            'favouritesCount': user_dict.get('favouritesCount', 0),  # Total likes
            'location': user_dict.get('location', ''),
            'rawDescription': user_dict.get('rawDescription', ''),  # Bio
            'profileImageUrl': user_dict.get('profileImageUrl', ''),
            'verified': user_dict.get('verified', False),
            'blue': user_dict.get('blue', False),  # Verified account
        }

    def aggregate_by_user(self):
        """
        Group all tweets by username and organize user-level data.

        Returns:
            pandas.DataFrame: One row per user with all their tweets
        """

        if self.raw_tweets is None:
            raise ValueError("No data loaded")

        print("Extracting user profile information...")
        print("Processing tweets...")

        # Extract user info from each tweet
        user_profiles = self.raw_tweets.apply(
            self.extract_user_fields,
            axis=1
        )

        # Convert list of dicts to dataframe
        user_profiles_df = pd.DataFrame(list(user_profiles))

        # Drop 'username' from user_profiles_df to avoid duplication (already exists in raw_tweets)
        user_profiles_df = user_profiles_df.drop(columns=['username'])

        # Add user profiles to the main dataframe
        self.raw_tweets = pd.concat([self.raw_tweets, user_profiles_df], axis=1)


        # Group by username and manually aggregate to avoid issues
        user_groups = []

        for username, group in self.raw_tweets.groupby('username'):
            user_record = {'username': username}

            # Get user profile fields
            if 'user_id' in group.columns:
                user_record['user_id'] = group['user_id'].iloc[0]
            if 'created' in group.columns:
                user_record['created'] = group['created'].iloc[0]
            if 'followersCount' in group.columns:
                user_record['followersCount'] = group['followersCount'].iloc[0]
            if 'friendsCount' in group.columns:
                user_record['friendsCount'] = group['friendsCount'].iloc[0]
            if 'statusesCount' in group.columns:
                user_record['statusesCount'] = group['statusesCount'].iloc[0]
            if 'favouritesCount' in group.columns:
                user_record['favouritesCount'] = group['favouritesCount'].iloc[0]
            if 'location' in group.columns:
                user_record['location'] = group['location'].iloc[0]
            if 'rawDescription' in group.columns:
                user_record['rawDescription'] = group['rawDescription'].iloc[0]
            if 'profileImageUrl' in group.columns:
                user_record['profileImageUrl'] = group['profileImageUrl'].iloc[0]
            if 'verified' in group.columns:
                user_record['verified'] = group['verified'].iloc[0]
            if 'blue' in group.columns:
                user_record['blue'] = group['blue'].iloc[0]

            # Collect tweet-level data into lists
            if 'text' in group.columns:
                user_record['text'] = group['text'].tolist()
            if 'epoch' in group.columns:
                user_record['epoch'] = group['epoch'].tolist()
            if 'replyCount' in group.columns:
                user_record['replyCount'] = group['replyCount'].tolist()
            if 'retweetCount' in group.columns:
                user_record['retweetCount'] = group['retweetCount'].tolist()
            if 'likeCount' in group.columns:
                user_record['likeCount'] = group['likeCount'].tolist()
            if 'hashtags' in group.columns:
                user_record['hashtags'] = group['hashtags'].tolist()
            if 'mentionedUsers' in group.columns:
                user_record['mentionedUsers'] = group['mentionedUsers'].tolist()
            if 'links' in group.columns:
                user_record['links'] = group['links'].tolist()
            if 'retweetedTweet' in group.columns:
                user_record['retweetedTweet'] = group['retweetedTweet'].tolist()
            if 'quotedTweet' in group.columns:
                user_record['quotedTweet'] = group['quotedTweet'].tolist()
            if 'in_reply_to_status_id_str' in group.columns:
                user_record['in_reply_to_status_id_str'] = group['in_reply_to_status_id_str'].tolist()

            user_groups.append(user_record)

        # Convert list of dicts to DataFrame
        user_groups = pd.DataFrame(user_groups)

        self.user_data = user_groups
        return self.user_data

    def get_user_data(self):
        # Get the processed user-level data
        if self.user_data is None:
            raise ValueError("No user data available.")
        return self.user_data

    def save_user_data(self, output_path="output/user_aggregated_data.csv"):
        # Save the aggregated user data to a CSV file
        if self.user_data is None:
            raise ValueError("No user data to save")

        print(f"\nSaving aggregated user data to {output_path}...")
        self.user_data.to_csv(output_path, index=False)
        print(f"   - Saved {len(self.user_data):,} user records\n")

