"""
VISUALIZER MODULE
Creates visualizations and charts for bot detection analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class BotVisualizer:
    def __init__(self, scores_df, features_df, output_dir="output"):
        self.scores = scores_df.copy()
        self.features = features_df.copy()
        self.output_dir = Path(output_dir) / "visualizations"

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Merge scores and features for easier analysis
        self.data = self.scores.merge(self.features, on=['username', 'user_id'], how='left')

    def create_all_visualizations(self):
        print("-" * 100)
        print("GENERATING VISUALIZATIONS")
        print("-" * 100)

        print("  1. Bot score distribution histogram")
        self.plot_score_distribution()

        print("  2. Classification breakdown pie chart")
        self.plot_classification_breakdown()

        print("  3. Tier score comparison")
        self.plot_tier_scores()

        print("  4. Top 20 suspicious accounts")
        self.plot_top_suspicious_accounts()

        print("  5. Feature importance chart")
        self.plot_feature_importance()

        print("  6. Account age distribution")
        self.plot_account_age_distribution()

    def plot_score_distribution(self):
        """
        Create histogram of bot score distribution.

        Shows how scores are distributed across all users.
        """
        plt.figure(figsize=(12, 6))

        # Create histogram
        plt.hist(self.scores['total_score'], bins=50, edgecolor='black', alpha=0.7)

        # Add vertical lines for classification thresholds
        plt.axvline(35, color='green', linestyle='--', linewidth=2, label='Human/Suspicious (35)')
        plt.axvline(60, color='orange', linestyle='--', linewidth=2, label='Suspicious/Likely Bot (60)')
        plt.axvline(85, color='red', linestyle='--', linewidth=2, label='Likely Bot/Definite Bot (85)')

        plt.xlabel('Bot Score (max 220)', fontsize=12, fontweight='bold')
        plt.ylabel('Number of Users', fontsize=12, fontweight='bold')
        plt.title('Distribution of Bot Scores Across All Users', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'score_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_classification_breakdown(self):
        """
        Creates pie chart showing classification breakdown
        """
        plt.figure(figsize=(10, 8))

        # Count classifications
        classification_counts = self.scores['classification'].value_counts()

        # Colors for each classification
        colors = {
            'Likely Human': '#2ecc71',      # Green
            'Suspicious': '#f39c12',        # Orange
            'Likely Bot': '#e74c3c',        # Red
            'Definite Bot': '#c0392b'       # Dark red
        }

        # Create ordered colors list
        color_list = [colors[label] for label in classification_counts.index]

        # Create pie chart
        plt.pie(classification_counts.values,
                labels=classification_counts.index,
                autopct='%1.1f%%',
                startangle=90,
                colors=color_list,
                textprops={'fontsize': 12, 'fontweight': 'bold'})

        plt.title('User Classification Breakdown', fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'classification_breakdown.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_tier_scores(self):
        """
        Creates boxplot comparing tier scores across classifications
        """
        fig, axes = plt.subplots(1, 3, figsize=(16, 6))

        order = ['Likely Human', 'Suspicious', 'Likely Bot', 'Definite Bot']

        # Tier 1 scores
        sns.boxplot(data=self.scores, x='classification', y='tier1_score',
                   order=order, ax=axes[0], palette='Reds')
        axes[0].set_title('Tier 1 Scores (Max: 180)', fontweight='bold')
        axes[0].set_xlabel('Classification', fontweight='bold')
        axes[0].set_ylabel('Score', fontweight='bold')
        axes[0].tick_params(axis='x', rotation=45)

        # Tier 2 scores
        sns.boxplot(data=self.scores, x='classification', y='tier2_score',
                   order=order, ax=axes[1], palette='Oranges')
        axes[1].set_title('Tier 2 Scores (Max: 20)', fontweight='bold')
        axes[1].set_xlabel('Classification', fontweight='bold')
        axes[1].set_ylabel('Score', fontweight='bold')
        axes[1].tick_params(axis='x', rotation=45)

        # Tier 3 scores
        sns.boxplot(data=self.scores, x='classification', y='tier3_score',
                   order=order, ax=axes[2], palette='Blues')
        axes[2].set_title('Tier 3 Scores (Max: 20)', fontweight='bold')
        axes[2].set_xlabel('Classification', fontweight='bold')
        axes[2].set_ylabel('Score', fontweight='bold')
        axes[2].tick_params(axis='x', rotation=45)

        plt.suptitle('Tier Score Comparison Across Classifications', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'tier_scores_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_top_suspicious_accounts(self):
        """
        Creates bar chart of top 20 most suspicious accounts
        """
        plt.figure(figsize=(12, 10))

        # Get top 20 accounts by score
        top_accounts = self.scores.nlargest(20, 'total_score')

        # Create horizontal bar chart
        colors = top_accounts['classification'].map({
            'Likely Human': '#2ecc71',
            'Suspicious': '#f39c12',
            'Likely Bot': '#e74c3c',
            'Definite Bot': '#c0392b'
        })

        plt.barh(range(len(top_accounts)), top_accounts['total_score'], color=colors)
        plt.yticks(range(len(top_accounts)), top_accounts['username'], fontsize=9)
        plt.xlabel('Bot Score (max 220)', fontsize=12, fontweight='bold')
        plt.ylabel('Username', fontsize=12, fontweight='bold')
        plt.title('Top 20 Most Suspicious Accounts', fontsize=14, fontweight='bold')
        plt.axvline(86, color='red', linestyle='--', linewidth=2, label='Flagged Threshold (86)')
        plt.legend()
        plt.grid(True, axis='x', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'top_suspicious_accounts.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_feature_importance(self):
        """
        Creates bar chart showing avg feature values by classification
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        features_to_plot = [
            ('avg_engagement_per_tweet', 'Average Engagement per Tweet'),
            ('lifetime_tweets_per_day', 'Lifetime Tweet Rate (tweets/day)')
        ]

        order = ['Likely Human', 'Suspicious', 'Likely Bot', 'Definite Bot']

        for idx, (feature, title) in enumerate(features_to_plot):
            ax = axes[idx]

            # Calculate mean by classification
            means = self.data.groupby('classification')[feature].mean().reindex(order)

            # Create bar plot
            bars = ax.bar(range(len(means)), means.values,
                         color=['#2ecc71', '#f39c12', '#e74c3c', '#c0392b'])

            ax.set_xticks(range(len(means)))
            ax.set_xticklabels(means.index, rotation=45, ha='right')
            ax.set_ylabel('Average Value', fontweight='bold')
            ax.set_title(title, fontweight='bold')
            ax.grid(True, alpha=0.3)

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom', fontsize=9)

        plt.suptitle('Key Feature Averages by Classification', fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_account_age_distribution(self):
        """
        Creates histogram showing account age distribution by classification
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        order = ['Likely Human', 'Suspicious', 'Likely Bot', 'Definite Bot']
        colors = ['#2ecc71', '#f39c12', '#e74c3c', '#c0392b']

        for idx, classification in enumerate(order):
            ax = axes[idx // 2, idx % 2]

            subset = self.data[self.data['classification'] == classification]

            # Filter out None values
            age_data = subset['account_age_days'].dropna()

            if len(age_data) > 0:
                ax.hist(age_data, bins=30, color=colors[idx], alpha=0.7, edgecolor='black')
                ax.set_xlabel('Account Age (days)', fontweight='bold')
                ax.set_ylabel('Number of Accounts', fontweight='bold')
                ax.set_title(f'{classification} (n={len(age_data)})', fontweight='bold')
                ax.grid(True, alpha=0.3)

                # Add median line
                median_age = age_data.median()
                ax.axvline(median_age, color='red', linestyle='--', linewidth=2,
                          label=f'Median: {median_age:.0f} days')
                ax.legend()

        plt.suptitle('Account Age Distribution by Classification', fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'account_age_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

