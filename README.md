# Twitter Bot Detection System
## 2024 US Election Analysis


**Dataset:** 100,000 tweets from October 2024 (2024 US Election period)
Dataset obtained from:
Ashwin Balasubramanian, Vito Zou, Hitesh Narayana, Christina You, Luca Luceri, Emilio Ferrara
University of Southern California

GitHub Repo containing the Dataset:
https://github.com/sinking8/x-24-us-election


## Project Overview
This project implements a rule-based bot detection system for Twitter/X accounts using behavioral analysis. Unlike black-box machine learning approaches, we implemented a tiered system based off common traits we found amongst bots. To identify potential bots, we look for 16 distinct behavioral indicators organized into three tiers.

The purpose of this repository is to detect bots and troll farms during the 2024 US Presidential Election.


## Key Features
- **16 Bot Detection Indicators** organized into 3 tiers by level of importance
- **User-level Analysis** (aggregates tweet-level data by account)
- **Tiered Scoring System** (0-185 points with clear thresholds)
- **Visualizations** (9 charts + summary dashboard)

---

## Methodology
### User Scoring System
- **0-35 points:** Likely Human-- normal user behavior, low bot indicators
- **36-60 points:** Suspicious-- user has some concerning traits associated with bots, needs review
- **61-85 points:** Likely Bot--user has strong traits associated with bots, probably a bot
- **86+ points:** Definite Bot--a bot, account has irregular human user behavior


### Detection Criteria
#### **TIER 1 (15 points each, max 135)**
High-importance indicators of bot behavior:

1. **Troll Farm Membership:** Member of coordinated bot network = 15 pts
2. **Low Engagement Rate:** <0.5 avg interactions = 15 pts, ≤2 = 10 pts, ≤5 = 5 pts
3. **Favorites-to-Tweets Ratio:** <0.1 = 15 pts, ≤0.3 = 10 pts, ≤0.5 = 5 pts
4. **Account Age:** Created same month as dataset = 15 pts, created in 2024 = 7.5 pts
5. **Account Activity:** >100 tweets/day since creation = 15 pts, ≥50 = 10 pts, ≥25 = 5 pts
6. **Reply Ratio:** ≥80% replies = 15 pts, ≥66% = 10 pts, ≥50% = 5 pts
7. **Text Similarity:** >80% similar = 15 pts, ≥60% = 10 pts, ≥40% = 5 pts
8. **Topic Concentration:** 100% = 15 pts, ≥80% = 10 pts, >70% = 5 pts

#### **TIER 2 (10 points each, max 30)**
Medium-importance indicators:

1. **High Posting Frequency:** >50 tweets/day = 10 pts, ≥30 = 7 pts, ≥20 = 3 pts
2. **Alphanumeric Username:** Bot-like naming pattern = 10 pts
3. **Default Profile Image:** Generic avatar = 10 pts

#### **TIER 3 (5 points each, max 20)**
Lower-importance indicators:

1. **Excessive Hashtags:** >3 per tweet = 5 pts
2. **Retweet Ratio:** >80% retweets vs original content = 5 pts
3. **Profile Completeness:** Empty bio + no location = 5 pts
4. **Mention Spam:** >3 mentions per tweet = 5 pts


### Special Rules
- **Verified Accounts:** Receive 25% score reduction

---

## Troll Farm Indicators
**What we Look For:**
- 10+ accounts posting the exact same message (exact text match)
-  over 50% of those accounts classified as bots
- Original tweets only (retweets excluded)

---

## Installation & Setup
### Prerequisites
- Python 3.8 or higher

### Installation
pip install -r requirements.txt


### Running
python3 main.py

### Data Files
1. **user_aggregated_data.csv**
   - User-level aggregation of all tweets
   - Contains: username, follower counts, all tweets lists, profile info

2. **user_features.csv**
   - All 16 calculated bot indicators per user
   - Contains: detection criteria for each username

3. **troll_farm_members.csv**
   - Accounts identified as a part of troll farms
   - Contains: username, shared message, bot farm size, percentage of accounts suspected to be bots, message frequency

4. **user_scores.csv**
   - Final scores and classifications (with troll farm bonuses)
   - Contains: how users scored for each tier (ex: tier 1 score, tier 2, etc..)

5. **user_scores_initial.csv**
   - Initial scores before troll farm detection update (for reference)
   - Contains: how users scored for each tier BEFORE troll farm point update

6. **flagged_bots.csv**
   - Accounts flagged as definite bot
   - Contains: detailed profile information for each bot

### Visualizations
1. **score_distribution.png** - Histogram of bot scores with threshold lines
2. **classification_breakdown.png** - Pie chart of user classifications
3. **tier_scores_comparison.png** - Boxplots comparing tier scores
4. **top_suspicious_accounts.png** - Top 20 highest-scoring accounts
5. **feature_importance.png** - Key feature averages by classification
6. **account_age_distribution.png** - Account creation date patterns


### Limitations
1. **Rule-based System:** We may miss complex bots that are more fine-tuned to mimic human behavior
2. **False Positives:** Users that post a lot may score high
3. **Text Normalization Matching:** Troll farm detection uses normalized text (lowercase, URL removal, whitespace trimming), so may miss coordinated campaigns with synonym substitution or semantic variations

# NOTE: High/low scores don't always prove bot/human status, so you should always verify manually