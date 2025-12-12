import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import warnings
import itertools
import json
import os
from typing import List, Dict, Any

warnings.filterwarnings('ignore')

class StudyPlanner:
    def __init__(self, data_path="data/study_plans.csv"):
        self.data_path = data_path
        self.df = None
        self.rules = None
        self.study_methods = {
            "Hard": "Spaced Repetition + Practice Problems",
            "Medium": "Active Recall + Summarization",
            "Easy": "Quick Review + Flashcards"
        }
        
    def load_and_preprocess(self):
        """Load and preprocess the dataset"""
        print("📊 Loading dataset...")
        
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"✅ Loaded {len(self.df)} rows")
        except Exception as e:
            print(f"❌ Failed to load CSV: {e}")
            # Create sample data if file doesn't exist
            self.df = self._create_sample_data()
        
        # Preprocessing
        self._preprocess_data()
        return self.df
    
    def _create_sample_data(self):
        """Create sample data for testing"""
        print("📝 Creating sample data...")
        data = {
            "gender": ["Male", "Female", "Male", "Female", "Male"],
            "age": [20, 21, 22, 23, 24],
            "chronotype": ["morning", "night", "morning", "flexible", "night"],
            "learning_style": ["visual", "auditory", "kinesthetic", "visual", "auditory"],
            "focus_duration": [45, 50, 40, 55, 60],
            "stress_tolerance": [0.7, 0.8, 0.5, 0.9, 0.6],
            "subject_difficulty": ["Hard", "Medium", "Easy", "Hard", "Medium"],
            "subjects": ['["math", "physics"]', '["chemistry"]', '["biology"]', '["math"]', '["physics", "chemistry"]'],
            "daily_hours_available": [3, 4, 2, 5, 3],
            "goal": ["Exam", "Project", "Learning", "Exam", "Project"],
            "deadline": ["2024-12-31", "2024-11-30", "2024-10-31", "2024-12-15", "2024-11-15"],
            "plan_created_at": ["2024-10-01", "2024-10-01", "2024-10-01", "2024-10-01", "2024-10-01"]
        }
        return pd.DataFrame(data)
    
    def _preprocess_data(self):
        """Preprocess the dataset"""
        print("🧹 Preprocessing data...")
        
        # Handle missing values
        num_cols = ["age", "focus_duration", "daily_hours_available"]
        for col in num_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                self.df[col] = self.df[col].fillna(self.df[col].median())
        
        cat_cols = ["gender", "chronotype", "learning_style", "stress_tolerance", "goal"]
        for col in cat_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0] if not self.df[col].mode().empty else 'Unknown')
        
        # Process subject difficulty
        if "subject_difficulty" in self.df.columns:
            def parse_difficulty(diff):
                if pd.isna(diff):
                    return 2
                if isinstance(diff, str):
                    diff = diff.lower()
                    if 'hard' in diff or '3' in diff:
                        return 3
                    elif 'medium' in diff or '2' in diff:
                        return 2
                    elif 'easy' in diff or '1' in diff:
                        return 1
                return 2
            
            self.df["max_difficulty"] = self.df["subject_difficulty"].apply(parse_difficulty)
            self.df = self.df.drop("subject_difficulty", axis=1)
        
        # Process subjects
        if "subjects" in self.df.columns:
            def extract_subjects(x):
                if pd.isna(x) or str(x).lower() in ['nan', 'none', '']:
                    return []
                if isinstance(x, str):
                    if x.startswith('[') and x.endswith(']'):
                        import ast
                        try:
                            return ast.literal_eval(x)
                        except:
                            pass
                    if ',' in x:
                        return [s.strip() for s in x.split(',')]
                return [str(x)]
            
            self.df["subjects_list"] = self.df["subjects"].apply(extract_subjects)
            
            # Get unique subjects
            all_subjects = []
            for sublist in self.df["subjects_list"]:
                all_subjects.extend(sublist)
            unique_subjects = list(set(all_subjects))[:10]
            
            # Create binary columns
            for subject in unique_subjects:
                col_name = f"subject_{subject.replace(' ', '_').lower()[:15]}"
                self.df[col_name] = self.df["subjects_list"].apply(lambda x: 1 if subject in x else 0)
            
            self.df = self.df.drop(["subjects", "subjects_list"], axis=1)
        
        # Process dates
        if "deadline" in self.df.columns and "plan_created_at" in self.df.columns:
            self.df["deadline"] = pd.to_datetime(self.df["deadline"], errors='coerce')
            self.df["plan_created_at"] = pd.to_datetime(self.df["plan_created_at"], errors='coerce')
            
            # Remove timezone
            for col in ["deadline", "plan_created_at"]:
                if hasattr(self.df[col].dt, 'tz') and self.df[col].dt.tz is not None:
                    self.df[col] = self.df[col].dt.tz_localize(None)
            
            self.df["study_duration_days"] = (self.df["deadline"] - self.df["plan_created_at"]).dt.days
            self.df["study_duration_days"] = self.df["study_duration_days"].fillna(30)
            self.df = self.df.drop(["deadline", "plan_created_at"], axis=1)
        
        print(f"✅ Preprocessed data shape: {self.df.shape}")
    
    def run_eclat(self, min_support=0.15, min_confidence=0.6):
        """Run ECLAT algorithm to find association rules"""
        print("🔍 Running ECLAT algorithm...")
        
        # Convert to binary for ECLAT
        df_binary = self.df.copy()
        for col in df_binary.columns:
            if df_binary[col].dtype in ['int64', 'float64']:
                median_val = df_binary[col].median()
                df_binary[col] = (df_binary[col] > median_val).astype(bool)
            else:
                df_binary[col] = df_binary[col].astype(bool)
        
        items = list(df_binary.columns)
        n_transactions = len(df_binary)
        min_support_count = int(min_support * n_transactions)
        
        # Vertical database format
        vertical_db = {}
        for item in items:
            vertical_db[item] = set(df_binary.index[df_binary[item]].tolist())
        
        # Find frequent 1-itemsets
        frequent_itemsets = {}
        for item, tids in vertical_db.items():
            if len(tids) >= min_support_count:
                frequent_itemsets[frozenset([item])] = len(tids)
        
        print(f"Found {len(frequent_itemsets)} frequent 1-itemsets")
        
        # Generate larger itemsets (up to 3-itemsets)
        for k in [2, 3]:
            new_itemsets = {}
            itemsets_list = list(frequent_itemsets.keys())
            
            for i in range(len(itemsets_list)):
                for j in range(i + 1, len(itemsets_list)):
                    itemset1 = itemsets_list[i]
                    itemset2 = itemsets_list[j]
                    
                    if len(itemset1.union(itemset2)) == k:
                        candidate = itemset1.union(itemset2)
                        cols = list(candidate)
                        support_count = (df_binary[cols].all(axis=1)).sum()
                        
                        if support_count >= min_support_count:
                            new_itemsets[candidate] = support_count
            
            if new_itemsets:
                frequent_itemsets.update(new_itemsets)
                print(f"Found {len(new_itemsets)} frequent {k}-itemsets")
        
        # Generate association rules
        rules = []
        for itemset, support_count in frequent_itemsets.items():
            if len(itemset) >= 2:
                itemset_support = support_count / n_transactions
                items_list = list(itemset)
                
                for i in range(1, len(items_list)):
                    for antecedent in itertools.combinations(items_list, i):
                        antecedent_set = frozenset(antecedent)
                        consequent_set = frozenset(items_list) - antecedent_set
                        
                        antecedent_support = frequent_itemsets.get(antecedent_set, 0) / n_transactions
                        if antecedent_support > 0:
                            confidence = itemset_support / antecedent_support
                            
                            if confidence >= min_confidence:
                                rules.append({
                                    'antecedents': list(antecedent_set),
                                    'consequents': list(consequent_set),
                                    'support': itemset_support,
                                    'confidence': confidence
                                })
        
        self.rules = pd.DataFrame(rules)
        if len(self.rules) > 0:
            self.rules = self.rules.sort_values('confidence', ascending=False)
        
        print(f"✅ Generated {len(self.rules)} association rules")
        return self.rules
    
    def generate_study_plan(self, user_input: Dict[str, Any]) -> List[Dict]:
        """Generate personalized study plan"""
        print(f"📚 Generating study plan for: {user_input['goal']}")
        
        # Validate input
        required = ['subjects', 'subject_difficulty', 'daily_hours_available', 'deadline', 'goal']
        for key in required:
            if key not in user_input:
                raise ValueError(f"Missing required field: {key}")
        
        # Parse dates
        try:
            today = datetime.now()
            deadline = datetime.strptime(user_input['deadline'], '%Y-%m-%d')
            total_days = max(1, (deadline - today).days)
            total_days = min(total_days, 30)  # Limit to 30 days
        except Exception as e:
            print(f"⚠️ Date parsing error: {e}, using 7 days")
            total_days = 7
        
        # Find matching ECLAT rules
        matching_rules = []
        if self.rules is not None and len(self.rules) > 0:
            user_features = self._extract_user_features(user_input)
            for _, rule in self.rules.iterrows():
                if set(rule['antecedents']).issubset(set(user_features)):
                    matching_rules.append(rule.to_dict())
        
        # Sort subjects based on chronotype
        subjects_with_diff = list(zip(user_input['subjects'], user_input['subject_difficulty']))
        
        if user_input.get('chronotype') == 'morning':
            subjects_with_diff.sort(key=lambda x: {"Hard": 3, "Medium": 2, "Easy": 1}[x[1]], reverse=True)
        elif user_input.get('chronotype') == 'night':
            subjects_with_diff.sort(key=lambda x: {"Hard": 3, "Medium": 2, "Easy": 1}[x[1]])
        
        # Generate daily plans
        daily_plans = []
        
        for day in range(total_days):
            current_date = today + timedelta(days=day)
            
            # Skip weekends for long plans
            if total_days > 14 and current_date.weekday() >= 5:
                continue
            
            day_plan = self._generate_daily_plan(
                current_date, subjects_with_diff, user_input, matching_rules
            )
            
            if day_plan['sessions']:
                daily_plans.append(day_plan)
            
            # Stop if we have enough plans
            if len(daily_plans) >= 14:  # Max 2 weeks
                break
        
        print(f"✅ Generated {len(daily_plans)} daily plans")
        return daily_plans
    
    def _extract_user_features(self, user_input: Dict) -> List[str]:
        """Extract features for rule matching"""
        features = []
        
        # Add subjects
        features.extend([f"subject_{s.lower().replace(' ', '_')}" for s in user_input['subjects']])
        
        # Add difficulties
        features.extend([f"difficulty_{d.lower()}" for d in user_input['subject_difficulty']])
        
        # Add chronotype
        chronotype = user_input.get('chronotype', 'flexible')
        features.append(f"chronotype_{chronotype}")
        
        # Add stress level
        stress = user_input.get('stress_tolerance', 0.7)
        if stress > 0.7:
            features.append("high_stress_tolerance")
        elif stress < 0.4:
            features.append("low_stress_tolerance")
        else:
            features.append("medium_stress_tolerance")
        
        return features
    
    def _generate_daily_plan(self, date, subjects_with_diff, user_input, matching_rules):
        """Generate plan for a single day"""
        sessions = []
        total_minutes = user_input['daily_hours_available'] * 60
        
        # Determine start time based on chronotype
        chronotype = user_input.get('chronotype', 'flexible')
        if chronotype == 'morning':
            start_hour = 8
        elif chronotype == 'night':
            start_hour = 18
        else:
            start_hour = 10
        
        current_time = datetime.combine(date.date(), time(start_hour, 0))
        session_num = 1
        
        for subject, difficulty in subjects_with_diff:
            if total_minutes <= 30:  # Minimum time needed
                break
            
            # Calculate session duration
            base_duration = user_input.get('focus_duration', 45)
            
            # Adjust based on difficulty
            if difficulty == "Hard":
                duration = int(base_duration * 1.1)
            elif difficulty == "Medium":
                duration = base_duration
            else:  # Easy
                duration = int(base_duration * 0.9)
            
            # Adjust based on stress tolerance
            stress = user_input.get('stress_tolerance', 0.7)
            duration = int(duration * (0.6 + stress * 0.8))
            
            # Ensure reasonable bounds
            duration = max(25, min(duration, 90))
            
            if duration + user_input.get('break_duration', 10) > total_minutes:
                duration = total_minutes - user_input.get('break_duration', 10)
            
            if duration < 20:
                break
            
            # Apply ECLAT rule adjustments if any
            for rule in matching_rules[:2]:  # Apply top 2 rules
                consequents = rule.get('consequents', [])
                if any('long' in str(c) for c in consequents):
                    duration = int(duration * 1.2)
                elif any('short' in str(c) for c in consequents):
                    duration = int(duration * 0.8)
            
            # Create session
            end_time = current_time + timedelta(minutes=duration)
            
            session = {
                'session_number': session_num,
                'subject': subject,
                'difficulty': difficulty,
                'start_time': current_time.strftime("%H:%M"),
                'end_time': end_time.strftime("%H:%M"),
                'duration_minutes': duration,
                'break_minutes': user_input.get('break_duration', 10),
                'study_method': self.study_methods.get(difficulty, "Standard Study")
            }
            
            sessions.append(session)
            
            # Update for next session
            total_minutes -= (duration + user_input.get('break_duration', 10))
            current_time = end_time + timedelta(minutes=user_input.get('break_duration', 10))
            session_num += 1
        
        # Calculate totals
        total_study = sum(s['duration_minutes'] for s in sessions)
        total_break = sum(s['break_minutes'] for s in sessions)
        
        return {
            'date': date.strftime("%Y-%m-%d"),
            'day_name': date.strftime("%A"),
            'sessions': sessions,
            'total_study_minutes': total_study,
            'total_break_minutes': total_break
        }
    
    def get_rules(self, limit: int = 20):
        """Get association rules"""
        if self.rules is None or len(self.rules) == 0:
            return []
        
        return self.rules.head(limit).to_dict('records')