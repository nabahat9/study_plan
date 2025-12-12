import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import warnings
import pickle
import json
import os
from typing import List, Dict, Any

warnings.filterwarnings('ignore')

class StudyPlanner:
    def __init__(self):
        """Initialize with pre-trained model"""
        self.model = None
        self.rules = None
        self.study_methods = {
            "Hard": "Spaced Repetition + Practice Problems",
            "Medium": "Active Recall + Summarization",
            "Easy": "Quick Review + Flashcards"
        }
        
        # Load pre-trained model
        self._load_model()
    
    def _load_model(self):
        """Load pre-trained ECLAT model"""
        model_path = "trained_models/eclat_model.pkl"
        
        try:
            if os.path.exists(model_path):
                print(f"📦 Loading pre-trained model from {model_path}...")
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                
                self.rules = self.model.get('rules', pd.DataFrame())
                print(f"✅ Model loaded successfully!")
                print(f"   - Rules: {len(self.rules)}")
                
                # Load metadata if exists
                metadata_path = "trained_models/metadata.json"
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    print(f"   - Trained on: {metadata.get('training_date', 'Unknown')}")
                    print(f"   - Dataset: {metadata.get('dataset_size', 'Unknown')} samples")
            else:
                print("⚠️  No pre-trained model found. Using fallback mode.")
                self.rules = pd.DataFrame()
                
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("⚠️  Using fallback mode without ECLAT rules.")
            self.rules = pd.DataFrame()
    
    def generate_study_plan(self, user_input: Dict[str, Any]) -> List[Dict]:
        """Generate study plan using pre-trained model"""
        print(f"📚 Generating study plan for: {user_input.get('goal', 'Unknown')}")
        
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
        
        # Find matching ECLAT rules (if model loaded)
        matching_rules = []
        if self.rules is not None and len(self.rules) > 0:
            user_features = self._extract_user_features(user_input)
            for _, rule in self.rules.iterrows():
                antecedents = set(rule.get('antecedents', []))
                if antecedents and antecedents.issubset(set(user_features)):
                    matching_rules.append(rule.to_dict())
        
        # Sort subjects based on chronotype
        subjects = user_input['subjects']
        difficulties = user_input['subject_difficulty']
        
        if len(subjects) != len(difficulties):
            difficulties = difficulties[:len(subjects)] if len(difficulties) > len(subjects) else difficulties + ['Medium'] * (len(subjects) - len(difficulties))
        
        subjects_with_diff = list(zip(subjects, difficulties))
        
        chronotype = user_input.get('chronotype', 'flexible')
        if chronotype == 'morning':
            subjects_with_diff.sort(key=lambda x: {"Hard": 3, "Medium": 2, "Easy": 1}[x[1]], reverse=True)
        elif chronotype == 'night':
            subjects_with_diff.sort(key=lambda x: {"Hard": 3, "Medium": 2, "Easy": 1}[x[1]])
        
        # Generate daily plans
        daily_plans = []
        
        for day in range(min(total_days, 14)):  # Max 14 days
            current_date = today + timedelta(days=day)
            
            # Skip weekends for long plans
            if total_days > 7 and current_date.weekday() >= 5:
                continue
            
            day_plan = self._generate_daily_plan(current_date, subjects_with_diff, user_input, matching_rules)
            
            if day_plan['sessions']:
                daily_plans.append(day_plan)
        
        print(f"✅ Generated {len(daily_plans)} daily plans")
        return daily_plans
    
    def _extract_user_features(self, user_input: Dict) -> List[str]:
        """Extract features for rule matching"""
        features = []
        
        # Add subjects
        features.extend([f"subject_{s.lower().replace(' ', '_')}" for s in user_input['subjects'][:3]])
        
        # Add difficulties
        features.extend([f"difficulty_{d.lower()}" for d in user_input['subject_difficulty'][:3]])
        
        # Add chronotype
        chronotype = user_input.get('chronotype', 'flexible')
        features.append(f"chronotype_{chronotype}")
        
        # Add stress level
        stress = user_input.get('stress_tolerance', 0.7)
        if stress > 0.7:
            features.append("high_stress_tolerance")
        elif stress < 0.4:
            features.append("low_stress_tolerance")
        
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
