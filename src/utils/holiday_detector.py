"""
Holiday and Special Event Detection for EagleEye
Provides Egypt-specific holiday detection and special event handling for demand forecasting.
"""

import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Set


class HolidayDetector:
    """
    Detects holidays and special events for demand forecasting adjustment.
    Focuses on Egypt-specific holidays and regional events.
    """
    
    # Fixed Egyptian holidays (month, day)
    FIXED_HOLIDAYS = {
        (1, 1): "New Year's Day",
        (1, 25): "Revolution Day",
        (4, 25): "Sinai Liberation Day",
        (5, 1): "Labour Day",
        (6, 30): "June 30 Revolution",
        (7, 23): "Revolution Day (1952)",
        (10, 6): "Armed Forces Day",
        (12, 25): "Christmas",
    }
    
    # Islamic holidays (approximate - shift by ~11 days each year)
    ISLAMIC_HOLIDAYS_2024 = {
        (9, 15): "Prophet's Birthday (Mawlid)",
    }
    
    ISLAMIC_HOLIDAYS_2025 = {
        (9, 4): "Prophet's Birthday (Mawlid)",
    }
    
    ISLAMIC_HOLIDAYS_2026 = {
        (8, 25): "Prophet's Birthday (Mawlid)",
    }
    
    # Coptic Christmas Dates (Fixed, but treating as range for consistency if needed, though usually just 1 day)
    COPTIC_CHRISTMAS_DATES = {
        2024: (date(2024, 1, 7), date(2024, 1, 7)),
        2025: (date(2025, 1, 7), date(2025, 1, 7)),
        2026: (date(2026, 1, 7), date(2026, 1, 7)),
    }

    # Eid al-Fitr Dates (7 days starting day after Ramadan)
    EID_AL_FITR_DATES = {
        2024: (date(2024, 4, 9), date(2024, 4, 15)),
        2025: (date(2025, 3, 30), date(2025, 4, 5)),
        2026: (date(2026, 3, 20), date(2026, 3, 26)),
    }

    # Eid al-Adha Dates (7 days starting Day 1)
    EID_AL_ADHA_DATES = {
        2024: (date(2024, 6, 16), date(2024, 6, 22)),
        2025: (date(2025, 6, 6), date(2025, 6, 12)),
        2026: (date(2026, 5, 27), date(2026, 6, 2)),
    }
    
    # Ramadan approximate dates (start, end) for 2024-2026
    RAMADAN_DATES = {
        2024: (date(2024, 3, 10), date(2024, 4, 8)),
        2025: (date(2025, 2, 28), date(2025, 3, 29)),
        2026: (date(2026, 2, 17), date(2026, 3, 19)),
    }
    
    def __init__(self):
        """Initialize the holiday detector."""
        self._holiday_cache: Dict[int, Set[date]] = {}
    
    def _get_holidays_for_year(self, year: int) -> Set[date]:
        """Get all holidays for a specific year."""
        if year in self._holiday_cache:
            return self._holiday_cache[year]
        
        holidays = set()
        
        # Add fixed holidays
        for (month, day), name in self.FIXED_HOLIDAYS.items():
            try:
                holidays.add(date(year, month, day))
            except ValueError:
                pass  # Invalid date
        
        # Add Islamic holidays based on year
        islamic_holidays = {
            2024: self.ISLAMIC_HOLIDAYS_2024,
            2025: self.ISLAMIC_HOLIDAYS_2025,
            2026: self.ISLAMIC_HOLIDAYS_2026,
        }.get(year, self.ISLAMIC_HOLIDAYS_2025)  # Default to 2025 if not defined
        
        for (month, day), name in islamic_holidays.items():
            try:
                holidays.add(date(year, month, day))
            except ValueError:
                pass
        
        self._holiday_cache[year] = holidays
        return holidays
    
    def is_holiday(self, check_date: date) -> bool:
        """Check if a date is a holiday."""
        if isinstance(check_date, datetime):
            check_date = check_date.date()
        
        holidays = self._get_holidays_for_year(check_date.year)
        if check_date in holidays:
            return True
            
        # Check range-based holidays
        if self.is_eid_fitr(check_date) or self.is_eid_adha(check_date) or self.is_coptic_christmas(check_date):
            return True
            
        return False
    
    def is_ramadan(self, check_date: date) -> bool:
        """Check if a date falls within Ramadan."""
        if isinstance(check_date, datetime):
            check_date = check_date.date()
        
        ramadan = self.RAMADAN_DATES.get(check_date.year)
        if ramadan:
            start, end = ramadan
            return start <= check_date <= end
        return False

    def _is_in_range(self, check_date: date, ranges: Dict[int, tuple]) -> bool:
        """Helper to check if date is in a year-based range dictionary."""
        date_range = ranges.get(check_date.year)
        if date_range:
            start, end = date_range
            return start <= check_date <= end
        return False

    def is_eid_fitr(self, check_date: date) -> bool:
        if isinstance(check_date, datetime):
            check_date = check_date.date()
        return self._is_in_range(check_date, self.EID_AL_FITR_DATES)

    def is_eid_adha(self, check_date: date) -> bool:
        if isinstance(check_date, datetime):
            check_date = check_date.date()
        return self._is_in_range(check_date, self.EID_AL_ADHA_DATES)

    def is_coptic_christmas(self, check_date: date) -> bool:
        if isinstance(check_date, datetime):
            check_date = check_date.date()
        return self._is_in_range(check_date, self.COPTIC_CHRISTMAS_DATES)
    
    def is_western_christmas(self, check_date: date) -> bool:
        if isinstance(check_date, datetime):
            check_date = check_date.date()
        return check_date.month == 12 and check_date.day == 25
    
    def is_end_of_month(self, check_date: date) -> bool:
        """Check if date is near end of month (potential pay-day spike)."""
        if isinstance(check_date, datetime):
            check_date = check_date.date()
        return check_date.day >= 25
    
    def get_holiday_name(self, check_date: date) -> Optional[str]:
        """Get the name of the holiday if it is one."""
        if isinstance(check_date, datetime):
            check_date = check_date.date()
        
        # Check fixed holidays
        key = (check_date.month, check_date.day)
        if key in self.FIXED_HOLIDAYS:
            return self.FIXED_HOLIDAYS[key]
        
        # Check Islamic holidays
        year = check_date.year
        islamic_holidays = {
            2024: self.ISLAMIC_HOLIDAYS_2024,
            2025: self.ISLAMIC_HOLIDAYS_2025,
            2026: self.ISLAMIC_HOLIDAYS_2026,
        }.get(year, {})
        
        if key in islamic_holidays:
            return islamic_holidays[key]
            
        # Check range-based holidays
        if self.is_eid_fitr(check_date):
            return "Eid al-Fitr"
        if self.is_eid_adha(check_date):
            return "Eid al-Adha"
        if self.is_coptic_christmas(check_date):
            return "Coptic Christmas"
        
        return None
    
    def add_holiday_features(self, df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
        """
        Add holiday-related features to a DataFrame for forecasting.
        
        Args:
            df: DataFrame with a date column
            date_col: Name of the date column
            
        Returns:
            DataFrame with added holiday features
        """
        df = df.copy()
        
        # Ensure date column is proper type
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df[date_col] = pd.to_datetime(df[date_col])
        
        # Add features
        df['is_holiday'] = df[date_col].apply(lambda x: self.is_holiday(x.date() if hasattr(x, 'date') else x))
        df['is_ramadan'] = df[date_col].apply(lambda x: self.is_ramadan(x.date() if hasattr(x, 'date') else x))
        df['is_end_of_month'] = df[date_col].apply(lambda x: self.is_end_of_month(x.date() if hasattr(x, 'date') else x))
        df['is_weekend'] = df[date_col].dt.dayofweek.isin([4, 5])  # Friday-Saturday weekend in Egypt
        
        # Days until/since major events (useful for pre/post holiday effects)
        df['holiday_name'] = df[date_col].apply(lambda x: self.get_holiday_name(x.date() if hasattr(x, 'date') else x))
        
        return df
    
    def get_demand_multiplier(self, check_date: date) -> float:
        """
        Get demand adjustment multiplier for a specific date.
        
        Returns:
            Multiplier (1.0 = normal, >1 = higher demand, <1 = lower demand)
        """
        multiplier = 1.0
        
        # Ramadan: demand shifts (lower during day, higher at night - net effect varies)
        if self.is_ramadan(check_date):
            multiplier *= 0.85  # Generally lower during fasting
        
        # Eid holidays: high demand period
        holiday_name = self.get_holiday_name(check_date)
        if holiday_name and ('Eid' in str(holiday_name) or 'Christmas' in str(holiday_name)):
            multiplier *= 1.5  # Major celebrations
        elif holiday_name:
            multiplier *= 1.2  # Other holidays
        
        # End of month: salary period
        if self.is_end_of_month(check_date):
            multiplier *= 1.1
        
        return multiplier
    
    def get_upcoming_events(self, days_ahead: int = 14) -> List[Dict]:
        """Get list of upcoming holidays/events."""
        events = []
        today = date.today()
        
        for i in range(days_ahead):
            check_date = today + timedelta(days=i)
            holiday_name = self.get_holiday_name(check_date)
            
            if holiday_name:
                event_type = 'holiday'
                if 'Eid' in str(holiday_name):
                    event_type = 'eid'
                elif 'Christmas' in str(holiday_name):
                    event_type = 'christmas'
                    
                events.append({
                    'date': check_date.isoformat(),
                    'event': holiday_name,
                    'type': event_type,
                    'demand_multiplier': self.get_demand_multiplier(check_date)
                })
            elif self.is_ramadan(check_date) and (i == 0 or not self.is_ramadan(today + timedelta(days=i-1))):
                events.append({
                    'date': check_date.isoformat(),
                    'event': 'Ramadan',
                    'type': 'ramadan',
                    'demand_multiplier': 0.85
                })
            
            # Check for range-based events (Eid, Coptic Christmas)
            # Show if: 
            # 1. It is the start of the event (standard lookahead)
            # 2. OR if i==0 (today) and the event is active (so it shows up in the list even if it started yesterday)
            elif self.is_eid_fitr(check_date):
                 if i == 0 or not self.is_eid_fitr(today + timedelta(days=i-1)):
                    events.append({'date': check_date.isoformat(), 'event': 'Eid al-Fitr', 'type': 'eid', 'demand_multiplier': 1.5})
            elif self.is_eid_adha(check_date):
                if i == 0 or not self.is_eid_adha(today + timedelta(days=i-1)):
                    events.append({'date': check_date.isoformat(), 'event': 'Eid al-Adha', 'type': 'eid', 'demand_multiplier': 1.5})
            elif self.is_coptic_christmas(check_date):
                if i == 0 or not self.is_coptic_christmas(today + timedelta(days=i-1)):
                    events.append({'date': check_date.isoformat(), 'event': 'Coptic Christmas', 'type': 'christmas', 'demand_multiplier': 1.5})
                events.append({'date': check_date.isoformat(), 'event': 'Coptic Christmas', 'type': 'christmas', 'demand_multiplier': 1.5})
        
        return events


# Singleton instance for easy import
holiday_detector = HolidayDetector()


if __name__ == "__main__":
    detector = HolidayDetector()
    
    print("=== Holiday Detector Test ===\n")
    
    # Test specific dates
    test_dates = [
        date(2026, 2, 7),   # Today (roughly)
        date(2026, 1, 7),   # Coptic Christmas
        date(2025, 3, 30),  # Eid al-Fitr 2025
        date(2026, 3, 1),   # During Ramadan 2026
        date(2026, 3, 20),  # Eid al-Fitr 2026
    ]
    
    for d in test_dates:
        print(f"{d}: Holiday={detector.is_holiday(d)}, Ramadan={detector.is_ramadan(d)}, "
              f"Name={detector.get_holiday_name(d)}, Multiplier={detector.get_demand_multiplier(d):.2f}")
    
    print("\n--- Upcoming Events ---")
    for event in detector.get_upcoming_events(30):
        print(f"{event['date']}: {event['event']} (x{event['demand_multiplier']:.2f})")
