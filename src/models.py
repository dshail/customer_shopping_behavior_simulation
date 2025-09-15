
"""
Customer Shopping Behavior Simulation - Core Data Models
========================================================

This module contains the core data models and classes for the customer
shopping behavior simulation system.

Author: dshail
Date: August 2025
"""

import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import uuid


class ShoppingFrequency(Enum):
    """Enumeration for customer shopping frequencies"""
    DAILY = "daily"
    ALTERNATE = "alternate" 
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class LocationType(Enum):
    """Enumeration for customer location types"""
    URBAN = "urban"
    SUBURBAN = "suburban"
    RURAL = "rural"
    MIXED = "mixed"


@dataclass
class Demographics:
    """Customer demographic information"""
    age: int
    income: int
    location_type: str
    education: str
    family_status: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class BasketItem:
    """Individual item in shopping basket"""
    category: str
    quantity: int
    price_per_item: float
    total_price: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class ShoppingBasket:
    """Complete shopping basket for a transaction"""
    items: List[BasketItem] = field(default_factory=list)
    total_amount: float = 0.0
    num_items: int = 0

    def add_item(self, item: BasketItem) -> None:
        """Add item to basket"""
        self.items.append(item)
        self.total_amount += item.total_price
        self.num_items += 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'items': [item.to_dict() for item in self.items],
            'total_amount': round(self.total_amount, 2),
            'num_items': self.num_items
        }


@dataclass
class Customer:
    """Individual customer profile"""
    customer_id: str
    name: str
    persona_type: str
    demographics: Demographics
    contact_info: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'customer_id': self.customer_id,
            'name': self.name,
            'persona_type': self.persona_type,
            **self.demographics.to_dict(),
            **self.contact_info
        }


@dataclass
class Transaction:
    """Complete transaction record"""
    transaction_id: str
    customer_id: str
    customer_name: str
    persona_type: str
    timestamp: datetime
    basket: ShoppingBasket
    context: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for export"""
        return {
            'transaction_id': self.transaction_id,
            'date': self.timestamp.strftime('%Y-%m-%d'),
            'timestamp': self.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'customer_id': self.customer_id,
            'customer_name': self.customer_name,
            'persona_type': self.persona_type,
            'total_amount': self.basket.total_amount,
            'num_items': self.basket.num_items,
            'items_json': str(self.basket.to_dict()['items']),
            **self.context
        }


@dataclass  
class PersonaConfig:
    """Configuration for a customer persona"""
    name: str
    frequency: str
    preferred_times: List[str]
    demographics: Dict[str, Any]
    basket_profile: Dict[str, Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class SimulationConfig:
    """Configuration for simulation parameters"""
    simulation_days: int = 30
    customers_per_persona: int = 1000
    start_date: str = "2024-01-01"
    random_seed: int = 42
    festival_dates: List[str] = field(default_factory=list)
    seasonal_multipliers: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


class PerformanceMetrics:
    """Track simulation performance metrics"""

    def __init__(self):
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.customers_generated: int = 0
        self.transactions_generated: int = 0
        self.total_revenue: float = 0.0
        self.errors_encountered: int = 0

    def start_timer(self) -> None:
        """Start performance timer"""
        self.start_time = datetime.now()

    def stop_timer(self) -> None:
        """Stop performance timer"""
        self.end_time = datetime.now()

    def get_duration(self) -> Optional[timedelta]:
        """Get total execution duration"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None

    def add_transaction(self, amount: float) -> None:
        """Add transaction to metrics"""
        self.transactions_generated += 1
        self.total_revenue += amount

    def add_customer(self) -> None:
        """Add customer to metrics"""
        self.customers_generated += 1

    def add_error(self) -> None:
        """Add error to metrics"""
        self.errors_encountered += 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        duration = self.get_duration()
        return {
            'execution_time_seconds': duration.total_seconds() if duration else 0,
            'customers_generated': self.customers_generated,
            'transactions_generated': self.transactions_generated,
            'total_revenue': round(self.total_revenue, 2),
            'avg_transaction_value': (
                round(self.total_revenue / self.transactions_generated, 2) 
                if self.transactions_generated > 0 else 0
            ),
            'transactions_per_second': (
                round(self.transactions_generated / duration.total_seconds(), 2)
                if duration and duration.total_seconds() > 0 else 0
            ),
            'errors_encountered': self.errors_encountered
        }


# Utility functions for data models
def generate_transaction_id() -> str:
    """Generate unique transaction ID"""
    return f"TXN_{datetime.now().strftime('%Y%m%d')}_{str(uuid.uuid4())[:8].upper()}"


def generate_customer_id(persona_name: str, index: int) -> str:
    """Generate customer ID based on persona"""
    clean_name = persona_name.lower().replace(' ', '_').replace('-', '_')
    return f"{clean_name}_{index:04d}"


def parse_time_slot(time_slot: str) -> Tuple[datetime.time, datetime.time]:
    """Parse time slot string into start and end times"""
    try:
        # Handle different types of dashes that might appear in time slots
        dash_chars = ['–', '—', '-', '−', '‒', '―', 'â€"']  # Added the corrupted encoding
        start_str, end_str = None, None
        
        for dash in dash_chars:
            if dash in time_slot:
                parts = time_slot.split(dash, 1)
                if len(parts) == 2:
                    start_str, end_str = parts
                    break
        
        if start_str is None or end_str is None:
            raise ValueError("No valid dash separator found")
            
        start_time = datetime.strptime(start_str.strip(), "%I:%M%p").time()
        end_time = datetime.strptime(end_str.strip(), "%I:%M%p").time()
        return start_time, end_time
    except ValueError as e:
        logging.error(f"Failed to parse time slot '{time_slot}': {e}")
        # Return default time slot
        return datetime.strptime("9:00am", "%I:%M%p").time(), datetime.strptime("10:00am", "%I:%M%p").time()
