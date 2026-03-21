# src/utils/exceptions.py
"""Custom exceptions"""

class DataValidationError(Exception):
    """Raised when data fails validation"""
    pass

class APIClientError(Exception):
    """Raised when API request fails"""
    pass
