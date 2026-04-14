#!/usr/bin/env python3
"""Test script to verify type normalization and validation."""

import sys
from src.constrained_decoder import ConstrainedDecoder

def test_normalize_type():
    """Test the type normalization method."""
    print("🧪 Testing type normalization...\n")
    
    test_cases = [
        ("number", "number"),
        ("integer", "number"),
        ("float", "number"),
        ("boolean", "boolean"),
        ("bool", "boolean"),
        ("string", "string"),
    ]
    
    for input_type, expected in test_cases:
        result = ConstrainedDecoder._normalize_type(input_type)
        status = "✓" if result == expected else "✗"
        print(f"{status} '{input_type}' → '{result}' (expected: '{expected}')")
    
    print("\n🧪 Testing invalid types (should raise error)...\n")
    
    invalid_types = ["numbetdfsfr", "bool_int", "foo", "123", ""]
    for invalid_type in invalid_types:
        try:
            ConstrainedDecoder._normalize_type(invalid_type)
            print(f"✗ '{invalid_type}' should have raised ValueError but didn't")
        except ValueError as e:
            print(f"✓ '{invalid_type}' correctly raised error: {e}")

if __name__ == "__main__":
    test_normalize_type()
    print("\n✅ All type validation tests completed!")
