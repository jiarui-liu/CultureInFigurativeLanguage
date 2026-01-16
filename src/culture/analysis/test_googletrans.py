"""
Test script for googletrans functionality in entity_clustering.py
Tests the translate_entities function with Chinese entities.
"""

from entity_clustering import translate_entities


def test_translate_entities():
    """Test the translate_entities function with sample Chinese entities."""
    
    # Sample Chinese entities to test
    test_entities = [
        "孔子",
        "孟子",
        "老子",
        "庄子",
        "李白",
        "杜甫",
        "苏轼",
        "王羲之",
        "秦始皇",
        "汉武帝"
    ]
    
    print("Testing translate_entities function...")
    print(f"Input entities ({len(test_entities)}):")
    for entity in test_entities:
        print(f"  - {entity}")
    print()
    
    try:
        print("Translating entities from Chinese to English...")
        translated = translate_entities(test_entities, src='zh-cn', dest='en')
        
        print("\nTranslation results:")
        print("-" * 60)
        for chinese, english in zip(test_entities, translated):
            print(f"{chinese:15} -> {english}")
        print("-" * 60)
        
        print(f"\n✓ Successfully translated {len(translated)} entities")
        
        # Test the format used in plotting
        print("\nFormatted labels (as used in plot):")
        formatted_labels = [f"{chinese} + {english}" for chinese, english in zip(test_entities, translated)]
        for label in formatted_labels:
            print(f"  - {label}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Translation failed with error: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False


def test_single_entity():
    """Test translation of a single entity."""
    print("\n" + "=" * 60)
    print("Testing single entity translation...")
    print("=" * 60)
    
    test_entity = "孔子"
    try:
        translated = translate_entities([test_entity], src='zh-cn', dest='en')
        print(f"Input: {test_entity}")
        print(f"Output: {translated[0]}")
        print(f"Formatted: {test_entity} + {translated[0]}")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def test_empty_list():
    """Test translation with empty list."""
    print("\n" + "=" * 60)
    print("Testing empty list...")
    print("=" * 60)
    
    try:
        translated = translate_entities([], src='zh', dest='en')
        print(f"Empty list result: {translated}")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Googletrans Test Suite")
    print("=" * 60)
    print()
    
    # Run tests
    results = []
    
    results.append(("Multiple entities", test_translate_entities()))
    results.append(("Single entity", test_single_entity()))
    results.append(("Empty list", test_empty_list()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name:20} : {status}")
    
    all_passed = all(result for _, result in results)
    print("\n" + "=" * 60)
    if all_passed:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")
    print("=" * 60)

