#!/usr/bin/env python3
"""
Unit test for Voyage AI integration
Tests actual API calls to verify implementation
"""

import os
import sys
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_voyage_api():
    """Test Voyage AI API integration with real API call"""

    print("=" * 60)
    print("VOYAGE AI INTEGRATION TEST")
    print("=" * 60)

    # Step 1: Check environment
    print("\n1. Checking environment...")
    api_key = os.getenv("VOYAGE_API_KEY")
    if not api_key:
        print("❌ VOYAGE_API_KEY not set")
        return False
    print(f"✓ VOYAGE_API_KEY is set (length: {len(api_key)})")

    # Step 2: Import voyageai package
    print("\n2. Importing voyageai package...")
    try:
        import voyageai
        print(f"✓ voyageai version: {voyageai.__version__}")
    except ImportError as e:
        print(f"❌ Failed to import voyageai: {e}")
        return False

    # Step 3: Test direct API call
    print("\n3. Testing direct Voyage AI API call...")
    try:
        client = voyageai.Client(api_key=api_key)
        print("✓ Client initialized")

        # Test with simple English and Chinese words
        test_texts = ["hello", "world", "你好", "世界"]
        print(f"  Testing with {len(test_texts)} words: {test_texts}")

        result = client.embed(
            test_texts,
            model="voyage-3",
            input_type="document"
        )

        print(f"✓ API call successful")
        print(f"  Response type: {type(result)}")
        print(f"  Has embeddings: {hasattr(result, 'embeddings')}")

        if hasattr(result, 'embeddings'):
            embeddings = result.embeddings
            print(f"  Embeddings count: {len(embeddings)}")
            if embeddings:
                first_embedding = embeddings[0]
                print(f"  First embedding type: {type(first_embedding)}")
                print(f"  First embedding length: {len(first_embedding)}")
                print(f"  First embedding sample: {first_embedding[:5]}...")

                # Convert to numpy array
                embeddings_array = np.array(embeddings)
                print(f"  Numpy array shape: {embeddings_array.shape}")
                print(f"  Numpy array dtype: {embeddings_array.dtype}")
                print(f"  Range: [{embeddings_array.min():.4f}, {embeddings_array.max():.4f}]")
        else:
            print("❌ Response missing 'embeddings' attribute")
            print(f"  Response attributes: {dir(result)}")
            return False

    except Exception as e:
        error_str = str(e)
        if "payment method" in error_str.lower() or "rate limit" in error_str.lower():
            print(f"⚠️  Rate limit warning (this is OK for testing): {error_str[:200]}...")
            print("  Waiting 20 seconds and retrying...")
            import time
            time.sleep(20)

            # Retry
            try:
                result = client.embed(
                    test_texts,
                    model="voyage-3",
                    input_type="document"
                )
                print(f"✓ API call successful on retry")
            except Exception as retry_error:
                print(f"❌ API call failed on retry: {retry_error}")
                import traceback
                traceback.print_exc()
                return False
        else:
            print(f"❌ Direct API call failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    # Step 4: Test VoyageModel class
    print("\n4. Testing VoyageModel class...")
    try:
        # Suppress streamlit output during test
        import streamlit as st
        from io import StringIO
        from unittest.mock import patch

        # Mock streamlit functions to suppress output
        with patch('streamlit.info'), patch('streamlit.success'), patch('streamlit.error'):
            from models.voyage_model import VoyageModel

            print("  Creating VoyageModel instance...")
            model = VoyageModel(
                model_path="voyage-3",
                model_name="Voyage-3 (Test)"
            )
            print("✓ VoyageModel initialized")

            print("  Testing get_embeddings...")
            embeddings = model.get_embeddings(test_texts, debug_flag=False)

            print(f"✓ get_embeddings returned")
            print(f"  Type: {type(embeddings)}")
            print(f"  Shape: {embeddings.shape}")
            print(f"  Dtype: {embeddings.dtype}")
            print(f"  Range: [{embeddings.min():.4f}, {embeddings.max():.4f}]")

            # Verify results
            assert embeddings.shape[0] == len(test_texts), f"Expected {len(test_texts)} embeddings, got {embeddings.shape[0]}"
            assert embeddings.shape[1] == 1024, f"Expected 1024 dimensions for voyage-3, got {embeddings.shape[1]}"
            assert not np.isnan(embeddings).any(), "Embeddings contain NaN values"
            assert not np.isinf(embeddings).any(), "Embeddings contain Inf values"

            print("✓ All assertions passed")

    except Exception as e:
        print(f"❌ VoyageModel test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 5: Test with multilingual content
    print("\n5. Testing multilingual capability...")
    try:
        with patch('streamlit.info'), patch('streamlit.success'), patch('streamlit.error'), patch('streamlit.expander'):
            multilingual_texts = [
                "English word",
                "中文词语",
                "Français mot",
                "Español palabra",
                "Deutsch Wort"
            ]

            embeddings = model.get_embeddings(multilingual_texts, debug_flag=False)
            print(f"✓ Multilingual embeddings generated")
            print(f"  Shape: {embeddings.shape}")
            print(f"  Range: [{embeddings.min():.4f}, {embeddings.max():.4f}]")

    except Exception as e:
        print(f"❌ Multilingual test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_voyage_api()
    sys.exit(0 if success else 1)
