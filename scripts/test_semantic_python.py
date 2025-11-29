#!/usr/bin/env python3
"""
Python reference test for MiniLM semantic similarity.
This script tests the same semantic queries as the C test to compare results.
"""

try:
    import torch
    import numpy as np
    from transformers import AutoTokenizer, AutoModel
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError as e:
    print("Error: Required packages not installed.")
    print("Please install with: pip install torch transformers scikit-learn")
    print(f"Missing package: {e}")
    exit(1)

# Model name - should match what was used to create bert_weights.tbf
# Based on dump_tbf1.py, it uses sentence-transformers/all-MiniLM-L6-v2
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def load_model():
    """Load the MiniLM model and tokenizer."""
    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()  # Set to evaluation mode
    print("Model loaded successfully")
    return model, tokenizer

def mean_pooling(model_output, attention_mask):
    """
    Mean pooling - average token embeddings, excluding padding tokens.
    This matches the C implementation's nn_mean_pooling.
    """
    # First element of model_output contains all token embeddings
    token_embeddings = model_output[0]  # Shape: [batch_size, seq_len, hidden_size]
    
    # Expand attention mask to match embedding dimensions
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    
    # Sum embeddings, excluding padding tokens
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    
    # Count non-padding tokens
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    # Average
    mean_embeddings = sum_embeddings / sum_mask
    return mean_embeddings

def embed_text(model, tokenizer, text, max_length=256):
    """
    Embed a text string using the model.
    Returns normalized embedding vector.
    """
    # Tokenize
    encoded_input = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    
    # Get attention mask (1 for real tokens, 0 for padding)
    attention_mask = encoded_input['attention_mask']
    
    # Generate embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)
    
    # Mean pooling (average all non-padding tokens)
    sentence_embeddings = mean_pooling(model_output, attention_mask)
    
    # Normalize (L2 normalization)
    sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
    
    return sentence_embeddings.squeeze(0).numpy(), attention_mask.squeeze(0).numpy()

def cosine_sim(a, b):
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def test_semantic_queries():
    """Test semantic similarity queries matching the C test."""
    model, tokenizer = load_model()
    
    # Test cases matching minilm_test.c
    test_cases = [
        # Capital cities
        {"query": "what's the capital of germany?", "expected": "berlin", "index": 2},
        {"query": "what's the capital of france?", "expected": "paris", "index": 0},
        {"query": "what's the capital of spain?", "expected": "madrid", "index": 3},
        {"query": "what's the capital of italy?", "expected": "rome", "index": 4},
        {"query": "what's the capital of england?", "expected": "london", "index": 1},
        # Additional semantic queries
        {"query": "the capital city of france", "expected": "paris", "index": 0},
        {"query": "germany's capital", "expected": "berlin", "index": 2},
        {"query": "capital of spain", "expected": "madrid", "index": 3},
        {"query": "italy capital city", "expected": "rome", "index": 4},
        {"query": "london is the capital of", "expected": "london", "index": 1},
    ]
    
    # City names (matching C test order)
    cities = ["paris", "london", "berlin", "madrid", "rome"]
    
    print("\n=== Embedding city names ===")
    city_embeddings = {}
    for city in cities:
        emb, mask = embed_text(model, tokenizer, city, max_length=256)
        city_embeddings[city] = emb
        non_padding = np.sum(mask)
        print(f"  {city}: {len(emb)} dims, {non_padding} non-padding tokens")
    
    print("\n=== Testing semantic queries ===")
    passed = 0
    total = len(test_cases)
    
    for test_case in test_cases:
        query = test_case["query"]
        expected = test_case["expected"]
        expected_idx = test_case["index"]
        
        # Embed query
        query_emb, query_mask = embed_text(model, tokenizer, query, max_length=256)
        query_non_padding = np.sum(query_mask)
        print(f"\nQuery: '{query}' ({query_non_padding} non-padding tokens)")
        
        # Compute similarities
        similarities = {}
        for city in cities:
            sim = cosine_sim(query_emb, city_embeddings[city])
            similarities[city] = sim
        
        # Find best match
        best_city = max(similarities, key=similarities.get)
        best_sim = similarities[best_city]
        
        # Print all similarities
        print("Similarities:", end=" ")
        for city in cities:
            print(f"{city}={similarities[city]:.6f}", end=" ")
        print()
        
        # Check result
        is_correct = (best_city == expected)
        status = "✓" if is_correct else "✗"
        print(f"Answer: '{best_city}' (expected: '{expected}', similarity: {best_sim:.6f}) {status}")
        
        if is_correct:
            passed += 1
        else:
            print(f"  ⚠ Warning: Expected '{expected}' but got '{best_city}'")
    
    print(f"\n=== Results: {passed}/{total} passed ===")
    return passed == total

def test_layer_outputs():
    """Test layer-by-layer outputs to compare with C implementation."""
    model, tokenizer = load_model()
    
    test_text = "berlin"
    print(f"\n=== Testing layer outputs for: '{test_text}' ===")
    
    # Tokenize
    encoded_input = tokenizer(
        test_text,
        padding='max_length',
        truncation=True,
        max_length=256,
        return_tensors='pt'
    )
    
    attention_mask = encoded_input['attention_mask']
    input_ids = encoded_input['input_ids']
    
    # Convert attention mask to float for newer transformers versions
    attention_mask_float = attention_mask.float()
    
    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Attention mask shape: {attention_mask.shape}")
    print(f"Non-padding tokens: {attention_mask.sum().item()}")
    
    with torch.no_grad():
        # Get embeddings layer output
        embeddings = model.embeddings(input_ids)
        print(f"\nEmbeddings output shape: {embeddings.shape}")
        print(f"Embeddings sum: {embeddings.sum().item():.6f}")
        print(f"Embeddings mean: {embeddings.mean().item():.6f}")
        
        # Process through encoder layers
        hidden_states = embeddings
        for i, layer in enumerate(model.encoder.layer):
            layer_output = layer(hidden_states, attention_mask=attention_mask_float)
            hidden_states = layer_output[0]
            print(f"\nEncoder layer {i} output shape: {hidden_states.shape}")
            print(f"Encoder layer {i} sum: {hidden_states.sum().item():.6f}")
            print(f"Encoder layer {i} mean: {hidden_states.mean().item():.6f}")
        
        # Mean pooling
        pooled = mean_pooling((hidden_states,), attention_mask)
        print(f"\nMean pooled shape: {pooled.shape}")
        print(f"Mean pooled sum: {pooled.sum().item():.6f}")
        
        # Normalize
        normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
        print(f"\nNormalized shape: {normalized.shape}")
        print(f"Normalized sum: {normalized.sum().item():.6f}")
        print(f"Normalized norm: {torch.norm(normalized).item():.6f}")

if __name__ == "__main__":
    print("=" * 60)
    print("Python MiniLM Semantic Similarity Test")
    print("=" * 60)
    print("\nNote: This script requires torch and transformers packages.")
    print("Install with: pip install torch transformers scikit-learn")
    print("=" * 60)
    
    try:
        # Test layer outputs first
        test_layer_outputs()
        
        # Then test semantic queries
        success = test_semantic_queries()
        
        if success:
            print("\n✓ All tests passed!")
            exit(0)
        else:
            print("\n✗ Some tests failed")
            exit(1)
            
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

