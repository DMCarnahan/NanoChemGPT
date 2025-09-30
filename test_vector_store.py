#!/usr/bin/env python3
"""
Test the vector store functionality more thoroughly
"""

def test_vector_store_operations():
    """Test actual vector store operations that might hang"""
    print("🔍 Testing vector store operations...")
    
    try:
        import vector_store.uploads_vector as vs
        print("✅ Vector store imported")
        
        # Test 1: Basic search (should work)
        print("🔍 Testing search...")
        results = vs.search("test query", k=5)
        print(f"✅ Search completed: {len(results) if results else 0} results")
        
        # Test 2: Add content (this might hang)
        print("🔍 Testing add_to_store...")
        test_content = "This is a test document about nanoparticle synthesis."
        
        import time
        start_time = time.time()
        
        # Set a timeout for this operation
        import threading
        result = [None]
        error = [None]
        
        def add_operation():
            try:
                result[0] = vs.add_to_store("test.pdf", test_content, kind="pdf")
                print(f"✅ add_to_store completed: {result[0]}")
            except Exception as e:
                error[0] = e
                print(f"❌ add_to_store failed: {e}")
        
        thread = threading.Thread(target=add_operation)
        thread.start()
        thread.join(timeout=10)  # 10 second timeout
        
        if thread.is_alive():
            print("⚠️  add_to_store operation is hanging (timeout after 10s)")
            return False
        elif error[0]:
            print(f"❌ add_to_store error: {error[0]}")
            return False
        else:
            print("✅ add_to_store completed successfully")
            return True
            
    except Exception as e:
        print(f"❌ Vector store test failed: {e}")
        return False

def test_sentence_transformers():
    """Test sentence-transformers specifically"""
    print("\n🔍 Testing sentence-transformers...")
    try:
        import sentence_transformers
        print("✅ sentence-transformers imported")
        
        # Try to load a model (this might hang)
        import time
        start = time.time()
        
        model = sentence_transformers.SentenceTransformer('all-MiniLM-L6-v2')
        elapsed = time.time() - start
        print(f"✅ Model loaded in {elapsed:.1f}s")
        
        # Try encoding
        start = time.time()
        embeddings = model.encode(["test sentence"])
        elapsed = time.time() - start
        print(f"✅ Encoding completed in {elapsed:.1f}s")
        
        return True
    except Exception as e:
        print(f"❌ sentence-transformers test failed: {e}")
        return False

def test_dependencies():
    """Test various dependencies that might cause hanging"""
    print("\n🔍 Testing dependencies...")
    
    deps_to_test = [
        ("torch", "import torch"),
        ("numpy", "import numpy"),
        ("sentence_transformers", "import sentence_transformers"),
        ("transformers", "import transformers"),
        ("faiss", "import faiss"),
    ]
    
    for name, import_stmt in deps_to_test:
        try:
            exec(import_stmt)
            print(f"✅ {name} imported successfully")
        except Exception as e:
            print(f"❌ {name} failed: {e}")

if __name__ == "__main__":
    print("🧪 Testing Vector Store Operations")
    print("=" * 40)
    
    test_dependencies()
    test_sentence_transformers()
    test_vector_store_operations()