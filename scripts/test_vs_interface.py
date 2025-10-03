#!/usr/bin/env python3
"""
Test the actual vector_store module interface as used by app.py
"""


def test_vector_store_import():
    """Test importing vector_store module like app.py does"""
    print("🔍 Testing vector_store module import...")

    try:
        import vector_store as vs_module

        print("✅ vector_store module imported successfully")

        # Check available methods
        methods = [attr for attr in dir(vs_module) if not attr.startswith("_")]
        print(f"Available methods: {methods}")

        # Test the specific methods used by app.py
        if hasattr(vs_module, "add_to_store"):
            print("✅ add_to_store method found")
        else:
            print("❌ add_to_store method NOT found")

        if hasattr(vs_module, "search"):
            print("✅ search method found")
        else:
            print("❌ search method NOT found")

        if hasattr(vs_module, "clear_uploads"):
            print("✅ clear_uploads method found")
        else:
            print("❌ clear_uploads method NOT found")

        return True

    except Exception as e:
        print(f"❌ vector_store import failed: {e}")
        return False


def test_vector_store_operations():
    """Test actual vector_store operations"""
    print("\n🔍 Testing vector_store operations...")

    try:
        import vector_store as vs_module

        # Test search operation
        print("Testing search...")
        try:
            # Try different parameter names
            if hasattr(vs_module, "search"):
                result = vs_module.search("test query", k=1)
                print(f"✅ search with k=1 successful: {type(result)}")
            else:
                print("❌ No search method")
        except Exception as e:
            print(f"❌ search failed: {e}")

        # Test add_to_store operation
        print("Testing add_to_store...")
        try:
            if hasattr(vs_module, "add_to_store"):
                # Use a timeout to catch hanging
                import threading

                result = [None]
                error = [None]

                def add_operation():
                    try:
                        result[0] = vs_module.add_to_store(
                            "test document content", tag="test"
                        )
                        print(f"✅ add_to_store successful: {result[0]}")
                    except Exception as e:
                        error[0] = e
                        print(f"❌ add_to_store failed: {e}")

                thread = threading.Thread(target=add_operation)
                thread.start()
                thread.join(timeout=15)  # 15 second timeout

                if thread.is_alive():
                    print("⚠️  add_to_store operation is hanging (timeout after 15s)")
                    return False
                elif error[0]:
                    print(f"❌ add_to_store error: {error[0]}")
                    return False
                else:
                    print("✅ add_to_store completed successfully")
                    return True
            else:
                print("❌ No add_to_store method")
                return False

        except Exception as e:
            print(f"❌ add_to_store test failed: {e}")
            return False

    except Exception as e:
        print(f"❌ vector_store operations test failed: {e}")
        return False


if __name__ == "__main__":
    print("🧪 Testing Vector Store Module Interface")
    print("=" * 45)

    # Test 1: Import
    import_success = test_vector_store_import()

    if import_success:
        # Test 2: Operations
        ops_success = test_vector_store_operations()

        if ops_success:
            print("\n🎯 All tests passed! Vector store should work properly.")
        else:
            print("\n⚠️  Import successful but operations have issues.")
    else:
        print("\n❌ Import failed - vector store not available.")
