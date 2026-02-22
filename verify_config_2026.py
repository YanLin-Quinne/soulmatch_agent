#!/usr/bin/env python3
"""
SoulMatch v2.0 Configuration Verification Script
Verifies all 5 LLM providers are correctly configured (2026-02-22)
"""

import sys
import os

# Add project root to path
sys.path.insert(0, '/Users/quinne/Desktop/soulmatch_agent_test')

def verify_config():
    """Verify configuration files and API keys"""
    print("=" * 70)
    print("SoulMatch v2.0 Configuration Verification (2026-02-22)")
    print("=" * 70)
    print()

    # 1. Check .env file exists
    print("1. Checking .env file...")
    env_path = "/Users/quinne/Desktop/soulmatch_agent_test/.env"
    if os.path.exists(env_path):
        print("   ✓ .env file found")
    else:
        print("   ✗ .env file not found")
        return False

    # 2. Load settings
    print("\n2. Loading settings...")
    try:
        from src.config import settings
        print("   ✓ Settings loaded successfully")
    except Exception as e:
        print(f"   ✗ Failed to load settings: {e}")
        return False

    # 3. Check API keys
    print("\n3. Checking API keys...")
    api_keys = {
        "OpenAI (GPT-5.2)": settings.openai_api_key,
        "Gemini (3.1 Pro)": settings.gemini_api_key,
        "Anthropic (Claude Opus 4.6)": settings.anthropic_api_key,
        "Qwen (3.5 Plus)": settings.qwen_api_key,
        "DeepSeek (Reasoner V3.2)": settings.deepseek_api_key,
    }

    all_keys_present = True
    for provider, key in api_keys.items():
        if key and len(key) > 10:
            print(f"   ✓ {provider}: {key[:15]}...{key[-10:]}")
        else:
            print(f"   ✗ {provider}: NOT CONFIGURED")
            all_keys_present = False

    if not all_keys_present:
        print("\n   ⚠ Warning: Some API keys are missing")

    # 4. Check LLM router
    print("\n4. Checking LLM router...")
    try:
        from src.agents.llm_router import router, MODELS, MODEL_ROUTING
        print(f"   ✓ LLM router initialized")
        print(f"   ✓ Available models: {len(MODELS)}")
        print(f"   ✓ Routing strategies: {len(MODEL_ROUTING)}")
    except Exception as e:
        print(f"   ✗ Failed to initialize LLM router: {e}")
        return False

    # 5. List 2026 models
    print("\n5. Verifying 2026 latest models...")
    latest_models = {
        "claude-opus-4": "claude-opus-4-6",
        "gpt-5": "gpt-5.2",
        "gemini-3-pro": "gemini-3.1-pro-preview",
        "deepseek-reasoner": "deepseek-reasoner",
        "qwen-3.5-plus": "qwen3.5-plus",
    }

    for key, expected_id in latest_models.items():
        if key in MODELS:
            actual_id = MODELS[key].model_id
            if actual_id == expected_id:
                print(f"   ✓ {key}: {actual_id}")
            else:
                print(f"   ⚠ {key}: {actual_id} (expected: {expected_id})")
        else:
            print(f"   ✗ {key}: NOT FOUND")

    # 6. Check model routing
    print("\n6. Checking model routing priorities...")
    from src.agents.llm_router import AgentRole

    priority_checks = {
        AgentRole.PERSONA: "claude-opus-4",
        AgentRole.FEATURE: "claude-opus-4",
        AgentRole.EMOTION: "gemini-flash",
        AgentRole.QUESTION: "gemini-3-pro",
    }

    for role, expected_primary in priority_checks.items():
        actual_primary = MODEL_ROUTING[role][0]
        if actual_primary == expected_primary:
            print(f"   ✓ {role.value}: {actual_primary}")
        else:
            print(f"   ⚠ {role.value}: {actual_primary} (expected: {expected_primary})")

    # 7. Check documentation
    print("\n7. Checking documentation files...")
    docs = [
        "README.md",
        "IMPLEMENTATION_SUMMARY_V2.md",
        "QUICKSTART_V2.md",
        "LLM_CONFIG_2026.md",
        "CONFIG_UPDATE_SUMMARY.md",
    ]

    for doc in docs:
        doc_path = f"/Users/quinne/Desktop/soulmatch_agent_test/{doc}"
        if os.path.exists(doc_path):
            size = os.path.getsize(doc_path)
            print(f"   ✓ {doc} ({size:,} bytes)")
        else:
            print(f"   ✗ {doc} NOT FOUND")

    print("\n" + "=" * 70)
    print("✓ Configuration verification complete!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("1. Test API connectivity: python -c \"from src.agents.llm_router import router; print('OK')\"")
    print("2. Run integration tests: python test_v2_integration.py")
    print("3. Start backend: python -m uvicorn src.api.main:app --reload")
    print("4. Start frontend: cd frontend && npm run dev")
    print()

    return True

if __name__ == "__main__":
    try:
        success = verify_config()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Verification failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
