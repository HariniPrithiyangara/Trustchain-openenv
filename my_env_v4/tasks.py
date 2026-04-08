
"""Task definitions for Backend API Automation."""

TASKS = {
    "easy": {
        "description": "Basic Query Handling: Fetch user email from user_db.",
        "target_resource": "user_db",
        "expected_sequence": ["fetch_data"],
        "initial_status": "waiting_for_fetch",
        "context": {"user_id": "usr_99", "table": "users"}
    },
    "medium": {
        "description": "Multi-step Workflow: Fetch, validate, and return user profile.",
        "target_resource": "profile_api",
        "expected_sequence": ["fetch_data", "validate_input", "return_response"],
        "initial_status": "workflow_start",
        "context": {"api_endpoint": "/v1/profile", "schema": "strict"}
    },
    "hard": {
        "description": "Error Recovery: Fetch data, detect broken payload, fix it, and complete workflow.",
        "target_resource": "legacy_service",
        "expected_sequence": ["fetch_data", "validate_input", "fix_error", "return_response"],
        "initial_status": "recovery_mode",
        "context": {"service": "legacy_v0", "issue": "malformed_json"}
    }
}
