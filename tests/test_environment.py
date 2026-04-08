import pytest
from server.openenv_environment import TrustchainEnvironment
from models import TrustchainAction

def test_reset():
    env = TrustchainEnvironment()
    obs = env.reset(task_id="trustchain_easy")
    assert obs.claim != ""
    assert obs.difficulty == "easy"
    assert obs.done is False

def test_step():
    env = TrustchainEnvironment()
    obs = env.reset(task_id="trustchain_easy")
    
    action = TrustchainAction(decision="accept")
    next_obs = env.step(action)
    
    assert next_obs.reward in [0.0, 0.3, 0.5, 0.6, 1.0]
    assert env.state.step_count == 1

def test_reproducibility():
    env1 = TrustchainEnvironment()
    env2 = TrustchainEnvironment()
    
    obs1 = env1.reset(task_id="trustchain_easy", seed=42)
    obs2 = env2.reset(task_id="trustchain_easy", seed=42)
    
    assert obs1.claim == obs2.claim

def test_tasks_enumeration():
    env = TrustchainEnvironment()
    # Check that all 3 named tasks are accessible
    for task_id in ["trustchain_easy", "trustchain_medium", "trustchain_hard"]:
        obs = env.reset(task_id=task_id)
        assert obs.difficulty == task_id.split("_")[-1]
