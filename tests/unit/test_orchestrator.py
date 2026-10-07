"""Unit tests for Orchestrator stale-state clearing (issue #15).

Covers two independent leaks that both have to be fixed for a second
review of the same profile to reflect fresh tool results:

1. ``ContextManager`` memoizing tool results across separate ``run()``
   calls on the same ``Orchestrator`` instance.
2. ``SessionStore`` merging new results into whatever stale state was
   already persisted for a ``profile_id``, instead of clearing it first.
"""

from agent.memory.context_manager import ContextManager
from agent.memory.session_store import SessionStore
from agent.orchestrator import Orchestrator


class FakeResult:
    def __init__(self, data):
        self.data = data


class FakeRedis:
    """Minimal in-memory stand-in for redis.Redis backing get/setex/delete."""

    def __init__(self):
        self.store: dict[str, str] = {}

    def get(self, key):
        return self.store.get(key)

    def setex(self, key, ttl_seconds, value):
        self.store[key] = value

    def delete(self, key):
        self.store.pop(key, None)


class FakeTool:
    """A tool whose result changes on every call, the way a live GitHub
    API response could change between two reviews."""

    def __init__(self, name: str):
        self.name = name
        self.call_count = 0

    def execute(self, tool_input):
        self.call_count += 1
        return FakeResult({"call_number": self.call_count, "stars": 10 * self.call_count})


def make_profile_data(with_readme: bool = False) -> dict:
    data = {
        "github_username": "someuser",
        "projects": [{"github_repo": "some-repo"}],
    }
    if with_readme:
        data["readme_content"] = "# Some README"
    return data


class TestContextManagerReset:
    """ContextManager half of the fix."""

    def test_reset_clears_cached_results(self):
        cm = ContextManager()
        cm.store_tool_result("github_tool", "somehash", FakeResult({"x": 1}))
        assert cm.get_all_results() != {}

        cm.reset()

        assert cm.get_all_results() == {}
        assert cm.get_tool_result("github_tool", "somehash") is None

    def test_orchestrator_run_does_not_reuse_cache_across_runs(self):
        tool = FakeTool("github_tool")
        orchestrator = Orchestrator(tools={"github_tool": tool})
        profile_data = make_profile_data()

        result1 = orchestrator.run("profile-1", profile_data)
        result2 = orchestrator.run("profile-1", profile_data)

        assert result1["tool_results"]["github_tool"] == {"call_number": 1, "stars": 10}
        assert result2["tool_results"]["github_tool"] == {"call_number": 2, "stars": 20}
        assert tool.call_count == 2


class TestSessionStoreNotRetainedAcrossRuns:
    """SessionStore half of the fix."""

    def test_stale_keys_from_a_prior_review_are_not_retained(self):
        fake_redis = FakeRedis()
        session_store = SessionStore(fake_redis)  # type: ignore[arg-type]
        tools = {
            "github_tool": FakeTool("github_tool"),
            "readme_scorer": FakeTool("readme_scorer"),
            "market_analyzer": FakeTool("market_analyzer"),
        }
        orchestrator = Orchestrator(tools=tools, session_store=session_store)

        # Review 1 includes a readme, so its plan (and session state)
        # has a "readme_scorer" key that review 2 won't produce.
        orchestrator.run("profile-1", make_profile_data(with_readme=True))
        stored_after_first = session_store.get("profile-1")
        assert stored_after_first
        assert "readme_scorer" in stored_after_first

        orchestrator.run("profile-1", make_profile_data(with_readme=False))
        stored_after_second = session_store.get("profile-1")

        assert stored_after_second
        assert "readme_scorer" not in stored_after_second
        assert "github_tool" in stored_after_second
