"""
GitHub API Tools

Tools for searching and retrieving GitHub issues.
Used to find similar issues that might help with diagnosis.

Key Concept: LangChain Tools
    Tools are functions that an LLM can call. They have:
    - A name (used by the LLM to reference the tool)
    - A description (helps the LLM decide when to use it)
    - Input schema (what parameters the function accepts)

    In our case, we use tools directly (not via LLM tool-calling),
    but we could easily add agent-style tool use later.
"""

from typing import Optional

import httpx

from src.config import config
from src.state import GitHubIssue


# =============================================================================
# GitHub API Client
# =============================================================================

GITHUB_API_BASE = "https://api.github.com"


def _get_headers() -> dict[str, str]:
    """Get HTTP headers for GitHub API requests."""
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    # Add authentication if token is available
    if config.github_token:
        headers["Authorization"] = f"Bearer {config.github_token}"

    return headers


# =============================================================================
# Search Functions
# =============================================================================


def search_github_issues(
    query: str,
    repo: Optional[str] = None,
    state: str = "all",
    max_results: int = 5,
) -> list[GitHubIssue]:
    """
    Search GitHub issues for a repository.

    Uses GitHub's search API to find issues matching the query.
    Results are ordered by relevance.

    Args:
        query: Search query (e.g., "event loop already running")
        repo: Repository in format 'owner/repo' (defaults to config)
        state: Issue state filter ('open', 'closed', 'all')
        max_results: Maximum number of results to return

    Returns:
        List of GitHubIssue objects

    Example:
        issues = search_github_issues(
            query="RuntimeError event loop",
            repo="langchain-ai/langgraph"
        )
        for issue in issues:
            print(f"#{issue['number']}: {issue['title']}")
    """
    repo = repo or config.default_repo

    # Build the search query
    # GitHub search syntax: "query repo:owner/repo is:issue"
    search_query = f"{query} repo:{repo} is:issue"
    if state != "all":
        search_query += f" state:{state}"

    try:
        response = httpx.get(
            f"{GITHUB_API_BASE}/search/issues",
            headers=_get_headers(),
            params={
                "q": search_query,
                "per_page": max_results,
                "sort": "relevance",
            },
            timeout=10.0,
        )
        response.raise_for_status()

    except httpx.HTTPError as e:
        # Log error and return empty list
        print(f"GitHub API error: {e}")
        return []

    data = response.json()
    items = data.get("items", [])

    # Convert to our GitHubIssue format
    issues: list[GitHubIssue] = []
    for i, item in enumerate(items):
        issues.append(
            {
                "number": item["number"],
                "title": item["title"],
                "url": item["html_url"],
                "state": item["state"],
                # Relevance score: higher rank = higher score
                "relevance_score": 1.0 - (i * 0.1),
                "summary": (item.get("body") or "")[:300],
            }
        )

    return issues


def get_issue_details(
    repo: str,
    issue_number: int,
) -> Optional[dict]:
    """
    Get detailed information about a specific GitHub issue.

    Retrieves the full issue body and top comments.
    Useful for getting more context about a related issue.

    Args:
        repo: Repository in format 'owner/repo'
        issue_number: The issue number

    Returns:
        Issue details dict, or None if not found

    Example:
        details = get_issue_details("langchain-ai/langgraph", 123)
        if details:
            print(details["body"])
    """
    try:
        # Get the issue
        response = httpx.get(
            f"{GITHUB_API_BASE}/repos/{repo}/issues/{issue_number}",
            headers=_get_headers(),
            timeout=10.0,
        )
        response.raise_for_status()
        issue = response.json()

        # Get comments
        comments_response = httpx.get(
            f"{GITHUB_API_BASE}/repos/{repo}/issues/{issue_number}/comments",
            headers=_get_headers(),
            params={"per_page": 5},
            timeout=10.0,
        )
        comments = []
        if comments_response.status_code == 200:
            comments = comments_response.json()

        return {
            "number": issue["number"],
            "title": issue["title"],
            "state": issue["state"],
            "body": issue.get("body", ""),
            "url": issue["html_url"],
            "comments": [
                {
                    "author": c.get("user", {}).get("login", "unknown"),
                    "body": c.get("body", ""),
                }
                for c in comments
            ],
            "labels": [label["name"] for label in issue.get("labels", [])],
            "created_at": issue.get("created_at"),
            "closed_at": issue.get("closed_at"),
        }

    except httpx.HTTPError as e:
        print(f"GitHub API error: {e}")
        return None


# =============================================================================
# Library Detection
# =============================================================================

# Patterns to detect which library is involved
LIBRARY_PATTERNS = {
    "langgraph": [
        "langgraph",
        "StateGraph",
        "CompiledGraph",
        "ToolNode",
        "tool_node",
        "pregel",
        "MemorySaver",
        "SqliteSaver",
        "PostgresSaver",
        "checkpoint",
        "create_agent",
        "create_react_agent",
        "MessagesState",
        "AgentState",
    ],
    "langchain": [
        "langchain",
        "langchain_core",
        "langchain_community",
        "langchain_openai",
        "ChatOpenAI",
        "ChatAnthropic",
        "ChatGoogleGenerativeAI",
        "BaseChatModel",
        "BaseRetriever",
        "RunnableSequence",
        "LCEL",
        "StructuredTool",
        "@tool",
        "BaseTool",
        "AgentExecutor",
    ],
    "langsmith": [
        "langsmith",
        "LangSmith",
        "tracing",
        "LANGSMITH_API_KEY",
        "LANGSMITH_PROJECT",
        "run_tree",
        "trace",
    ],
}


def detect_libraries(text: str) -> dict:
    """
    Detect which LangChain ecosystem libraries are mentioned in the text.

    Returns a dict with:
    - primary: The main library involved
    - all_libraries: List of all detected libraries
    - components: Specific components mentioned
    - confidence: How confident we are in the detection
    """
    text_lower = text.lower()
    detected = {}
    components = []

    for library, patterns in LIBRARY_PATTERNS.items():
        count = 0
        for pattern in patterns:
            if pattern.lower() in text_lower or pattern in text:
                count += 1
                components.append(pattern)
        if count > 0:
            detected[library] = count

    if not detected:
        return {
            "primary": "unknown",
            "all_libraries": [],
            "components": [],
            "confidence": 0.0,
        }

    # Primary is the one with most matches
    primary = max(detected, key=detected.get)
    total_matches = sum(detected.values())
    confidence = min(1.0, total_matches / 5)  # Saturates at 5 matches

    return {
        "primary": primary,
        "all_libraries": list(detected.keys()),
        "components": list(set(components)),
        "confidence": confidence,
    }


def get_repo_for_library(library: str) -> str:
    """Map library name to GitHub repository."""
    mapping = {
        "langgraph": "langchain-ai/langgraph",
        "langchain": "langchain-ai/langchain",
        "langsmith": "langchain-ai/langsmith-sdk",
    }
    return mapping.get(library, config.default_repo)


# =============================================================================
# Search Node (for use in LangGraph)
# =============================================================================


def github_search_node(state: dict) -> dict:
    """
    LangGraph node that searches GitHub for related issues.

    Uses the bug report to build search queries and finds similar issues.
    Searches across multiple LangChain ecosystem repositories based on
    library detection.

    Args:
        state: Current graph state

    Returns:
        State updates with related_issues and library_detection
    """
    bug_report = state.get("bug_report", {})

    # Combine all text for library detection
    full_text = "\n".join([
        bug_report.get("raw_description", ""),
        bug_report.get("title", ""),
        bug_report.get("error_message", ""),
        bug_report.get("stack_trace", ""),
    ])

    # Detect which libraries are involved
    library_detection = detect_libraries(full_text)
    primary_library = library_detection["primary"]

    all_issues = []
    seen_keys = set()  # Use repo+number as key to allow same number from different repos

    def add_issue(issue: GitHubIssue, repo: str):
        """Add issue if not already seen."""
        key = f"{repo}#{issue['number']}"
        if key not in seen_keys:
            # Add repo info to the issue
            issue["repo"] = repo
            all_issues.append(issue)
            seen_keys.add(key)

    # Determine which repos to search
    if primary_library != "unknown":
        # Search the primary library's repo first, then others
        repos_to_search = [get_repo_for_library(primary_library)]
        for lib in library_detection["all_libraries"]:
            repo = get_repo_for_library(lib)
            if repo not in repos_to_search:
                repos_to_search.append(repo)
    else:
        # Search all LangChain ecosystem repos
        repos_to_search = config.langchain_repos

    # Build search queries
    queries = []

    # Query 1: Title-based (most specific)
    if bug_report.get("title"):
        queries.append(("title", bug_report["title"]))

    # Query 2: Error message keywords
    if bug_report.get("error_message"):
        error_line = bug_report["error_message"].split("\n")[0]
        error_keywords = []
        for word in error_line.split():
            if len(word) > 3 and (
                "Error" in word
                or "Exception" in word
                or (word[0].isupper() and not word.isupper())
            ):
                error_keywords.append(word.strip(":'\"(),"))
        if error_keywords:
            queries.append(("error", " ".join(error_keywords[:5])))

    # Query 3: Component-specific search
    if library_detection["components"]:
        # Pick the most specific component (longest name)
        best_component = max(library_detection["components"], key=len)
        queries.append(("component", best_component))

    # Query 4: Raw description fallback
    if not queries:
        raw = bug_report.get("raw_description", "")[:100]
        if raw:
            queries.append(("fallback", raw))

    # Execute searches across repos
    for repo in repos_to_search:
        for query_type, query in queries:
            # Limit results per query to avoid too many API calls
            max_per_query = 3 if query_type in ("title", "error") else 2
            issues = search_github_issues(query, repo=repo, max_results=max_per_query)
            for issue in issues:
                add_issue(issue, repo)

    # Sort by relevance score and limit to top 8 (increased from 5)
    all_issues.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
    all_issues = all_issues[:8]

    # Compute GitHub confidence score based on match quality
    github_confidence = compute_github_confidence(
        all_issues, bug_report.get("title", "")
    )

    return {
        "related_issues": all_issues,
        "library_detection": library_detection,
        "github_confidence": github_confidence,
        "current_phase": "searching",  # Stay in searching for RAG
    }


def compute_github_confidence(issues: list[GitHubIssue], ticket_title: str) -> float:
    """
    Compute a confidence score for GitHub search results.

    Factors:
    - Exact or near-exact title match (high confidence)
    - Closed issues with solutions (higher confidence)
    - Number of relevant issues found
    - Relevance scores of top issues

    Returns:
        Confidence score from 0.0 to 1.0
    """
    if not issues:
        return 0.0

    confidence = 0.0
    ticket_title_lower = ticket_title.lower().strip()

    # Factor 1: Title similarity (up to 0.4)
    if ticket_title_lower:
        best_title_match = 0.0
        for issue in issues[:3]:  # Check top 3 issues
            issue_title_lower = issue["title"].lower().strip()
            # Check for exact match
            if ticket_title_lower == issue_title_lower:
                best_title_match = 1.0
                break
            # Check for high overlap (most words match)
            ticket_words = set(ticket_title_lower.split())
            issue_words = set(issue_title_lower.split())
            if ticket_words and issue_words:
                overlap = len(ticket_words & issue_words) / max(
                    len(ticket_words), len(issue_words)
                )
                best_title_match = max(best_title_match, overlap)
        confidence += best_title_match * 0.4

    # Factor 2: Closed issues with potential solutions (up to 0.3)
    closed_count = sum(1 for issue in issues if issue.get("state") == "closed")
    if closed_count > 0:
        confidence += min(0.3, closed_count * 0.1)

    # Factor 3: Number of relevant issues found (up to 0.2)
    if len(issues) >= 3:
        confidence += 0.2
    elif len(issues) >= 1:
        confidence += 0.1

    # Factor 4: Average relevance score of top issues (up to 0.1)
    if issues:
        avg_relevance = sum(
            issue.get("relevance_score", 0) for issue in issues[:3]
        ) / min(3, len(issues))
        confidence += avg_relevance * 0.1

    return min(1.0, confidence)
