#!/usr/bin/env python3
"""Create a companion GitHub Discussion for a Mila blog post and link the two.

Reads a post's front matter (title, description, slug), opens a Discussion in the configured
category via `gh api graphql`, then writes the new Discussion URL back into the post's
`discussion:` front-matter field so the blog's "Discuss on GitHub" link resolves. Every slot in
the thread is a projection of the post -- nothing is authored twice.

Auth comes from the GitHub CLI: this shells out to `gh` and never handles a token itself, so
`gh auth status` must be green.

Usage:
    python Tools/Blog/new_post_discussion.py <slug-or-path> [--category NAME] [--dry-run] [--force]

Examples:
    python Tools/Blog/new_post_discussion.py flash-decoding-mqa-cuda --dry-run
    python Tools/Blog/new_post_discussion.py flash-decoding-mqa-cuda
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_OWNER = "ToddThomson"
REPO_NAME = "Mila"
BASE_URL = "https://mila.toddt.me"
DEFAULT_CATEGORY = "Show and tell"

# The companion-thread body. {title}/{url}/{summary} are filled from the post's front matter.
BODY_TEMPLATE = """\
> [!NOTE]
> **Full writeup on the Mila blog:** [{title}]({url})

{summary}

Discussion lives here — questions, corrections, and war stories welcome below.
"""


def repo_root() -> Path:
    # This file lives at <root>/Tools/Blog/new_post_discussion.py.
    return Path(__file__).resolve().parents[2]


def blog_dir() -> Path:
    return repo_root() / "Web" / "content" / "blog"


def resolve_post(slug_or_path: str) -> Path:
    candidate = Path(slug_or_path)
    if candidate.suffix == ".md" and candidate.exists():
        return candidate.resolve()

    by_slug = blog_dir() / f"{slug_or_path}.md"
    if by_slug.exists():
        return by_slug

    sys.exit(f"error: no blog post found for '{slug_or_path}' (looked for {by_slug})")


def split_front_matter(text: str):
    """Return (front_matter_lines, closing_fence_index, all_lines_with_endings)."""
    if not text.startswith("---"):
        sys.exit("error: post has no YAML front matter (missing opening --- fence)")

    lines = text.splitlines(keepends=True)
    closing = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
    if closing is None:
        sys.exit("error: unterminated front matter (no closing ---)")

    return lines[1:closing], closing, lines


def fm_get(front_matter, key):
    for line in front_matter:
        match = re.match(rf"^{re.escape(key)}:\s*(.*)$", line)
        if match:
            value = match.group(1).strip()
            if len(value) >= 2 and value[0] in "\"'" and value[-1] == value[0]:
                value = value[1:-1]
            return value

    return None


def gh_graphql(query: str, **variables):
    args = ["gh", "api", "graphql", "-f", f"query={query}"]
    for key, value in variables.items():
        args += ["-f", f"{key}={value}"]

    try:
        result = subprocess.run(args, capture_output=True, text=True, encoding="utf-8")
    except FileNotFoundError:
        sys.exit("error: `gh` not found. Install the GitHub CLI and run `gh auth login`.")

    if result.returncode != 0:
        sys.exit(f"gh error:\n{result.stderr.strip()}")

    data = json.loads(result.stdout)
    if "errors" in data:
        sys.exit("graphql error:\n" + json.dumps(data["errors"], indent=2))

    return data["data"]


def get_repo_and_category(category_name: str):
    query = """
    query($owner:String!, $name:String!){
      repository(owner:$owner, name:$name){
        id
        discussionCategories(first:25){ nodes{ id name } }
      }
    }"""
    repo = gh_graphql(query, owner=REPO_OWNER, name=REPO_NAME)["repository"]
    categories = {c["name"]: c["id"] for c in repo["discussionCategories"]["nodes"]}

    if category_name not in categories:
        sys.exit(f"error: category '{category_name}' not found. Available: {', '.join(categories)}")

    return repo["id"], categories[category_name]


def create_discussion(repository_id: str, category_id: str, title: str, body: str):
    mutation = """
    mutation($repositoryId:ID!, $categoryId:ID!, $title:String!, $body:String!){
      createDiscussion(input:{repositoryId:$repositoryId, categoryId:$categoryId, title:$title, body:$body}){
        discussion{ url number }
      }
    }"""
    discussion = gh_graphql(
        mutation,
        repositoryId=repository_id,
        categoryId=category_id,
        title=title,
        body=body,
    )["createDiscussion"]["discussion"]

    return discussion["url"], discussion["number"]


def write_back_discussion(path: Path, front_matter, closing_index, all_lines, url: str):
    """Set (or insert) the `discussion:` field in the post's front matter, leaving all else intact."""
    new_line = f'discussion: "{url}"\n'

    for i, line in enumerate(front_matter):
        if re.match(r"^discussion:\s*", line):
            front_matter[i] = new_line
            break
    else:
        # Insert after description:, else after title:, else at the end of the block.
        anchor = next((i for i, l in enumerate(front_matter) if re.match(r"^description:\s*", l)), None)
        if anchor is None:
            anchor = next((i for i, l in enumerate(front_matter) if re.match(r"^title:\s*", l)), None)
        insert_at = (anchor + 1) if anchor is not None else len(front_matter)
        front_matter.insert(insert_at, new_line)

    rebuilt = [all_lines[0]] + front_matter + all_lines[closing_index:]
    path.write_text("".join(rebuilt), encoding="utf-8")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    parser = argparse.ArgumentParser(
        description="Create a companion GitHub Discussion for a Mila blog post and link the two."
    )
    parser.add_argument("post", help="post slug (e.g. flash-decoding-mqa-cuda) or path to the .md")
    parser.add_argument("--category", default=DEFAULT_CATEGORY,
                        help=f"Discussion category (default: {DEFAULT_CATEGORY!r})")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the thread that would be created; make no changes")
    parser.add_argument("--force", action="store_true",
                        help="create even if the post already has a discussion: link")
    args = parser.parse_args()

    path = resolve_post(args.post)
    slug = path.stem
    front_matter, closing_index, all_lines = split_front_matter(path.read_text(encoding="utf-8"))

    title = fm_get(front_matter, "title")
    description = fm_get(front_matter, "description") or ""
    existing = fm_get(front_matter, "discussion")

    if not title:
        sys.exit("error: post front matter has no title:")
    if existing and not args.force:
        sys.exit(f"post already links a discussion:\n  {existing}\nUse --force to create another.")
    if not description:
        print("warning: post has no description: -- summary will be a placeholder", file=sys.stderr)

    url = f"{BASE_URL}/blog/{slug}/"
    summary = description or "_(add a one-line summary)_"
    body = BODY_TEMPLATE.format(title=title, url=url, summary=summary)

    print(f"post:     {path}")
    print(f"title:    {title}")
    print(f"blog url: {url}")
    print(f"category: {args.category}")
    print("--- discussion body ---")
    print(body)
    print("-----------------------")

    if args.dry_run:
        print("dry-run: nothing created.")
        return

    repository_id, category_id = get_repo_and_category(args.category)
    discussion_url, number = create_discussion(repository_id, category_id, title, body)
    print(f"created Discussion #{number}: {discussion_url}")

    write_back_discussion(path, front_matter, closing_index, all_lines, discussion_url)
    print(f'wrote discussion: "{discussion_url}" into {path.name}')
    print("next: publish (push to dev) so the post's 'Discuss on GitHub' link points at the new thread.")


if __name__ == "__main__":
    main()
