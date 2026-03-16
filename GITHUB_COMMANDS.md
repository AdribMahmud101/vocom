# GitHub Commands Cheat Sheet

This file lists the most useful day-to-day Git and GitHub CLI commands.

## 1. Setup

```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
git config --global init.defaultBranch main
```

Check your config:

```bash
git config --list
```

## 2. Start Working With a Repository

Clone:

```bash
git clone https://github.com/<owner>/<repo>.git
cd <repo>
```

Or initialize a new local repo:

```bash
git init
```

Add remote later:

```bash
git remote add origin https://github.com/<owner>/<repo>.git
git remote -v
```

## 3. Daily Workflow

Check status:

```bash
git status
```

See changes:

```bash
git diff
git diff --staged
```

Stage files:

```bash
git add <file>
git add .
```

Commit:

```bash
git commit -m "Short clear message"
```

Push current branch:

```bash
git push
```

First push of a new branch:

```bash
git push -u origin <branch-name>
```

Pull latest updates:

```bash
git pull
```

## 4. Branching

List branches:

```bash
git branch
git branch -a
```

Create and switch:

```bash
git switch -c <branch-name>
```

Switch existing branch:

```bash
git switch <branch-name>
```

Merge into current branch:

```bash
git merge <branch-name>
```

Delete local branch:

```bash
git branch -d <branch-name>
```

Force delete local branch:

```bash
git branch -D <branch-name>
```

## 5. Sync With Main

```bash
git switch main
git pull origin main
git switch <feature-branch>
git merge main
```

## 6. View History

Compact history:

```bash
git log --oneline --graph --decorate --all
```

Show one commit:

```bash
git show <commit-hash>
```

## 7. Undo / Fix Mistakes

Unstage file:

```bash
git restore --staged <file>
```

Discard local file changes:

```bash
git restore <file>
```

Amend last commit message/content:

```bash
git commit --amend
```

Revert a commit safely (keeps history):

```bash
git revert <commit-hash>
```

Reset branch to previous commit (local history rewrite):

```bash
git reset --soft HEAD~1
git reset --mixed HEAD~1
```

## 8. Stash

Save temporary work:

```bash
git stash
```

List stashes:

```bash
git stash list
```

Re-apply latest stash:

```bash
git stash pop
```

## 9. Tags and Releases

Create tag:

```bash
git tag v1.0.0
```

Push tag:

```bash
git push origin v1.0.0
```

Push all tags:

```bash
git push --tags
```

## 10. GitHub CLI (Optional but Useful)

Authenticate:

```bash
gh auth login
```

Create repo from current folder:

```bash
gh repo create <repo-name> --public --source=. --remote=origin --push
```

Create pull request:

```bash
gh pr create --fill
```

List PRs:

```bash
gh pr list
```

Check out a PR locally:

```bash
gh pr checkout <pr-number>
```

## 11. Useful Remote Commands

Show remotes:

```bash
git remote -v
```

Rename remote:

```bash
git remote rename origin upstream
```

Change remote URL:

```bash
git remote set-url origin https://github.com/<owner>/<repo>.git
```

## 12. Recommended Team Workflow

1. Update main: `git switch main && git pull origin main`
2. Create feature branch: `git switch -c feature/<name>`
3. Work + commit: `git add . && git commit -m "..."`
4. Push branch: `git push -u origin feature/<name>`
5. Open PR: `gh pr create --fill` (or via GitHub web UI)
6. Merge PR, then clean up branch.
