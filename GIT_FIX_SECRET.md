# How to fix the blocked push (secret in history)

GitHub blocked the push because **commit a627f60** in your history contains an Azure API key in `src/config/settings.py`. Your latest commit (016bf2a) is already clean, but the old commit is still in the history.

Use one of these approaches.

---

## Option A: New history with one clean commit (recommended)

This creates a new root commit with your current (clean) code and replaces `main` with it. No secret in history.

Run in the project root (PowerShell):

```powershell
# 1. Create a new branch with no history (orphan)
git checkout --orphan temp_main

# 2. Stage everything (current clean code)
git add -A

# 3. Create one new commit with no parent
git commit -m "Confidence Aware Agent code"

# 4. Delete the old main branch
git branch -D main

# 5. Rename this branch to main
git branch -m main

# 6. Force push the new history (no secret)
git push --force origin main
```

After this, your repo will have a single commit on `main` with no secrets. You can delete `GIT_FIX_SECRET.md` and commit that in a normal way later.

---

## Option B: Edit the bad commit with rebase

If you want to keep two commits but remove the secret from the first one:

```powershell
# 1. Start interactive rebase from root
git rebase -i --root

# 2. In the editor: change "pick" to "edit" for the first commit (a627f60). Save and close.

# 3. When rebase stops at that commit, overwrite settings.py with the clean version:
git show 016bf2a:src/config/settings.py | Out-File -Encoding utf8 src/config/settings.py

# 4. Amend the commit and continue
git add src/config/settings.py
git commit --amend --no-edit
git rebase --continue

# 5. Force push
git push --force-with-lease origin main
```

---

## Important: Rotate the exposed key

The API key that was in commit a627f60 is now exposed (it’s in your local history and possibly in GitHub’s scan). You should:

1. In **Azure Portal** → your Cognitive Services resource → **Keys and Endpoint**
2. Regenerate the key (or create a new one and switch to it)
3. Update your local `.env` with the new key
4. Do **not** use the old key anymore

After rotating the key, the blocked-push fix above is still needed so that the old secret is no longer in the branch history you push.
