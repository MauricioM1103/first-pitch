# Pushing this repo to GitHub

You have a fully committed local repo at `C:\Users\MR123\first-pitch`. To
push it to GitHub:

1. Sign in at https://github.com.
2. Click the **+** (top right) → **New repository**.
3. Name it something like `first-pitch`. Choose **Public** (Render's free
   tier needs a public repo to auto-deploy — or Private if you're OK
   connecting Render to your GitHub account with permissions).
4. **Do NOT** check "Add a README" or add a .gitignore — this repo already
   has both. Leave the repo empty.
5. Click **Create repository**.
6. GitHub shows a "…or push an existing repository from the command line"
   block. Copy the two commands that look like:

   ```
   git remote add origin https://github.com/<you>/first-pitch.git
   git branch -M main
   git push -u origin main
   ```

7. Paste them into a terminal in this folder:

   ```powershell
   cd C:\Users\MR123\first-pitch
   git remote add origin https://github.com/<you>/first-pitch.git
   git branch -M main
   git push -u origin main
   ```

   Git will prompt for GitHub credentials. Use a personal access token if
   asked for a password (github.com → Settings → Developer settings →
   Personal access tokens → Fine-grained tokens → generate one with
   "repo" scope).

8. Done — the repo is on GitHub. Now follow the Render steps in README.md.
