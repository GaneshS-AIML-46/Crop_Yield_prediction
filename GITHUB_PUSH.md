# One-time GitHub push (run after creating repo on github.com)

# 1. Create repo at https://github.com/new (name: crop-yield-prediction, Public)

# 2. Replace YOUR_USERNAME below, then run in PowerShell:

git remote add origin https://github.com/YOUR_USERNAME/crop-yield-prediction.git
git push -u origin main

# 3. Enable GitHub Pages: Repo Settings -> Pages -> Branch main, Folder /frontend

# 4. Deploy backend on Render.com: New Web Service -> connect repo -> use render.yaml

# 5. Update frontend/config.js with your Render URL, then:
git add frontend/config.js README.md
git commit -m "Set production API URL"
git push
