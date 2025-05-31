# 1. Create repository on GitHub named DARREN-2000.github.io
# 2. Clone the repository
git clone https://github.com/DARREN-2000/DARREN-2000.github.io.git
cd DARREN-2000.github.io

# 3. Create directory structure
mkdir -p css js images/{hero,companies,projects,technologies,certifications,misc}
mkdir -p documents/{transcripts,certifications,references,publications,language,resume}
mkdir -p assets/{fonts,data,animations}
mkdir -p .github/{workflows,ISSUE_TEMPLATE}

# 4. Initialize npm project
npm init -y

# 5. Install development dependencies
npm install --save-dev gh-pages html-minifier imagemin lighthouse live-server

# 6. Add all files and commit
git add .
git commit -m "Initial portfolio setup with complete structure"
git push origin main

# 7. Enable GitHub Pages in repository settings
# Go to Settings > Pages > Source: Deploy from branch (main)
