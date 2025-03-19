#!/bin/bash
# Script to initialize and push the Interview Analyzer project to GitHub

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=== Initializing GitHub Repository for Interview Analyzer ===${NC}"

# Ensure we're in the correct directory
cd "$(dirname "$0")"
ROOT_DIR="$(pwd)"
echo -e "Working directory: ${GREEN}${ROOT_DIR}${NC}"

# Check if .git already exists
if [ -d ".git" ]; then
    echo -e "${RED}Git repository already initialized.${NC}"
    echo -e "If you want to start fresh, you can delete the .git directory."
    read -p "Do you want to continue with the existing Git repository? [Y/n] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]] && [[ ! $REPLY == "" ]]; then
        echo -e "${RED}Aborted.${NC}"
        exit 1
    fi
else
    # Initialize git repository
    echo -e "\n${YELLOW}Initializing Git repository...${NC}"
    rm -rf .git
    git init
fi

# Remind about the GitHub repository
echo -e "\n${YELLOW}Before continuing, please create a new repository on GitHub.${NC}"
echo -e "Go to ${GREEN}https://github.com/new${NC}"
echo -e "Repository name: ${GREEN}interview_analyzer${NC}"
echo -e "Description: ${GREEN}A comprehensive tool for analyzing interviews using facial expression recognition, speech emotion detection, and text sentiment analysis.${NC}"
echo -e "Set the repository to ${GREEN}Public${NC} or ${GREEN}Private${NC} as desired."
echo -e "Do NOT initialize with README, .gitignore, or license as we already have those."

# Ask for GitHub repository URL
read -p "Enter your GitHub repository URL (e.g., https://github.com/biggs122/Interview_Analyzer.git): " REPO_URL

if [ -z "$REPO_URL" ]; then
    echo -e "${RED}No repository URL provided. Exiting.${NC}"
    exit 1
fi

# Add specific files and directories
git add .gitignore README.md requirements.txt push_to_github.sh src/ tests/

# Add .pkl files to gitignore
echo "/models/**/*.pkl" >> .gitignore

# Check status
echo -e "\n${YELLOW}Current Git status:${NC}"
git status

# Confirm before commit
read -p "Do you want to commit these files? [Y/n] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]] && [[ ! $REPLY == "" ]]; then
    echo -e "${RED}Commit aborted.${NC}"
    exit 1
fi

# Commit
echo -e "\n${YELLOW}Committing files...${NC}"
git commit -m "Initial commit without data and model files"

# Add remote
echo -e "\n${YELLOW}Adding remote repository...${NC}"
git remote add origin "$REPO_URL"

# Push to GitHub
echo -e "\n${YELLOW}Pushing to GitHub...${NC}"
git push -u origin main

# Check if push was successful
if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}Successfully pushed to GitHub!${NC}"
    echo -e "Your repository is now available at: ${GREEN}${REPO_URL}${NC}"
else
    echo -e "\n${RED}Push failed. Please check your repository URL and try again.${NC}"
    echo -e "You can manually push using:"
    echo -e "${GREEN}git push -u origin main${NC}"
fi

# Remove data directory from git tracking
git rm -r --cached data 

rm -rf .git # Supprimez l'ancien dépôt Git
cp -r /tmp/interview_analyzer_clean/.git .

cd /Users/abderrahim_boussyf/interview_analyzer
rm -rf .git
cp -r /tmp/interview_analyzer_clean/.git .
git remote add origin https://github.com/biggs122/Interview_Analyzer.git
git push -u origin main --force 