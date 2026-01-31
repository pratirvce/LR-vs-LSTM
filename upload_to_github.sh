#!/bin/bash
# GitHub Upload Script for LR-vs-LSTM Repository
# Run this script from the submission_final directory

echo "=========================================="
echo "GitHub Upload Script"
echo "Repository: https://github.com/pratirvce/LR-vs-LSTM"
echo "=========================================="
echo ""

# Step 1: Navigate to the correct directory
echo "Step 1: Navigating to submission_final..."
cd "$(dirname "$0")"
pwd

# Step 2: Initialize git repository (if not already)
echo ""
echo "Step 2: Initializing Git repository..."
git init

# Step 3: Add remote (if not already added)
echo ""
echo "Step 3: Adding GitHub remote..."
git remote add origin https://github.com/pratirvce/LR-vs-LSTM.git 2>/dev/null || git remote set-url origin https://github.com/pratirvce/LR-vs-LSTM.git

# Step 4: Rename README_GITHUB.md to README.md
echo ""
echo "Step 4: Setting up README.md for GitHub..."
if [ -f "README_GITHUB.md" ]; then
    mv README.md README_ORIGINAL.md
    mv README_GITHUB.md README.md
    echo "✓ README.md prepared for GitHub"
fi

# Step 5: Create .gitignore if it doesn't exist
echo ""
echo "Step 5: Checking .gitignore..."
if [ -f ".gitignore" ]; then
    echo "✓ .gitignore exists"
else
    echo "✗ .gitignore not found"
fi

# Step 6: Stage all files
echo ""
echo "Step 6: Staging files for commit..."
git add .
git add .gitignore

# Step 7: Show status
echo ""
echo "Step 7: Git status:"
git status

# Step 8: Create commit
echo ""
echo "Step 8: Creating commit..."
git commit -m "Initial commit: Logistic Regression vs LSTM sentiment analysis

- Implemented Logistic Regression with embeddings (88.14% test accuracy)
- Implemented LSTM with packing/unpacking (83.95% test accuracy)
- Comprehensive hyperparameter tuning (batch size, learning rate)
- Model correctness verification
- Error analysis with visualizations
- 8 plots comparing both models
- LaTeX report with all deliverables
- Prediction outputs for validation and test sets

Key Finding: Simpler LR model outperformed LSTM by 4.19%!"

# Step 9: Set default branch to main
echo ""
echo "Step 9: Setting default branch to main..."
git branch -M main

# Step 10: Push to GitHub
echo ""
echo "Step 10: Pushing to GitHub..."
echo "NOTE: You may be prompted for your GitHub credentials"
echo ""
git push -u origin main

echo ""
echo "=========================================="
echo "✓ Upload Complete!"
echo "=========================================="
echo ""
echo "Your repository is now available at:"
echo "https://github.com/pratirvce/LR-vs-LSTM"
echo ""
echo "Next steps:"
echo "1. Visit the repository URL above"
echo "2. Verify all files are uploaded"
echo "3. Check that README.md displays correctly"
echo "4. Add topics/tags to your repository (optional)"
echo ""
