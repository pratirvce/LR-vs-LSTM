# GitHub Upload Instructions

## Quick Upload (Recommended)

Run the automated script:

```bash
cd submission_final
./upload_to_github.sh
```

## Manual Upload (If Script Fails)

If you prefer to run commands manually or the script has issues:

### Step 1: Navigate to folder
```bash
cd /Users/pratibharevankar/Desktop/submission/submission_final
```

### Step 2: Prepare README for GitHub
```bash
mv README.md README_ORIGINAL.md
mv README_GITHUB.md README.md
```

### Step 3: Initialize Git
```bash
git init
```

### Step 4: Add remote repository
```bash
git remote add origin https://github.com/pratirvce/LR-vs-LSTM.git
```

### Step 5: Stage all files
```bash
git add .
```

### Step 6: Check what will be committed
```bash
git status
```

You should see:
- assignment_solution.py
- error_analysis.py
- README.md
- requirements.txt
- .gitignore
- outputs/ (4 .npy files)
- overleaf_report/ (main.tex + 8 plots)
- And other documentation files

### Step 7: Create commit
```bash
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
```

### Step 8: Set branch to main
```bash
git branch -M main
```

### Step 9: Push to GitHub
```bash
git push -u origin main
```

**Note:** You may be prompted for GitHub credentials:
- Username: `pratirvce`
- Password: Use a Personal Access Token (not your password)

### Step 10: Verify upload
Visit: https://github.com/pratirvce/LR-vs-LSTM

## Creating a Personal Access Token (if needed)

If Git asks for a password, you need a Personal Access Token:

1. Go to GitHub.com → Settings → Developer settings
2. Click "Personal access tokens" → "Tokens (classic)"
3. Click "Generate new token" → "Generate new token (classic)"
4. Give it a name: "LR-vs-LSTM Upload"
5. Select scopes: Check "repo" (full control of private repositories)
6. Click "Generate token"
7. Copy the token (you won't see it again!)
8. Use this token as your password when pushing

## Alternative: Upload via GitHub Web Interface

If git commands are not working:

1. Go to https://github.com/pratirvce/LR-vs-LSTM
2. Click "uploading an existing file"
3. Drag and drop all files from `submission_final/`
4. Add commit message
5. Click "Commit changes"

Note: This method requires uploading files in batches due to GitHub's web interface limitations.

## Troubleshooting

### "Permission denied" error
- Make sure you're authenticated with GitHub
- Use a Personal Access Token instead of password
- Or use SSH keys

### "Repository not found" error
- Verify the repository exists: https://github.com/pratirvce/LR-vs-LSTM
- Check that you have write access to the repository

### Files too large
- GitHub has a 100MB file limit
- Our largest files are prediction arrays (~100KB each) - well under the limit
- If you added model weights (.pt files), they might be too large

## What Gets Uploaded

✓ Source code (2 Python files)
✓ Documentation (README, requirements.txt)
✓ Prediction outputs (4 .npy files, ~240KB total)
✓ LaTeX report (main.tex)
✓ Plots (8 PNG images, ~1.1MB total)
✓ Instructions and guides

✗ NOT uploaded (per .gitignore):
- Large model weight files (*.pt)
- Python cache files
- IDE settings
- OS files (.DS_Store)

Total upload size: ~1.5 MB

## After Upload

Once uploaded, your repository will show:
- 📊 Professional README with results table
- 📁 Clean file structure
- 🎯 All deliverables
- 📈 Plots visible in README
- 📝 Complete documentation

Good luck! 🚀
