# Upload the ALS recommender to GitHub

Repository: <https://github.com/ahmedamen1/Recommendation-System-using-ALS->

Clone URL: `https://github.com/ahmedamen1/Recommendation-System-using-ALS-.git`

The report is `report/aamin_AMLAS_draft.pdf`. Its abstract ends with a clickable
link to this repository. The asterisks in the requested name were interpreted as
Markdown formatting; the filename uses `aamin_AMLAS_draft.pdf`.

## 1. Files prepared for upload

A clean copy of the current project is prepared locally in
`.build/github-upload-package/`. It contains:

| Files or folder | Purpose |
|---|---|
| `train.py`, `movie_recommender.ipynb`, `notebook_tools.py` | Training code and notebook |
| `test_train.py`, `requirements.txt` | Existing training checks and dependencies |
| `dashboard/`, `start_dashboard.bat` | Local application, static assets, demo model, and dependencies |
| `inference_bundle/` | Compact saved model and catalog needed to run the application |
| `results/`, except `results/models/` | Recorded metrics, learning histories, and embedding coordinates |
| `report/` | Named draft PDF, LaTeX source, vector figures, tables, bibliography, and audit scripts |
| `README.md`, `.gitignore`, `GITHUB_UPLOAD.md` | Project documentation and upload instructions |

The copy excludes raw data, full training checkpoints, virtual environments,
local caches and archives, Python bytecode, temporary LaTeX files, credentials,
and duplicate `main.pdf`/`main_updated.pdf` report copies. The project downloads
MovieLens when training is run. Full training checkpoints must be regenerated
for checks that require those files; the compact inference model is included
for running the application immediately.

This upload folder is a snapshot of the current draft. For subsequent edits,
copy the changed project files into the cloned repository before committing.

## 2. Clone your GitHub repository

Open PowerShell and run the following from the project folder. Run `git clone`
only once for this destination. Cloning first retains the repository's existing
history and also works when the remote repository is empty.

```powershell
Set-Location "D:\AIMS_Courses_learning\recommender_system"
git clone "https://github.com/ahmedamen1/Recommendation-System-using-ALS-.git" ".build\github-upload"
if ($LASTEXITCODE -ne 0) { throw "Clone failed. Check the repository URL and sign-in before continuing." }
```

If Git reports that the repository cannot be found, confirm that it exists and
authenticate with the GitHub account that owns it. A private repository requires
authenticated access.

## 3. Copy the prepared files into the clone

```powershell
robocopy ".build\github-upload-package" ".build\github-upload" /E
if ($LASTEXITCODE -ge 8) { throw "File copy failed. Review the robocopy output before continuing." }
Set-Location ".build\github-upload"
git status --short
```

`robocopy /E` copies the prepared files without deleting files already in the
remote clone. [Robocopy exit codes](https://learn.microsoft.com/en-us/windows-server/administration/windows-commands/robocopy)
below 8 indicate success or nonfatal copy differences; 8 or higher indicates a
copy failure.

## 4. Set your commit identity and review the upload

Replace the email placeholder with the verified email or private noreply email
shown in your GitHub account settings. These settings apply only to this clone.

```powershell
git config user.name "Ahmed Amin"
git config user.email "YOUR_VERIFIED_GITHUB_EMAIL"
git add .
git diff --cached --stat
git status --short
```

Confirm that `report/aamin_AMLAS_draft.pdf`, the code, and `inference_bundle/`
are listed. The dataset, `.venv/`, `.build/`, and `results/models/` should not be
listed. The supplied `.gitignore` excludes them.

## 5. Commit and upload

```powershell
git commit -m "Add ALS recommender, results, and AAMIN report draft"
git push -u origin HEAD
```

Authenticate when Git prompts. [`HEAD` pushes the current branch](https://git-scm.com/docs/git-push),
so it works with the repository's existing default branch name.

If the default branch requires a pull request, push the same commit on a new
branch instead:

```powershell
git switch -c report-draft
git push -u origin report-draft
```

Then open the repository on GitHub and choose **Compare & pull request**.

## 6. Review the uploaded project

Open <https://github.com/ahmedamen1/Recommendation-System-using-ALS-> and confirm:

- The README links to `report/aamin_AMLAS_draft.pdf`.
- The report opens and the link at the end of its abstract opens this repository.
- Training code, the notebook, the dashboard, and the compact inference bundle
  are present.
- Raw MovieLens downloads and full training checkpoints are absent.

To run a fresh checkout:

```powershell
python -m pip install -r dashboard/requirements.txt
python dashboard/server.py
```

Open <http://127.0.0.1:8000>.

## File sizes and official instructions

GitHub blocks files larger than 100 MiB in ordinary Git repositories and limits
browser uploads to 25 MiB per file. The prepared project excludes the raw dataset
and full training checkpoints. See [GitHub's file size documentation](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).

GitHub documents initialization, remotes, authentication, commits, and pushes in
[Adding locally hosted code to GitHub](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github).
