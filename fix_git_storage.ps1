# Remove Qdrant's binary storage from git.
#
# Commit c3c7895 committed data/qdrant_server/ - Qdrant's own on-disk index.
# That folder holds three 32 MB write-ahead-log files plus the vector segments,
# so roughly 150-200 MB of binary that changes on every single query. Left in
# git it makes the repo unusable to clone and GitHub will eventually refuse it.
#
# This script inspects first and only rewrites when it is safe to do so.
# It never touches the Qdrant data itself - only what git tracks.
#
#   powershell -ExecutionPolicy Bypass -File .\fix_git_storage.ps1          # inspect
#   powershell -ExecutionPolicy Bypass -File .\fix_git_storage.ps1 -Apply   # fix

param([switch]$Apply)

$ErrorActionPreference = "Stop"

if (-not (Test-Path ".git")) {
    Write-Host "Run this from the repo root (C:\Users\Akshat\projects\adaptive-retriever)."
    exit 1
}

Write-Host ("=" * 70)
Write-Host "CURRENT STATE"
Write-Host ("=" * 70)

$tracked = @(git ls-files -- data/qdrant_server)
Write-Host ("  files tracked under data/qdrant_server : {0}" -f $tracked.Count)

$commits = @(git log --oneline --all -- data/qdrant_server)
Write-Host ("  commits that touched it                : {0}" -f $commits.Count)
foreach ($c in $commits) { Write-Host "      $c" }

$sizeRaw = git count-objects -vH
$sizeLine = ($sizeRaw | Select-String "size-pack")
Write-Host ("  packed repo size                       : {0}" -f ($sizeLine -replace 'size-pack: ', ''))

$upstream = git rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>$null
if ($LASTEXITCODE -ne 0 -or -not $upstream) {
    $pushed = $false
    Write-Host "  branch has no upstream                 : nothing has been pushed"
} else {
    $ahead = @(git log --oneline "$upstream..HEAD")
    $pushed = ($ahead.Count -eq 0)
    Write-Host ("  upstream                               : {0}" -f $upstream)
    Write-Host ("  commits not yet pushed                 : {0}" -f $ahead.Count)
}

$ignored = Select-String -Path ".gitignore" -Pattern "data/qdrant_server/" -Quiet
Write-Host ("  .gitignore covers it                   : {0}" -f $ignored)

if ($tracked.Count -eq 0) {
    Write-Host ""
    Write-Host "Nothing tracked under data/qdrant_server. Already clean."
    exit 0
}

Write-Host ""
Write-Host ("=" * 70)
Write-Host "PLAN"
Write-Host ("=" * 70)

$canRewrite = ($commits.Count -eq 1) -and (-not $pushed)

if ($canRewrite) {
    Write-Host "  Exactly one unpushed commit added these files, so the clean fix is"
    Write-Host "  to rebuild that commit without them. History stays linear and the"
    Write-Host "  binaries never enter a pushed history."
    Write-Host ""
    Write-Host "    git reset --soft HEAD~1"
    Write-Host "    git rm -r --cached data/qdrant_server --ignore-unmatch"
    Write-Host "    git commit -m <same message>"
} else {
    Write-Host "  Cannot safely rewrite in place:"
    if ($commits.Count -ne 1) { Write-Host ("    - {0} commits touched the folder, not 1" -f $commits.Count) }
    if ($pushed)              { Write-Host  "    - the commit is already pushed" }
    Write-Host ""
    Write-Host "  Falling back to untracking going forward. The blobs stay in history;"
    Write-Host "  clearing those needs git filter-repo, which is a separate decision."
    Write-Host ""
    Write-Host "    git rm -r --cached data/qdrant_server --ignore-unmatch"
    Write-Host "    git commit -m 'Untrack Qdrant server storage'"
}

if (-not $Apply) {
    Write-Host ""
    Write-Host "Inspection only. Re-run with  -Apply  to make the change."
    exit 0
}

Write-Host ""
Write-Host ("=" * 70)
Write-Host "APPLYING"
Write-Host ("=" * 70)

if (-not $ignored) {
    Add-Content .gitignore "`ndata/qdrant_server/"
    Write-Host "  added data/qdrant_server/ to .gitignore"
}

# NOTE: no `git add -A` anywhere below. Other edits in the working tree stay
# unstaged so they are not swallowed into a commit that is not about them.
if ($canRewrite) {
    $msg = git log -1 --pretty=%B
    git reset --soft HEAD~1
    git rm -r --cached data/qdrant_server --ignore-unmatch --quiet
    git add .gitignore
    $msg | git commit -F -
    Write-Host "  rebuilt the commit without the Qdrant storage"
} else {
    git rm -r --cached data/qdrant_server --ignore-unmatch --quiet
    git add .gitignore
    git commit -m "Untrack Qdrant server storage - binary index files must not be versioned"
    Write-Host "  untracked going forward"
}

Write-Host ""
Write-Host ("=" * 70)
Write-Host "AFTER"
Write-Host ("=" * 70)
$stillTracked = @(git ls-files -- data/qdrant_server)
Write-Host ("  files still tracked under data/qdrant_server : {0}" -f $stillTracked.Count)
git log --oneline -2
Write-Host ""
Write-Host "  Qdrant's data on disk was not touched. Nothing needs re-indexing."
if ($stillTracked.Count -eq 0) { Write-Host "  PASS" } else { Write-Host "  FAIL - still tracked" ; exit 1 }
