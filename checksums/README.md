# Release checksums

Generate a fresh checksum manifest from the release root with:

```powershell
$root = (Get-Location).Path
Get-ChildItem -File -Recurse |
  Where-Object { $_.FullName -notmatch '\\(__pycache__|outputs?)\\' -and $_.Name -ne 'SHA256SUMS.txt' } |
  Get-FileHash -Algorithm SHA256 |
  Sort-Object Path |
  ForEach-Object { "{0}  {1}" -f $_.Hash.ToLowerInvariant(), $_.Path.Substring($root.Length + 1).Replace('\', '/') }
```

The most important model and data assets are also covered by the original locked-package `MANIFEST.json` and the deployment manifests.
