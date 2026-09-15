# Master, split in two

`out/space-edit.mp4` (CRF 29, 72 MB) is the version you can download in one click.
This folder holds the **CRF 23 master** — 1080×1920, 30 fps, 13.5 Mbit/s, 146 MB —
split into two parts because GitHub rejects single files over 100 MB.

Download both `.bin` files into the same folder, then join them:

```bash
# Linux / macOS
cat space-edit-master.part00.bin space-edit-master.part01.bin > space-edit-master.mp4
md5sum space-edit-master.mp4      # c7c96a1e946ff8cd46a34579b10c92b2
```

```bat
:: Windows (cmd)
copy /b space-edit-master.part00.bin + space-edit-master.part01.bin space-edit-master.mp4
```

```powershell
# Windows (PowerShell)
$out = [IO.File]::Create("space-edit-master.mp4")
foreach ($p in "space-edit-master.part00.bin","space-edit-master.part01.bin") {
  $in = [IO.File]::OpenRead((Resolve-Path $p)); $in.CopyTo($out); $in.Close()
}
$out.Close()
Get-FileHash space-edit-master.mp4 -Algorithm MD5   # C7C96A1E946FF8CD46A34579B10C92B2
```

The joined file is bit-identical to the render (verified here).
