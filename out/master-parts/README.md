# Master, split in two

`out/space-edit.mp4` (CRF 30, 75 MB) is the version you can download in one click.
This folder holds the **CRF 23 master** — 1080×1920, 30 fps, 17.3 Mbit/s, 187 MB —
split into two parts because GitHub rejects single files over 100 MB.

Download both `.bin` files into the same folder, then join them:

```bash
# Linux / macOS
cat space-edit-master.part00.bin space-edit-master.part01.bin > space-edit-master.mp4
md5sum space-edit-master.mp4      # 775e40d6a7fb8b0a80c23360c4741a6a
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
Get-FileHash space-edit-master.mp4 -Algorithm MD5   # 775E40D6A7FB8B0A80C23360C4741A6A
```

The joined file is bit-identical to the render (verified here).
