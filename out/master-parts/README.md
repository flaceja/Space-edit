# Master, split in two

`out/space-edit.mp4` (CRF 29, 79 MB) is the version you can download in one click.
This folder holds the **CRF 23 master** — 1080×1920, 30 fps, 15.2 Mbit/s, 164 MB —
split into two parts because GitHub rejects single files over 100 MB.

Download both `.bin` files into the same folder, then join them:

```bash
# Linux / macOS
cat space-edit-master.part00.bin space-edit-master.part01.bin > space-edit-master.mp4
md5sum space-edit-master.mp4      # b38c4cc1eeee846ca0a0929d38220b4e
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
Get-FileHash space-edit-master.mp4 -Algorithm MD5   # B38C4CC1EEEE846CA0A0929D38220B4E
```

The joined file is bit-identical to the render (verified here).
