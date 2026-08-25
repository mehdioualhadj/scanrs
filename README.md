# scanrs

Turns photos / screenshots of documents into searchable PDFs.
Built for Windows 11 IoT LTSC workstations; no admin required.

## Pipeline
1. multi-page split (separator detection)
2. page detection (corners / full-frame)
3. homography warp
4. curvature-gated TPS/Wendland dewarp (text-line baselines as landmarks)
5. six-way enhancement candidates
6. Tesseract OCR + confidence-based garbage-line filter
7. searchable PDF (per-line Helvetica-metrics text layer) + combined PDF
8. optional GUI landmark editor (--gui)

## Usage
    scanrs <image> [--out DIR] [--gui] [--debug] [--no-dewarp] [--precurl] [--open]
    right-click an image -> "Scan with scanrs"

## Selftest
    scanrs --selftest13     # synthetic curl certificate (24 px bow -> <2)

## Install / package
    scripts\pack.ps1                              # assemble dist\scanrs-workstation (prebuilt; no compiler on target)
    dist\scanrs-workstation\install.ps1           # per-user install + context menu
    see scripts\AUTOUNATTEND_NOTE.txt             # fresh-VM provisioning
