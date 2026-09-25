#!/usr/bin/env bash
# Downloads the SIL-OFL calligraphy fonts used by the video into ./fonts/
# (Ma Shan Zheng 马善政楷书, LXGW WenKai TC 霞鹜文楷, Noto Serif SC Black for the seals).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p fonts
fetch() {  # $1 = Google Fonts css2 family spec, $2 = output name
  local url
  url=$(curl -fsS "https://fonts.googleapis.com/css2?family=$1" | grep -o 'https://[^)]*\.ttf' | head -1)
  curl -fsSL -o "fonts/$2.ttf" "$url"
  echo "fonts/$2.ttf"
}
fetch "Ma+Shan+Zheng" MaShanZheng
fetch "LXGW+WenKai+TC" LXGWWenKaiTC
fetch "Noto+Serif+SC:wght@900" NotoSerifSCwght900
