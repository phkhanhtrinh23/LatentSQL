#!/usr/bin/env bash
# Download the six papers this project builds on into pdfs/.
# They are not committed (copyright + repo size); this fetches them on demand.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p pdfs
fetch () { # url, filename
  if [ -f "pdfs/$2" ]; then echo "have  $2"; else
    echo "get   $2"; curl -sSL --max-time 180 -A "Mozilla/5.0" -o "pdfs/$2" "$1"
  fi
}
# the two latent-agent papers (the method)
fetch "https://aclanthology.org/2026.acl-long.709.pdf" "2026.acl-long.709.pdf"   # IMAD / Latent Agents
fetch "https://arxiv.org/pdf/2511.20639"              "2511.20639v4.pdf"         # LatentMAS
# the four multi-agent Text-to-SQL systems (the teacher)
fetch "https://arxiv.org/pdf/2304.11015"              "2304.11015.pdf"           # DIN-SQL
fetch "https://arxiv.org/pdf/2308.15363"              "2308.15363.pdf"           # DAIL-SQL
fetch "https://arxiv.org/pdf/2405.16755"              "2405.16755.pdf"           # CHESS
fetch "https://arxiv.org/pdf/2410.01943"              "2410.01943.pdf"           # CHASE-SQL
echo; echo "done -> pdfs/"
