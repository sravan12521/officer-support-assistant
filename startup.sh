
#!/bin/bash
set -e

python -m pip install --upgrade pip
pip install -r requirements.txt

exec python -m streamlit run /home/site/wwwroot/app.py \
  --server.port "${PORT:-8000}" \
  --server.address 0.0.0.0 \
  --server.headless true \
  --server.enableCORS false \
  --server.enableXsrfProtection false