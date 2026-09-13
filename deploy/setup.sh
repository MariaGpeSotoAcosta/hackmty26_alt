#!/usr/bin/env bash
# Corre esto EN el servidor Vultr (Ubuntu 22.04/24.04), como root o con sudo.
# Deja el detector corriendo como servicio systemd en el puerto 8000, con
# reinicio automático si truena.
#
# Uso:
#   git clone <url-del-repo> hackmty26_alt
#   cd hackmty26_alt
#   sudo bash deploy/setup.sh

set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE_NAME="altur-detector"
PORT="${PORT:-8080}"

echo "==> Instalando Python y dependencias del sistema"
apt-get update -y
apt-get install -y python3 python3-venv python3-pip

echo "==> Creando entorno virtual en $APP_DIR/.venv"
python3 -m venv "$APP_DIR/.venv"
"$APP_DIR/.venv/bin/pip" install --upgrade pip
"$APP_DIR/.venv/bin/pip" install -r "$APP_DIR/requirements.txt"

echo "==> Verificando que el modelo entrenado este presente"
if [ ! -f "$APP_DIR/models/dialogue_model.joblib" ]; then
  echo "ERROR: falta $APP_DIR/models/dialogue_model.joblib"
  echo "Ese archivo debe venir en el repo (esta trackeado en git). Si falta, hagan:"
  echo "  python -m detector.train --from-wav --dual-view"
  echo "en una maquina con audio/ y turns/ disponibles, y suban el .joblib resultante."
  exit 1
fi

echo "==> Instalando servicio systemd"
cat > "/etc/systemd/system/${SERVICE_NAME}.service" <<EOF
[Unit]
Description=Altur synthetic-caller detector
After=network.target

[Service]
Type=simple
WorkingDirectory=${APP_DIR}
ExecStart=${APP_DIR}/.venv/bin/python -m uvicorn detector.app:app --host 0.0.0.0 --port ${PORT}
Restart=always
RestartSec=3
User=root

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable "${SERVICE_NAME}"
systemctl restart "${SERVICE_NAME}"

echo "==> Abriendo el puerto ${PORT} en el firewall (si ufw esta activo)"
if command -v ufw >/dev/null 2>&1; then
  ufw allow "${PORT}/tcp" || true
fi

sleep 2
echo "==> Estado del servicio:"
systemctl status "${SERVICE_NAME}" --no-pager || true

echo
echo "==> Prueba local en el servidor:"
curl -s "http://127.0.0.1:${PORT}/health" || echo "(el servicio aun no responde, revisa: journalctl -u ${SERVICE_NAME} -f)"
echo
echo "Listo. Desde fuera, prueben: curl http://<IP_PUBLICA>:${PORT}/health"
