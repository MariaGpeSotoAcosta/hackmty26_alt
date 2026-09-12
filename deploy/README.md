# Deploy en Vultr

Pasos para dejar `/detect` público y alcanzable durante el juicio.

## 1. Crear la instancia (una sola vez, manual en vultr.com)

1. Cuenta en [vultr.com](https://www.vultr.com) con un método de pago.
2. **Deploy New Server** → **Cloud Compute (Shared CPU)**.
3. Ubicación: la más cercana al lugar del evento (menor latencia).
4. Imagen: **Ubuntu 22.04 LTS x64**.
5. Plan: el más barato (1 vCPU / 1GB RAM) sobra — el modelo es una regresión logística sobre 21 números, no hay red neuronal pesada.
6. En "SSH Keys": suban su llave pública (o generen una ahí). Sin esto, solo van a poder entrar con contraseña por email, más lento y menos seguro.
7. Deploy. En 1-2 minutos les dan una **IP pública**.

## 2. Conectarse y subir el código

```bash
ssh root@<IP_PUBLICA>
```

En el servidor:

```bash
apt-get update -y && apt-get install -y git
git clone https://github.com/MariaGpeSotoAcosta/hackmty26_alt.git
cd hackmty26_alt
sudo bash deploy/setup.sh
```

`setup.sh` instala Python, crea el entorno virtual, instala dependencias, y deja el detector corriendo como servicio systemd (`altur-detector`) en el puerto 8000, con reinicio automático si truena.

## 3. Confirmar que quedó público

Desde tu propia laptop (no desde el servidor):

```bash
curl http://<IP_PUBLICA>:8000/health
python -m detector.test_endpoint call_0e1e2f29bfdc --url http://<IP_PUBLICA>:8000/detect
```

Si `/health` no responde desde afuera pero sí desde adentro del servidor (`curl http://127.0.0.1:8000/health`), casi siempre es el firewall de Vultr (no el de Ubuntu) — en el panel de Vultr, la instancia tiene una pestaña **Firewall**: agreguen una regla que permita entrada TCP al puerto 8000 desde cualquier IP (`0.0.0.0/0`).

## 4. Actualizar cuando cambien código o el modelo

```bash
ssh root@<IP_PUBLICA>
cd hackmty26_alt
git pull
sudo systemctl restart altur-detector
```

No hace falta correr `setup.sh` de nuevo salvo que cambien `requirements.txt`.

## 5. Antes del juicio, el día de

- Confirmar `curl http://<IP_PUBLICA>:8000/health` desde un celular con datos (no wifi del venue) para descartar que el problema sea la red del lugar.
- Dejar la IP y el puerto anotados en el README principal para que cualquiera del equipo los tenga a la mano.
- Si el servicio se cae: `sudo systemctl status altur-detector` y `sudo journalctl -u altur-detector -n 50` en el servidor para ver el error.
