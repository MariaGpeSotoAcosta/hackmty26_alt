# Dashboard: registrar llamadas en Tiger Data

`/detect` puede registrar cada veredicto (features, confianza, latencia, por qué decidió eso) en una base de datos, y verlo en `/dashboard`. Es opcional: si no está configurado, `/detect` funciona exactamente igual, solo no queda nada guardado.

## 1. Crear la base de datos (una sola vez, en tigerdata.com)

1. Cuenta en [tigerdata.com](https://www.tigerdata.com) (o [console.cloud.timescale.com](https://console.cloud.timescale.com), es el mismo producto).
2. **Create Service** → Postgres con Timescale habilitado. El plan gratis/más chico alcanza de sobra — este dashboard hace pocos inserts por llamada, nada de volumen alto.
3. Copien el **connection string** que les dan al crear el servicio, se ve así:
   ```
   postgresql://tsdbadmin:<password>@<host>.tsdb.cloud.timescale.com:<puerto>/tsdb?sslmode=require
   ```

## 2. Configurar el servidor para usarla

En el servidor de Vultr (o donde corra `uvicorn`), exporten la variable de entorno antes de arrancar:

```bash
export TIGER_DATA_URL="postgresql://tsdbadmin:<password>@<host>.tsdb.cloud.timescale.com:<puerto>/tsdb?sslmode=require"
```

Si usan el servicio systemd de `deploy/setup.sh`, agreguen esa línea al archivo de servicio (`/etc/systemd/system/altur-detector.service`), bajo `[Service]`:

```ini
Environment=TIGER_DATA_URL=postgresql://tsdbadmin:<password>@<host>.tsdb.cloud.timescale.com:<puerto>/tsdb?sslmode=require
```

Luego:
```bash
sudo systemctl daemon-reload
sudo systemctl restart altur-detector
```

La tabla (`calls`, convertida a hypertable) se crea sola en el primer `/detect` que llegue — no hay que correr ningún script de setup de base de datos a mano.

## 3. Verificar

```bash
curl http://<IP_o_localhost>:8000/dashboard/data
```

Debe decir `"enabled": true`. Después de mandar un par de llamadas a `/detect`, entren a `http://<IP>:8000/dashboard` en el navegador — deben aparecer en la tabla, con el desglose de features al hacer clic en una fila.

## Qué guarda

Por cada llamada a `/detect`: hora, veredicto, confianza, si se usó desempate, si hubo early-exit, latencia en ms, las 22 features de diálogo, y la contribución de cada feature a la decisión (peso × valor estandarizado — de dónde salió el veredicto, no solo cuál fue).

No se identifica al caller real de ningún modo — no hay audio ni transcripción guardados, solo los números de comportamiento ya derivados.
