"""
Plugin de Webhook saliente.

Cuando el motor genera una prediccion, la envia automaticamente
a la URL registrada via POST HTTP.

Uso:
    plugin = WebhookPlugin(
        url="https://mi-app.com/autoia-webhook",
        domains=["sports", "market"],
        secret="mi_secreto",  # para verificar autenticidad
    )
    bus.register(plugin)
"""

import json
import time
import hmac
import hashlib
import logging
import urllib.request
import urllib.error
from typing import List, Dict

from ..base_plugin import BasePlugin, DataPoint, PredictionResult

logger = logging.getLogger("autoia.integrations.webhook")


class WebhookPlugin(BasePlugin):
    """
    Plugin que reenvia predicciones a una URL via HTTP POST.
    No hace fetch() de datos — solo push de predicciones salientes.
    """

    def __init__(self, url: str, domains: List[str] = None,
                 secret: str = "", timeout: int = 5):
        name = f"webhook_{url.split('//')[-1].split('/')[0].replace('.', '_')}"
        super().__init__(config={"url": url, "secret": secret, "timeout": timeout})
        self.name              = name[:40]
        self.description       = f"Webhook -> {url}"
        self.supported_domains = domains or []
        self._url              = url
        self._secret           = secret
        self._timeout          = timeout

    def fetch(self) -> List[DataPoint]:
        """Webhook no hace fetch de datos."""
        return []

    def connect(self) -> bool:
        """Verifica que la URL es alcanzable con un ping."""
        try:
            req = urllib.request.Request(
                self._url, method="HEAD",
                headers={"User-Agent": "Autoia-ABA/1.0"},
            )
            with urllib.request.urlopen(req, timeout=self._timeout):
                pass
            self.connected = True
            return True
        except Exception:
            # No bloqueamos si el ping falla — puede estar OK para POST
            self.connected = True
            return True

    def push_prediction(self, result: PredictionResult) -> bool:
        """Envia prediccion al webhook."""
        payload = result.to_dict()
        payload["autoia_version"] = "1.0"

        body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")

        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "User-Agent":   "Autoia-ABA/1.0",
            "X-Autoia-Timestamp": str(int(time.time())),
        }

        # Firma HMAC si hay secreto
        if self._secret:
            sig = hmac.new(
                self._secret.encode(), body, hashlib.sha256
            ).hexdigest()
            headers["X-Autoia-Signature"] = f"sha256={sig}"

        try:
            req = urllib.request.Request(
                self._url, data=body, headers=headers, method="POST"
            )
            with urllib.request.urlopen(req, timeout=self._timeout):
                pass
            self._record_push()
            logger.debug(f"Webhook enviado: {result.domain}/{result.prediction} -> {self._url}")
            return True
        except urllib.error.HTTPError as e:
            logger.warning(f"Webhook HTTP error {e.code}: {self._url}")
            self._record_error()
            return False
        except Exception as e:
            logger.warning(f"Webhook error: {e}")
            self._record_error()
            return False
