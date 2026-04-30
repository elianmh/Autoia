"""
Capa de integracion con apps externas de analisis.

Arquitectura:
- IntegrationBus: bus central que conecta plugins con el motor ABA
- BasePlugin: clase base para cualquier app externa
- REST API: servidor HTTP para recibir datos en tiempo real
- Webhooks: notificaciones salientes a apps suscritas

Para conectar una nueva app:
    1. Crear un plugin en integrations/plugins/mi_app.py
    2. Heredar de BasePlugin
    3. Registrar en IntegrationBus
"""
