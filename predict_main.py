"""
Autoia ABA Prediction System — Entry Point

Uso:
    python predict_main.py                           # dashboard grafico
    python predict_main.py --domain sports           # prediccion por consola
    python predict_main.py --domain market --symbols BTC-USD,AAPL
    python predict_main.py --api-key TU_ODDS_API_KEY # con The Odds API
    python predict_main.py --serve                   # API REST en puerto 8765
    python predict_main.py --serve --port 9000       # puerto personalizado
    python predict_main.py --no-gui --all-domains    # todos los dominios consola

Dominios disponibles:
    sports   -- Prediccion de partidos deportivos
    market   -- Prediccion de mercados financieros
    masses   -- Prediccion de tendencias de masas
    betting  -- Analisis de valor en apuestas deportivas

API REST (con --serve):
    GET  /status
    GET  /predict?domain=sports&home=X&away=Y
    GET  /predictions?n=10
    POST /data        { source, domain, data_type, payload }
    POST /mo          { domain, description, mo_type, strength }
    POST /outcome     { subject, domain, outcome }
    POST /sentiment   { texts: [...] }
    POST /webhook/register { url, domains, secret }
"""

import sys
import os
import io
import argparse
import logging
import json
import time

# Encoding seguro en Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("prediction.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("autoia.predict_main")

# ── Path ─────────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prediction.engine import PredictionEngine


def try_connect_ollama():
    """Intenta conectar con Ollama local para analisis de sentimiento via LLM."""
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from llm.orchestrator import OllamaOrchestrator
        orc = OllamaOrchestrator()
        if orc.available:
            logger.info("Ollama conectado: sentimiento via LLM activado")
            return orc
    except Exception:
        pass
    logger.info("Ollama no disponible: usando analisis por patrones de palabras")
    return None


def run_console_demo(engine: PredictionEngine, domain: str):
    """Demo por consola sin pygame."""
    print("\n" + "="*70)
    print("  AUTOIA — SISTEMA DE PREDICCION ABA")
    print(f"  Dominio: {domain.upper()}")
    print("="*70)

    if domain == "sports":
        # Setup equipos
        rm = engine.sports.add_team("Real Madrid", "La Liga")
        for r in ["W", "W", "D", "W", "W", "L", "W", "W", "D", "W"]:
            engine.sports.update_team_result("Real Madrid", r, 2, 1)
        rm.home_advantage = True
        rm.rest_days = 4

        barca = engine.sports.add_team("Barcelona", "La Liga")
        for r in ["L", "W", "W", "D", "L", "W", "W", "L", "W", "D"]:
            engine.sports.update_team_result("Barcelona", r, 1, 1)
        barca.injuries = ["Lewandowski", "Pedri"]
        barca.rest_days = 2

        # MOs
        engine.add_mo("sports", "lesion", "Lewandowski baja confirmada", "AO", "Barcelona", 0.8)
        engine.add_mo("sports", "racha", "Real Madrid 8 victorias seguidas", "EO", "Real Madrid", 0.9)

        pred = engine.predict_sports_match("Real Madrid", "Barcelona")
        _print_prediction(pred)

    elif domain == "market":
        print("\n[INFO] Obteniendo datos de mercado (puede tardar)...")
        symbols = ["BTC-USD", "^GSPC"]
        for sym in symbols:
            engine.market.add_symbol(sym)
            pred = engine.predict_market(sym)
            _print_prediction(pred)
            time.sleep(1)

    elif domain == "masses":
        texts = [
            "Bitcoin superara los 100k, todo el mundo quiere entrar, FOMO masivo",
            "Las instituciones acumulan en silencio, el halving se acerca",
            "Analistas advierten de sobrevalorizacion, posible correccion",
        ]
        for text in texts:
            engine.analyze_text_mo(text, domain="masses")
            time.sleep(0.1)
        pred = engine.predict_mass_trend("bitcoin", texts)
        _print_prediction(pred)

    elif domain == "betting":
        pred = engine.predict_betting_value(
            "Real Madrid", "Barcelona",
            home_odds=1.85, draw_odds=3.60, away_odds=4.20
        )
        _print_prediction(pred)
        print("\n" + "-"*70)
        vr = engine.betting.get_vr_schedule_analysis()
        print("\n[ALERTA VR SCHEDULE - Por que los apostadores no pueden parar]")
        for reason in vr["why_addictive"]:
            print(f"  * {reason}")
        print(f"\n[ABA Mecanismo]")
        for k, v in vr["aba_mechanism"].items():
            print(f"  {k}: {v}")

    print("\n" + "="*70)
    print("  Marco ABA aplicado:")
    print("  4-term contingency: MO -> SD -> R -> C")
    print("  Matching Law: B1/(B1+B2) = R1/(R1+R2)")
    print("  Impulso Conductual: resistencia a extincion por historial")
    print("="*70)


def _print_prediction(pred):
    """Imprime prediccion en consola con formato ABA."""
    print(f"\n{'─'*60}")
    print(f"  Sujeto:      {pred.subject}")
    print(f"  Dominio:     {pred.domain.upper()}")
    print(f"  Prediccion:  {pred.predicted_outcome.upper()}")
    print(f"  Confianza:   {pred.confidence:.0%}")
    print(f"\n  Probabilidades:")
    for outcome, prob in pred.probabilities.items():
        bar = "=" * int(prob * 30)
        print(f"    {str(outcome)[:18]:<18} {bar:<30} {prob:.1%}")
    print(f"\n  Funcion conductual: {pred.dominant_function}")
    print(f"  Ventaja conductual: {pred.behavioral_edge}")
    print(f"  Momentum:           {pred.momentum_score:.2f}/5.0")
    print(f"  Sentimiento masivo: {pred.sentiment_score:+.2f}")
    if pred.active_mos:
        print(f"\n  MOs activos:")
        for mo in pred.active_mos[:3]:
            mo_type = mo.get("mo_type", mo.get("type", "?"))
            desc = (mo.get("description", mo.get("source",
                     str(mo.get("score", "")))))
            print(f"    [{mo_type}] {str(desc)[:60]}")
    if pred.key_factors:
        print(f"\n  Factores clave:")
        for f in pred.key_factors[:4]:
            print(f"    + {f}")
    if pred.risk_factors:
        print(f"\n  Factores de riesgo:")
        for r in pred.risk_factors[:3]:
            print(f"    ! {r}")
    if pred.matching_distribution:
        print(f"\n  Distribucion Matching Law:")
        for k, v in list(pred.matching_distribution.items())[:4]:
            print(f"    {str(k)[:16]:<16} {v:.1%}")


def main():
    parser = argparse.ArgumentParser(
        description="Autoia ABA Prediction System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--domain", choices=["sports", "market", "masses", "betting"],
                        default="sports",
                        help="Dominio de prediccion")
    parser.add_argument("--symbols", default="BTC-USD,AAPL,^GSPC",
                        help="Simbolos de mercado separados por coma")
    parser.add_argument("--sport", default="football",
                        help="Deporte (football, basketball, etc.)")
    parser.add_argument("--api-key", default=None,
                        help="API key de The Odds API (opcional)")
    parser.add_argument("--no-gui", action="store_true",
                        help="Modo consola (sin pygame dashboard)")
    parser.add_argument("--demo", action="store_true",
                        help="Modo demo con datos de ejemplo")
    parser.add_argument("--all-domains", action="store_true",
                        help="Mostrar prediccion de todos los dominios por consola")
    parser.add_argument("--serve", action="store_true",
                        help="Iniciar API REST para integraciones externas")
    parser.add_argument("--port", type=int, default=8765,
                        help="Puerto de la API REST (default: 8765)")
    parser.add_argument("--no-bus", action="store_true",
                        help="Deshabilitar bus de integracion")
    args = parser.parse_args()

    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║         AUTOIA — ABA PREDICTION SYSTEM                   ║
    ║  Matching Law + MOs + Impulso Conductual + FBA           ║
    ╚══════════════════════════════════════════════════════════╝
    """)

    # Conectar Ollama (opcional)
    orchestrator = None
    if not args.no_gui:
        orchestrator = try_connect_ollama()

    # Inicializar motor
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    engine = PredictionEngine(
        orchestrator=orchestrator,
        market_symbols=symbols,
        sports_sport=args.sport,
        betting_api_key=args.api_key,
    )

    # Integration bus y API REST
    bus = None
    api = None
    if not args.no_bus:
        from integrations.bus import IntegrationBus
        bus = IntegrationBus(engine=engine, poll_interval=60.0)

    if args.serve or (not args.no_gui and not args.demo and not args.all_domains):
        if bus is not None:
            from integrations.api_server import APIServer
            api = APIServer(bus=bus, engine=engine, port=args.port)
            api.start(blocking=False)
            if bus:
                bus.start()

    # Modo consola
    if args.no_gui or args.demo or args.all_domains:
        if args.all_domains:
            for domain in ["sports", "market", "masses", "betting"]:
                run_console_demo(engine, domain)
                print()
        else:
            run_console_demo(engine, args.domain)
        return

    # Modo GUI (pygame dashboard)
    try:
        from prediction.visualization.dashboard import Dashboard
        dash = Dashboard(engine)
        logger.info("Iniciando dashboard ABA...")
        dash.run()
    except ImportError as e:
        logger.warning(f"No se puede iniciar GUI: {e}")
        logger.info("Usando modo consola...")
        run_console_demo(engine, args.domain)
    except Exception as e:
        logger.error(f"Error en dashboard: {e}")
        raise


if __name__ == "__main__":
    main()
