"""
Punto de entrada del Mundo Visual de Autoia.
Inicia la simulación gráfica con pygame.

Uso:
    python world_main.py                    # Mundo solo (sin LLM)
    python world_main.py --with-llm         # Mundo + LLM integrado
    python world_main.py --seed 123         # Semilla específica del mundo
    python world_main.py --topic "física"   # Tema de aprendizaje
"""

import sys
import os
import logging
import argparse
from pathlib import Path

# Asegurar que el directorio raíz esté en el path
sys.path.insert(0, str(Path(__file__).parent))
Path("logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/world.log", mode="a"),
    ],
)
logger = logging.getLogger("autoia.world_main")


def launch_world(seed: int = 42, with_llm: bool = False,
                 topic: str = None, headless: bool = False):
    """
    Lanza el mundo visual de Autoia.

    Args:
        seed:      Semilla del generador de mundo
        with_llm:  Integrar el sistema LLM de Autoia
        topic:     Tema de aprendizaje (override de config.py)
        headless:  Modo sin ventana (solo simulación, para tests)
    """
    from world.world_sim import WorldSimulation
    from world.agents.autoia_agent import AutoiaWorldAgent

    # ── Inicializar mundo ──────────────────────────────────────────────────
    logger.info(f"Creando mundo (seed={seed})...")
    world = WorldSimulation(seed=seed)

    # ── Sistema LLM (opcional) ─────────────────────────────────────────────
    llm_system = None
    if with_llm:
        try:
            logger.info("Inicializando sistema LLM Autoia...")
            from config import CONFIG
            if topic:
                CONFIG.learning.topic = topic

            from main import AutoiaSystem
            llm_system = AutoiaSystem(CONFIG)
            llm_system.initialize()

            # Bootstrap tokenizer si es necesario
            if llm_system.tokenizer._tokenizer is None:
                logger.info("Entrenando tokenizer con textos de bootstrap...")
                llm_system.bootstrap_tokenizer()

            logger.info("Sistema LLM listo.")
        except ImportError as e:
            logger.warning(f"No se pudo cargar el sistema LLM: {e}")
            logger.warning("Continuando sin LLM (solo modo visual).")
            llm_system = None
        except Exception as e:
            logger.error(f"Error inicializando LLM: {e}")
            llm_system = None

    # ── Crear agente Autoia ────────────────────────────────────────────────
    spawn_x, spawn_y = world.terrain.find_walkable_spawn()
    # Spawnar cerca del centro del mapa
    spawn_x = world.pixel_w / 2 + (spawn_x - world.pixel_w/2) * 0.3
    spawn_y = world.pixel_h / 2 + (spawn_y - world.pixel_h/2) * 0.3
    # Asegurar que sea transitable
    if not world.terrain.is_walkable_at(spawn_x, spawn_y):
        spawn_x, spawn_y = world.terrain.find_walkable_spawn()

    autoia = AutoiaWorldAgent(
        agent_id=99,
        x=spawn_x, y=spawn_y,
        terrain_grid=world.terrain,
        llm_system=llm_system,
    )
    world.add_autoia(autoia)

    logger.info(f"Autoia creada en ({spawn_x:.0f}, {spawn_y:.0f})")

    if headless:
        logger.info("Modo headless: simulando sin ventana...")
        _run_headless(world, autoia)
        return

    # ── Lanzar visualización pygame ────────────────────────────────────────
    try:
        from world.renderer.app import WorldApp
        logger.info("Iniciando interfaz gráfica...")
        app = WorldApp(world, autoia_agent=autoia)
        app.run()
    except ImportError as e:
        logger.error(f"pygame no disponible: {e}")
        logger.info("Instala pygame: pip install pygame")
        logger.info("Continuando en modo headless...")
        _run_headless(world, autoia)
    except Exception as e:
        logger.exception(f"Error en la aplicación: {e}")
        raise


def _run_headless(world, autoia, duration: float = 60.0):
    """Modo headless: simula el mundo sin interfaz gráfica."""
    import time
    logger.info(f"Simulando {duration}s en modo headless...")
    dt = 1/60.0
    elapsed = 0.0
    t0 = time.time()

    while elapsed < duration:
        world.step(dt)
        elapsed += dt

        if int(elapsed) % 10 == 0 and elapsed % 10 < dt:
            stats = world.get_world_state_summary()
            logger.info(
                f"t={elapsed:.0f}s | Agentes: {stats['agents_alive']} | "
                f"{'Día' if stats['is_day'] else 'Noche'} | "
                f"Autoia energía: {autoia.energy*100:.0f}% | "
                f"Datos: {autoia.data_collected:.2f}"
            )

    logger.info("Simulación headless completada.")
    logger.info(f"Autoia: {len(autoia.observations)} observaciones acumuladas")


def main():
    parser = argparse.ArgumentParser(
        description="Autoia World — Mundo visual de IAs con leyes físicas"
    )
    parser.add_argument("--seed", type=int, default=42,
                        help="Semilla del mundo (default: 42)")
    parser.add_argument("--with-llm", action="store_true",
                        help="Integrar sistema LLM (requiere PyTorch)")
    parser.add_argument("--topic", type=str, default=None,
                        help="Tema de aprendizaje del LLM")
    parser.add_argument("--headless", action="store_true",
                        help="Modo sin interfaz gráfica (para tests)")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  AUTOIA WORLD — Mundo de IAs con Leyes Físicas")
    print("="*60)
    print(f"  Semilla del mundo: {args.seed}")
    print(f"  Modo LLM: {'activado' if args.with_llm else 'desactivado'}")
    print(f"  Interfaz: {'headless' if args.headless else 'pygame GUI'}")
    if args.topic:
        print(f"  Tema: {args.topic}")
    print("="*60)
    print()
    print("  Controles:")
    print("    WASD / Flechas  — Mover cámara")
    print("    +/-             — Zoom")
    print("    F               — Seguir a Autoia")
    print("    ESPACIO         — Pausar")
    print("    TAB / 1-4       — Cambiar panel")
    print("    </>             — Velocidad simulación")
    print("    ESC             — Salir")
    print("="*60 + "\n")

    launch_world(
        seed=args.seed,
        with_llm=args.with_llm,
        topic=args.topic,
        headless=args.headless,
    )


if __name__ == "__main__":
    main()
