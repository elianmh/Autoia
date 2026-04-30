"""
Entidades del mundo: recursos, obstáculos, partículas visuales.
"""

import math
import random
from typing import List, Tuple


class Resource:
    """
    Fuente de energía recolectable.
    Se regenera con el tiempo (conservación de energía del mundo).
    """

    def __init__(self, resource_id: int, x: float, y: float,
                 max_energy: float = 1.0, regen_rate: float = 0.05):
        self.resource_id  = resource_id
        self.x            = float(x)
        self.y            = float(y)
        self.energy_value = max_energy
        self.max_energy   = max_energy
        self.regen_rate   = regen_rate
        self.radius       = 7
        self.active       = True
        self.glow_phase   = random.uniform(0, 2*math.pi)

    def update(self, dt: float):
        # Regeneración lenta (conservación: la energía del mundo se conserva)
        self.energy_value = min(self.max_energy, self.energy_value + self.regen_rate * dt)
        self.active = self.energy_value > 0.01
        self.glow_phase += 2.0 * dt

    def collect(self, amount: float) -> float:
        """Un agente recolecta energía. Retorna lo que realmente tomó."""
        taken = min(amount, self.energy_value)
        self.energy_value -= taken
        self.active = self.energy_value > 0.01
        return taken

    @property
    def color(self) -> Tuple[int, int, int]:
        ratio = self.energy_value / max(self.max_energy, 0.01)
        r = int(50 + 205 * (1-ratio))
        g = int(200 * ratio)
        b = 50
        return (r, g, b)

    @property
    def glow_intensity(self) -> float:
        return 0.5 + 0.5 * math.sin(self.glow_phase)


class Particle:
    """Partícula visual (efectos de impacto, energía, pensamiento)."""

    def __init__(self, x: float, y: float, vx: float, vy: float,
                 color: Tuple[int,int,int], lifetime: float = 1.0,
                 size: float = 3.0):
        self.x        = x
        self.y        = y
        self.vx       = vx
        self.vy       = vy
        self.color    = color
        self.lifetime = lifetime
        self.max_life = lifetime
        self.size     = size
        self.alive    = True

    def update(self, dt: float):
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.vx *= 0.92
        self.vy *= 0.92
        self.lifetime -= dt
        if self.lifetime <= 0:
            self.alive = False

    @property
    def alpha(self) -> int:
        return int(255 * max(0, self.lifetime / self.max_life))

    @property
    def current_size(self) -> float:
        return self.size * max(0, self.lifetime / self.max_life)


class ParticleSystem:
    """Gestor de partículas visuales."""

    def __init__(self):
        self.particles: List[Particle] = []

    def update(self, dt: float):
        for p in self.particles:
            p.update(dt)
        self.particles = [p for p in self.particles if p.alive]

    def emit_energy(self, x: float, y: float, color=(255, 200, 50)):
        """Partículas de recogida de energía."""
        for _ in range(6):
            angle = random.uniform(0, 2*math.pi)
            speed = random.uniform(30, 80)
            self.particles.append(Particle(
                x, y,
                math.cos(angle)*speed, math.sin(angle)*speed,
                color, lifetime=0.6, size=3
            ))

    def emit_collision(self, x: float, y: float):
        """Partículas de colisión."""
        for _ in range(8):
            angle = random.uniform(0, 2*math.pi)
            speed = random.uniform(50, 120)
            self.particles.append(Particle(
                x, y,
                math.cos(angle)*speed, math.sin(angle)*speed,
                (255, 255, 255), lifetime=0.4, size=2
            ))

    def emit_data(self, x: float, y: float):
        """Partículas de absorción de datos (para Autoia)."""
        for _ in range(4):
            angle = random.uniform(0, 2*math.pi)
            speed = random.uniform(20, 60)
            self.particles.append(Particle(
                x, y,
                math.cos(angle)*speed, math.sin(angle)*speed,
                (80, 220, 220), lifetime=1.0, size=2.5
            ))

    def emit_death(self, x: float, y: float, color=(200, 50, 50)):
        """Partículas de muerte de agente."""
        for _ in range(15):
            angle = random.uniform(0, 2*math.pi)
            speed = random.uniform(40, 150)
            self.particles.append(Particle(
                x, y,
                math.cos(angle)*speed, math.sin(angle)*speed,
                color, lifetime=random.uniform(0.5, 1.5), size=random.uniform(2, 5)
            ))

    def emit_thought(self, x: float, y: float):
        """Partículas de pensamiento de Autoia."""
        for _ in range(3):
            self.particles.append(Particle(
                x + random.uniform(-5, 5),
                y + random.uniform(-5, 5),
                random.uniform(-10, 10),
                random.uniform(-30, -10),
                (180, 100, 255),
                lifetime=1.5, size=2
            ))
