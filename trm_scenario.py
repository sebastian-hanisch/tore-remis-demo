"""Szenario: Reglerwerte (Settings) und die daraus erzeugten Spiele."""

from dataclasses import dataclass

import trm_constants as C
import trm_goals as G


@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    delta: float = C.DEFAULT_DELTA
    k: int = C.DEFAULT_K
    g: float = C.DEFAULT_G
    h: float = C.DEFAULT_H
    home: str = "alternierend"
    disp: bool = False             # Tore überstreut (negativ binomial)
    r: float = C.DEFAULT_R         # Form, nur bei disp
    fit_h: bool = True             # Heimvorteil mitschätzen
    seed: int = C.DEFAULT_SEED

    @property
    def shape(self):
        """Form der negativen Binomialverteilung oder None (Poisson)."""
        return self.r if self.disp else None


def make_games(s: Settings) -> G.Games:
    """Deterministisch aus den Reglerwerten: gleiche Eingabe, gleiche Spiele."""
    return G.simulate_games(s.n, s.delta, s.k, s.g, s.h, s.home, s.shape, s.seed)
