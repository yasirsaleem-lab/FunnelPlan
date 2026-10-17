"""
Run logs for the long scripts (03, 04).

setup_logging("02_generate_tsr_labels", CONFIG) adds a file handler next to
the console, so everything that goes through `logging` (block starts, cache
hits, "encoded X/N", NaN repair counts, TSR per depth, errors) is kept in

    <RESULTS_ROOT>/logs/<script>_<run id>.log      (results_full/logs in full mode)

The parent process sets FUNNELPLAN_RUN_ID; every child process it starts
inherits it and appends to the same file, so one run is one log. Uncaught
exceptions are written to the log before the process exits.
Note: tqdm progress bars are not logged (they are console only).
"""

import logging
import os
import platform
import sys
import time

from config import settings

FMT = "%(asctime)s [pid %(process)d] %(levelname)s %(name)s: %(message)s"


def setup_logging(script: str, config: dict = None):
    run_id = os.environ.setdefault("FUNNELPLAN_RUN_ID", time.strftime("%Y%m%d_%H%M%S"))
    log_dir = settings.RESULTS_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{script}_{run_id}.log"

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
               for h in root.handlers):
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter(FMT))
        root.addHandler(sh)
    if not any(isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == str(path)
               for h in root.handlers):
        fh = logging.FileHandler(path, mode="a", encoding="utf-8")
        fh.setFormatter(logging.Formatter(FMT))
        root.addHandler(fh)
    logging.captureWarnings(True)

    def _hook(exc_type, exc, tb):
        logging.getLogger(script).critical("Uncaught exception", exc_info=(exc_type, exc, tb))
        sys.__excepthook__(exc_type, exc, tb)
    sys.excepthook = _hook

    log = logging.getLogger(script)
    log.info(f"=== {script} | argv {sys.argv[1:]} | corpus mode {settings.CORPUS_MODE} | log {path}")
    if config is not None and "--child" not in sys.argv:
        log.info(f"CONFIG {config}")
        try:
            import torch, transformers
            gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
            log.info(f"python {platform.python_version()} | torch {torch.__version__} | "
                     f"transformers {transformers.__version__} | gpu {gpu}")
        except Exception as e:
            log.info(f"version info unavailable: {e}")
    return path
