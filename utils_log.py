# logger_min.py
import os, csv, json, platform, socket, getpass
import psutil
from datetime import datetime
from pathlib import Path

ruta_carpeta_actual = os.getcwd()
LOG_PATH  = os.path.join(ruta_carpeta_actual, "registros_log.csv")
COLUMNS = [ "timestamp","carpeta","script","algoritmo","dataset",
    "clases_removidas","seed","n_train","n_test", "n_features","num_classes", 
    "fit_seconds","pred_seconds","ms_per_sample", "OA","F1_macro","system_info"
]

try:
    import psutil
except ImportError:
    psutil = None

def _system_info():
    info = {
        "host": socket.gethostname(),
        "os": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "cpu_count_logical": os.cpu_count(),   # núcleos lógicos
        "cpu_count_physical": psutil.cpu_count(logical=False) if psutil else "",
        "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2) if psutil else ""
    }
    # versiones de librerías
    for m in ["numpy","pandas","sklearn","torch","torchvision","timm","xgboost"]:
        try:
            mod = __import__(m)
            info[m] = getattr(mod, "__version__", "")
        except Exception:
            info[m] = ""
    # GPU info (si tienes torch)
    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_version"] = getattr(torch.version, "cuda", "")
        info["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
    except Exception:
        info["cuda_available"] = False
        info["cuda_version"] = ""
        info["gpu_name"] = ""
    return info

def log_row(**kwargs):
    """Escribe una fila en CSV; si algo falla, no detiene tu experimento."""
    try:
        # deducir carpeta si no se pasa explícitamente
        carpeta = kwargs.get("carpeta")
        if not carpeta:
            script = kwargs.get("script", "")
            if script and "/" in script:
                carpeta = Path(script).parent.name or "."
            else:
                carpeta = Path.cwd().name
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "script": kwargs.get("script",""),
            "carpeta": carpeta,
            "algoritmo": kwargs.get("algoritmo",""),
            "dataset": kwargs.get("dataset",""),
            "clases_removidas": ",".join(map(str, kwargs.get("clases_removidas", []))),
            "seed": kwargs.get("seed",""),
            "n_train": kwargs.get("n_train",""),
            "n_test": kwargs.get("n_test",""),
            "n_features": kwargs.get("n_features",""),
            "num_classes": kwargs.get("num_classes",""),
            "fit_seconds": round(kwargs.get("fit_seconds",0.0), 4),
            "pred_seconds": round(kwargs.get("pred_seconds",0.0), 4),
            "ms_per_sample": round(kwargs.get("ms_per_sample",0.0), 3),
            "OA": round(kwargs.get("OA",0.0), 4) if kwargs.get("OA") is not None else "",
            "F1_macro": round(kwargs.get("F1_macro",0.0), 4) if kwargs.get("F1_macro") is not None else "",
            "system_info": json.dumps(_system_info(), ensure_ascii=False)
        }
        # crear archivo con cabecera si no existe
        new_file = not os.path.exists(LOG_PATH)
        with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            if new_file:
                w.writeheader()
            w.writerow(row)
    except Exception as e:
        # no detiene el flujo de entrenamiento
        print(f"[LOG WARNING] no se pudo escribir log: {e}")