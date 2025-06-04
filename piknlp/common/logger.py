# piknlp/common/logger.py

from rich.logging import RichHandler
from sklearn.exceptions import UndefinedMetricWarning
import logging
import warnings

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(markup=True, rich_tracebacks=True)]
)

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sklearn").setLevel(logging.WARNING)
logging.getLogger("torch").setLevel(logging.WARNING)

def get_logger(name: str = "piknlp") -> logging.Logger:
    """
    이름(name)으로 로거를 가져옵니다.  
    동일한 이름의 로거가 이미 BasicConfig(RichHandler)와 함께 초기화되어 있으면,
    여기서 새로운 핸들러를 추가로 붙이지 않습니다.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = RichHandler(markup=True, rich_tracebacks=True)
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger