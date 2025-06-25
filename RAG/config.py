import os
from typing import Final

COOKIE_SECRET_KEY: Final = os.getenv("COOKIE_SECRET_KEY", "huy")