import logging

class ColorEmojiFormatter(logging.Formatter):
    # ANSI Color Codes
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    RESET = "\033[0m"

    # Map log levels to Emojis and Colors
    FORMATS = {
        logging.DEBUG: YELLOW + "⚙️ [EXECUTING] " + RESET + "%(asctime)s - %(message)s",
        logging.INFO: BLUE + "ℹ️ [INFO] " + RESET + "%(asctime)s - %(message)s",
        logging.WARNING: YELLOW + "⚠️ [WARNING] " + RESET + "%(asctime)s - %(message)s",
        logging.ERROR: RED + "❌ [ERROR] " + RESET + "%(asctime)s - %(message)s",
        # We can hack a custom "SUCCESS" level by using level 25
        25: GREEN + "✅ [SUCCESS] " + RESET + "%(asctime)s - %(message)s"
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno, self.FORMATS[logging.INFO])
        formatter = logging.Formatter(log_fmt, datefmt='%H:%M:%S')
        return formatter.format(record)

def get_logger(name: str):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Prevent duplicate logs if called multiple times
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(ColorEmojiFormatter())
        logger.addHandler(ch)
        
    # Add a custom success method to the logger
    logging.addLevelName(25, "SUCCESS")
    def success(message, *args, **kws):
        if logger.isEnabledFor(25):
            logger._log(25, message, args, **kws)
    logger.success = success
    
    return logger
