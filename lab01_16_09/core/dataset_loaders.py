FILE_PATH = "C:\\Users\\DoubleDD\\Downloads\\UD_English-EWT\\UD_English-EWT\\en_ewt-ud-train.txt"

def load_raw_text_data_from(path: str = FILE_PATH):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
        
    return content