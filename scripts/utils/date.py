from datetime import datetime


def get_formatted_date():
    now = datetime.now()
    formatted_date = now.strftime("%d%m%y_%H-%M-%S")
    return formatted_date
