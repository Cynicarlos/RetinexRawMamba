
class CustomLogger:
    def __init__(self, log_file_path='training_log.txt'):
        self.log_file_path = log_file_path

    def info(self, message):
        self._print_and_write_to_file('INFO', message)

    def warning(self, message):
        self._print_and_write_to_file('WARNING', message)

    def error(self, message):
        self._print_and_write_to_file('ERROR', message)

    def _print_and_write_to_file(self, level, message):
        log_message = f'{level} - {message}'

        # 打印到控制台
        print(log_message)

        # 写入到文件
        with open(self.log_file_path, 'a') as log_file:
            log_file.write(log_message + '\n')