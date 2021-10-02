import os
import logging.config
from ast.ast_processor import AstProcessor
from ast.basic_info_listener import BasicInfoListener


if __name__ == '__main__':
    relative_path = '/resources/logging/utiltools_log.conf'
    # logging_setting_path = os.path.join(os.getcwd(),'resources/logging/utiltools_log.conf')
    logging_setting_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),  'utiltools_log.conf')
    print(logging_setting_path)
    # logging_setting_path = 'resources/logging/utiltools_log.conf'
    # logging.config.fileConfig(logging_setting_path, disable_existing_loggers=False)
    logging.config.fileConfig(logging_setting_path)
    logger = logging.getLogger(__file__)

    target_file_path = 'data/1/lecturer/BasicSalary.java'

    #★ Point 1
    ast_info = AstProcessor(logging, BasicInfoListener()).execute(target_file_path)