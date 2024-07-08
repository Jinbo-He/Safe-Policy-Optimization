from harl.common.base_logger import BaseLogger


class MAMECLogger(BaseLogger):
    def get_task_name(self):
        return self.env_args["env_name"]
