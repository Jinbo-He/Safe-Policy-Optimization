from harl.common.base_logger import BaseLogger


class MaMECLogger(BaseLogger):
    def get_task_name(self):
        return self.env_args["env_name"]
