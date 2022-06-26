import paramiko
from scp_py.scp import SCPClient
import pcw
import hydra


def createSSHClient(server, port, user, password):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(server, port, user, password)
    return client


@hydra.main(config_path="./config", config_name="config_zephir")
def main(cfg):
    ssh = createSSHClient(**cfg.ssh)
    scp = SCPClient(ssh.get_transport())
    for element in cfg.files.elements:
        scp.get(remote_path=cfg.files.remote_root + element)

    pcw.render_pc(cfg.files.elements, [], 0.0005, True, False)


if __name__ == '__main__':
    main()
