import paramiko
from scp import SCPClient
import pcw
import yaml


def createSSHClient(server, user, port=22, password=None):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(server, port, user, password)
    return client



def main(config):
    with open(config, 'r') as stream:
        cfg = yaml.safe_load(stream)
        ssh = createSSHClient(**cfg['ssh'])
        scp = SCPClient(ssh.get_transport())
        for element in cfg['files']['elements']:
            scp.get(remote_path=cfg['files']['remote_root'] + element)

        pcw.render_pc(cfg['files']['elements'], [], 0.0005, True, False)


if __name__ == '__main__':
    main('config.yaml')
