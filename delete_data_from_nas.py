import os
import shlex
import uuid
import paramiko

# -------------------- SSH DEVICE --------------------
hostname = "192.168.0.230"
username = "sigvet"
password = os.environ.get("DEVICE_SSH_PASSWORD", "sigvet123")  # Defaults to sigvet123 or from env

# -------------------- NAS SHARE --------------------
# Mount the SMB share first, then point this to the mounted folder.
candidate_paths = [
    "/Volumes/sigvet_data/sigvet_data/SIGALP-N002-SV-0086",
    "/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=sigvet_data/sigvet_data/SIGALP-N002-SV-0086",
    "/mnt/nas/sigvet_data/sigvet_data/SIGALP-N002-SV-0086",
]
nas_directory = os.environ.get(
    "NAS_DIRECTORY",
    next((p for p in candidate_paths if os.path.isdir(p)), candidate_paths[1]),
)

device_directory = "/imgarc/sigvet/tempfs"

# Keep True for the first run to preview matches without deleting anything.
# After checking the output, change to False to perform deletion.
DRY_RUN = False


def is_valid_uuid(val):
    try:
        u = uuid.UUID(str(val))
        return str(u).lower() == str(val).lower()
    except (ValueError, AttributeError):
        return False


def get_nas_zip_names(directory):
    """Return ZIP basenames without .zip, case-insensitively."""
    if not os.path.isdir(directory):
        raise FileNotFoundError(
            f"NAS directory not found: {directory}\n"
            "Mount the SMB share and update nas_directory."
        )

    names = set()
    for filename in os.listdir(directory):
        full_path = os.path.join(directory, filename)
        if os.path.isfile(full_path) and filename.lower().endswith(".zip"):
            names.add(filename[:-4])  # remove the final '.zip'
    return names


def main():
    if not password:
        raise RuntimeError(
            "Set DEVICE_SSH_PASSWORD in your environment before running."
        )

    nas_names = get_nas_zip_names(nas_directory)
    print(f"Found {len(nas_names)} ZIP files on NAS.")

    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.RejectPolicy())

    try:
        client.connect(
            hostname=hostname,
            username=username,
            password=password,
            timeout=15,
        )

        # List only direct child directories in tempfs.
        command = (
            "find " + shlex.quote(device_directory)
            + " -mindepth 1 -maxdepth 1 -type d -printf '%f\\n'"
        )
        stdin, stdout, stderr = client.exec_command(command)
        device_output = stdout.read().decode().splitlines()
        error_output = stderr.read().decode().strip()
        exit_code = stdout.channel.recv_exit_status()

        if exit_code != 0:
            raise RuntimeError(
                f"Could not list device directories: {error_output}"
            )

        raw_matches = sorted(set(device_output) & nas_names)
        matches = []
        for m in raw_matches:
            if is_valid_uuid(m):
                matches.append(m)
            else:
                print(f"[PROTECTED] Skipping non-scan/system directory: {m}")
        print(f"Found {len(matches)} exact NAS/device scan folder matches.")

        if not matches:
            print("Nothing to delete.")
            return

        for folder_name in matches:
            device_path = f"{device_directory}/{folder_name}"

            if DRY_RUN:
                print(f"[DRY RUN] Would delete: {device_path}")
                continue

            # Quote the complete path to prevent shell interpretation.
            delete_command = (
                "sudo -S rm -rf -- " + shlex.quote(device_path)
            )
            stdin, stdout, stderr = client.exec_command(
                delete_command, get_pty=True
            )
            stdin.write(password + "\n")
            stdin.flush()
            stdin.channel.shutdown_write()

            out = stdout.read().decode().strip()
            err = stderr.read().decode().strip()
            exit_code = stdout.channel.recv_exit_status()

            if exit_code == 0:
                print(f"DELETED: {device_path}")
            else:
                print(f"FAILED to delete: {device_path}")
                if out:
                    print("stdout:", out)
                if err:
                    print("stderr:", err)

    finally:
        client.close()


if __name__ == "__main__":
    main()
