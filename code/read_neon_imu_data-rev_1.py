import csv
import glob
import logging
import os
import re
from collections import OrderedDict

import numpy as np
from natsort import natsorted

import imu_pb2

log = logging.getLogger(__name__)


def filereader(imu_raw_file, start_byte=0, stop_byte=-1):
    with open(imu_raw_file, "rb") as fp:
        data = fp.read()

    return data


def create_file_lookup_tree(filenames):
    """
    Convert a recording file list to a sanitized lookup tree structure
    >>> create_file_lookup_tree(['some file.txt', 'some file.csv', 'some other.txt'])
    {
        "some": {
            "file": {
                "txt": "some file.txt",
                "csv": "some file.csv"
            },
            "other": {
                "txt": "some other.txt"
            }
        }
    }
    """
    tree = OrderedDict()
    # split on any non word character
    regex = re.compile(r"[^\w]")
    for filename in natsorted(filenames, key=lambda f: f.lower()):
        sanitized_filename = filename.lower()
        node = tree
        parts = re.split(regex, sanitized_filename)
        for part in parts[:-1]:
            if part not in node:
                node[part] = OrderedDict()
            node = node[part]
        node[parts[-1]] = filename
    return tree


def parse_neon_imu_raw_packets(filename, filereader):
    buffer = filereader(filename)
    index = 0
    packet_sizes = []
    while True:
        nums = np.frombuffer(buffer[index : index + 2], np.uint16)
        if not nums:
            break
        index += 2
        packet_size = nums[0]
        packet_sizes.append(packet_size)
        packet_bytes = buffer[index : index + packet_size]
        index += packet_size
        packet = imu_pb2.ImuPacket()
        packet.ParseFromString(packet_bytes)
        yield packet


def read_neon_imu_datapoints(rec_path, filereader):
    """
    Returns a generator for all imu samples found in a recording
    Example:
    {
        "timestamp_ns": timestamp_ns,
        "gyro_x": gyro_x,
        "gyro_y": gyro_y,
        "gyro_z": gyro_z,
        "accel_x": accel_x,
        "accel_y": accel_y,
        "accel_z": accel_z,
        "rotvec_w": rotvec_w,
        "rotvec_x": rotvec_x,
        "rotvec_y": rotvec_y,
        "rotvec_z": rotvec_z,
    }
    """

    filenames = {
        os.path.basename(file_path): file_path
        for file_path in glob.iglob(os.path.join(rec_path, "*"))
    }

    log.debug("reading imu samples")
    lookup = create_file_lookup_tree(filenames)
    log.debug(f"file lookup: {lookup}")

    proto_file = (lookup.get("imu") or {}).get("proto")
    if proto_file:
        for ps in lookup.get("extimu") or {}:
            if "ps" not in ps:
                log.error(f"invalid imu file prefix: {ps}")
                continue

            imu_ps_files = lookup["extimu"][ps]
            if "raw" not in imu_ps_files:
                log.error(f"missing imu raw file for {ps}")
                continue

            imu_raw_file = imu_ps_files["raw"]
            imu_raw_vals_stream = parse_neon_imu_raw_packets(
                filenames[imu_raw_file], filereader
            )

            for packet in imu_raw_vals_stream:
                datapoint = {
                    "timestamp_ns": packet.tsNs,
                    "gyro_x": packet.gyroData.x,
                    "gyro_y": packet.gyroData.y,
                    "gyro_z": packet.gyroData.z,
                    "accel_x": packet.accelData.x,
                    "accel_y": packet.accelData.y,
                    "accel_z": packet.accelData.z,
                    "rotvec_w": packet.rotVecData.w,
                    "rotvec_x": packet.rotVecData.x,
                    "rotvec_y": packet.rotVecData.y,
                    "rotvec_z": packet.rotVecData.z,
                }

                log.debug(f"dp: {datapoint}")
                yield datapoint


if __name__ == "__main__":

    os.chdir('/Users/jyeon/Documents/GitHub/pilot_analysis/')
    rec_path = os.path.join(os.getcwd(), 'data/imu_testing')

    with open("imu.csv", "w") as fp:
        csv_writer = csv.writer(fp)
        csv_writer.writerow(
            [
                "timestamp_ns",
                "gyro_x",
                "gyro_y",
                "gyro_z",
                "accel_x",
                "accel_y",
                "accel_z",
                "rotvec_w",
                "rotvec_x",
                "rotvec_y",
                "rotvec_z",
            ]
        )

        for imu_data in read_neon_imu_datapoints(
            rec_path=rec_path, filereader=filereader
        ):
            csv_writer.writerow(imu_data.values())
