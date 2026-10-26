"""
Shared feature definitions for the Explainable AI Threat Detection System.

Both train_model.py (training) and app.py (inference + explanations) import this
module, so they always agree on feature names, order and wording.
All eight features are REAL network-flow measurements from CICIDS2017.
"""

CLASS_NAMES = {
    0: "Normal Activity",
    1: "Brute Force Attack",
    2: "Port Scanning",
    3: "Suspicious Attack",
}

# name    -> internal feature name used by the model
# column  -> column name in the CICIDS2017 CSV files (after stripping spaces)
FEATURES = [
    {"name": "Destination_Port", "column": "Destination Port", "label": "Destination Port",
     "description": "TCP/UDP port targeted by the network flow."},
    {"name": "Flow_Duration", "column": "Flow Duration", "label": "Flow Duration",
     "description": "Total duration of the flow, measured in microseconds."},
    {"name": "Fwd_Packets", "column": "Total Fwd Packets", "label": "Forward Packets",
     "description": "Number of packets sent from the source to the destination."},
    {"name": "Bwd_Packets", "column": "Total Backward Packets", "label": "Backward Packets",
     "description": "Number of packets returned from the destination to the source."},
    {"name": "Flow_Bytes_per_s", "column": "Flow Bytes/s", "label": "Flow Bytes per Second",
     "description": "Data transfer rate of the flow in bytes per second."},
    {"name": "Flow_Packets_per_s", "column": "Flow Packets/s", "label": "Flow Packets per Second",
     "description": "Packet rate of the flow in packets per second."},
    {"name": "SYN_Flag_Count", "column": "SYN Flag Count", "label": "SYN Flag Count",
     "description": "Number of TCP SYN flags (connection requests) seen in the flow."},
    {"name": "Avg_Packet_Size", "column": "Average Packet Size", "label": "Average Packet Size",
     "description": "Mean size of the packets in the flow, in bytes."},
]

FEATURE_NAMES = [f["name"] for f in FEATURES]
FEATURE_LABELS = {f["name"]: f["label"] for f in FEATURES}
RAW_COLUMNS = {f["column"]: f["name"] for f in FEATURES}

SERVICE_PORTS = {
    20: "FTP data", 21: "FTP", 22: "SSH", 23: "Telnet", 25: "SMTP", 53: "DNS", 80: "HTTP",
    110: "POP3", 123: "NTP", 135: "RPC", 139: "NetBIOS", 143: "IMAP", 389: "LDAP", 443: "HTTPS",
    445: "SMB", 1433: "MS SQL", 3306: "MySQL", 3389: "RDP", 8080: "HTTP alternate",
}


def _num(v):
    if abs(v - round(v)) < 1e-9 or abs(v) >= 100:
        return f"{v:,.0f}"
    return f"{v:,.2f}"


def format_value(name, value):
    """Human-readable value with units (used in tables and explanations)."""
    v = float(value)
    if name == "Destination_Port":
        return str(int(v))
    if name == "Flow_Duration":
        if v >= 1e6:
            return f"{v / 1e6:.2f} s"
        if v >= 1e3:
            return f"{v / 1e3:.1f} ms"
        return f"{v:,.0f} \u00b5s"
    if name == "Flow_Bytes_per_s":
        if v >= 1e6:
            return f"{v / 1e6:.2f} MB/s"
        if v >= 1e3:
            return f"{v / 1e3:.1f} KB/s"
        return f"{v:,.1f} B/s"
    if name == "Flow_Packets_per_s":
        return f"{v:,.1f} pkt/s"
    if name == "Avg_Packet_Size":
        return f"{v:,.1f} B"
    return _num(v)


def describe_feature(name, value, benign_median):
    """
    Turn one feature value into a plain-English sentence by comparing it with
    the median of normal (BENIGN) traffic measured on the training data.
    """
    label = FEATURE_LABELS[name]
    v = float(value)

    if name == "Destination_Port":
        port = int(v)
        service = SERVICE_PORTS.get(port)
        target = f"destination port {port} ({service})" if service else f"destination port {port} (a non-standard port)"
        return f"The traffic targets {target}, a pattern the model associates with this classification."

    shown = format_value(name, v)
    if benign_median is None:
        return f"{label} is {shown}."

    median = format_value(name, benign_median)
    if benign_median <= 0:
        if v == 0:
            return f"{label} is {shown}, in line with normal traffic."
        return f"{label} is {shown}, whereas normal traffic typically shows {median}."

    ratio = v / benign_median
    if ratio >= 1.5:
        return f"{label} is unusually high at {shown} (about {ratio:.1f}x the normal-traffic median of {median})."
    if ratio <= 0.67:
        return f"{label} is unusually low at {shown} (normal-traffic median: {median})."
    return f"{label} ({shown}) is close to the normal-traffic median of {median}, but it contributes in combination with other signals."