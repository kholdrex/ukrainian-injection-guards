"""Evaluation inputs: every text a detector sees, with its condition and label."""


def build_inputs(items: list[dict]) -> list[dict]:
    rows = []

    def add(item_id, group, condition, label, text):
        rows.append({"input_id": f"{item_id}/{condition}", "item_id": item_id, "group": group,
                     "condition": condition, "label": label, "text": text})

    for item in items:
        if item["group"] == "calibration":
            add(item["id"], "calibration", "en", 0, item["en"])
            add(item["id"], "calibration", "uk", 0, item["uk"])
        elif item["group"] == "main":
            add(item["id"], "main", "en", item["label"], item["en"])
            add(item["id"], "main", "uk", item["label"], item["uk"])
            add(item["id"], "main", "uk_alt", item["label"], item["uk_alt"])
            add(item["id"], "main", "pivot", item["label"], item["pivot"])

    carriers = [item for item in items if item["group"] == "carrier"]
    payloads = [item for item in items if item["group"] == "payload"]
    for carrier, payload in zip(carriers, payloads):
        add(carrier["id"], "embedded", "clean_en", 0, carrier["en"])
        add(carrier["id"], "embedded", "clean_uk", 0, carrier["uk"])
        for host in ("en", "uk"):
            for attack in ("en", "uk"):
                add(carrier["id"], "embedded", f"{host}+{attack}", 1, f"{carrier[host]}\n\n{payload[attack]}")
    return rows
