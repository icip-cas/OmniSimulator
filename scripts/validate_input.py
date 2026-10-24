"""Read-only structural and temporal checks; no model calls or runtime imports."""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


SCENES = {"视频浏览", "直播间", "商城购物", "广告推荐", "电商客服对话", "搜索行为"}
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def require(condition, location, message):
    if not condition:
        raise ValueError(f"{location}: {message}")


def check_event(event, location):
    require(isinstance(event, dict), location, "expected an event object")
    timestamp = event.get("timestamp")
    require(isinstance(timestamp, str), location, "timestamp must be a string")
    try:
        parsed = datetime.strptime(timestamp, TIME_FORMAT)
    except ValueError:
        raise ValueError(f"{location}.timestamp: expected YYYY-MM-DD HH:MM:SS") from None
    require(parsed.strftime(TIME_FORMAT) == timestamp, location, "use YYYY-MM-DD HH:MM:SS")
    require(isinstance(event.get("type"), str) and event["type"] in SCENES,
            location, "unsupported scene identifier")
    require(isinstance(event.get("context"), dict), location, "context must be an object")
    actions = event.get("action")
    require(isinstance(actions, list), location, "action must be a list of objects")
    for index, action in enumerate(actions):
        require(isinstance(action, dict) and isinstance(action.get("type"), str)
                and bool(action["type"]), f"{location}.action[{index}]", "expected an action with a type")
    return parsed


def check_history(events, location, ordered=True):
    require(isinstance(events, list), location, "expected a list of events")
    times = [check_event(event, f"{location}[{index}]") for index, event in enumerate(events)]
    if ordered:
        require(times == sorted(times), location, "events must be in chronological order")
    return times


def check_user(user, location):
    require(isinstance(user, dict), location, "expected a user object")
    require(isinstance(user.get("user_profile", ""), str), location, "user_profile must be text")


def validate_prepared(data):
    users = data.get("users") if isinstance(data, dict) else data
    require(isinstance(users, list) and bool(users), "users", "expected a nonempty list")
    targets = 0
    for index, user in enumerate(users):
        location = f"users[{index}]"
        check_user(user, location)
        if "test_actions" not in user:
            history = check_history(user.get("action_history", []), f"{location}.action_history")
            target = check_event(user.get("test_action"), f"{location}.test_action")
            require(all(time < target for time in history), location,
                    "history must strictly precede the target")
            targets += 1
            continue

        base = user.get("base_history", [])
        window = user.get("test_time_all_actions", [])
        base_times = check_history(base, f"{location}.base_history")
        window_times = check_history(window, f"{location}.test_time_all_actions")
        require(not base_times or not window_times or base_times[-1] < window_times[0],
                location, "base_history must strictly precede the target window")
        selected = user["test_actions"]
        require(isinstance(selected, list) and bool(selected), f"{location}.test_actions",
                "expected a nonempty list")
        seen = set()
        for item_index, item in enumerate(selected):
            item_location = f"{location}.test_actions[{item_index}]"
            require(isinstance(item, dict), item_location, "expected a target object")
            position = item.get("test_time_index")
            require(type(position) is int and 0 <= position < len(window), item_location,
                    "test_time_index must be a zero-based index into test_time_all_actions")
            require(position not in seen, item_location, "duplicate target index")
            seen.add(position)
            require(item.get("action") == window[position], item_location,
                    "action must match the event at test_time_index")
            require(all(time < window_times[position] for time in base_times + window_times[:position]),
                    item_location, "history must strictly precede the target")
            targets += 1
    return len(users), targets


def read_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def validate_path(path):
    """Return a summary, raising ValueError for a malformed or empty input."""
    path = Path(path)
    if path.is_file():
        users, targets = validate_prepared(read_json(path))
        return f"Prepared input: {users} user entries, {targets} target events."
    require(path.is_dir(), str(path), "input path does not exist or is not a file/directory")
    files = sorted(path.glob("*.json"))
    require(bool(files), str(path), "no JSON files found")
    seen = set()
    eligible = 0
    for file in files:
        data = read_json(file)
        require(isinstance(data, dict), str(file), "expected an object mapping user IDs to records")
        for index, (uid, user) in enumerate(data.items()):
            location = f"{file.name}: user entry {index}"
            require(uid not in seen, location, "duplicate user ID across files")
            seen.add(uid)
            check_user(user, location)
            # The generator sorts raw histories, so their input order is unrestricted.
            times = check_history(user.get("action_history"), f"{location}.action_history", ordered=False)
            eligible += len(times) >= 20
    require(eligible > 0, str(path), "no users have the 20 events required by raw-directory mode")
    return f"Raw input: {len(files)} files, {len(seen)} users, {eligible} users with at least 20 events."


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path", type=Path, help="Prepared JSON file or raw JSON directory")
    args = parser.parse_args(argv)
    try:
        summary = validate_path(args.input_path)
    except (ValueError, OSError) as exc:
        print(f"Invalid input: {exc}", file=sys.stderr)
        return 1
    print(f"OK: {summary}")
    print("Checked structure and temporal boundaries; scene-specific field semantics are not checked.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
