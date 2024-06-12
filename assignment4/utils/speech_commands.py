

COMMAND_OBJECTS = set([
    "staubsauger"
    "alarm",
    "lüftung",
    "ofen",
    "heizung",
    "fernseher",
    "licht",
    "radio"
])

COMMAND_ACTIONS = set(["an", "aus"])

def find_speech_commands(scene):
    detections = list()

    for cls, times in scene.items():
        detections.extend([(cls, t) for t in times])

    detections.sort(key=lambda x: x[1])

    commands = []

    latest_time = None
    latest_object = None

    for detection, time in detections:

        if detection in COMMAND_ACTIONS and latest_object != None and latest_time > time - 1.5:
            commands.append((latest_object, detection, latest_time, time))
        
        latest_object = detection
        latest_time = time

    return commands

def scene_cost(predicted_commands, true_commands):
   
    cost = 0

    p = 0
    t = 0

    while p < len(predicted_commands) or t < len(true_commands):

        # Command not deteced
        if p >= len(predicted_commands):
            cost += cost_missing_command(true_commands[t])
            t += 1
            continue

        # Additional command detected
        if t >= len(true_commands):
            cost += cost_additional_command(predicted_commands[p])
            p += 1
            continue

        true_object, true_action, true_start, true_end = true_commands[t]
        predicted_object, predicted_action, predicted_start, predicted_end = predicted_commands[p]

        true_avg_time = (true_start + true_end) / 2
        predicted_avg_time = (predicted_start + predicted_end) / 2
       
        if abs(true_avg_time - predicted_avg_time) < 1.5:
            cost += cost_match_command(true_object, true_action, predicted_object, predicted_action)
            p += 1
            t += 1
            continue

        if true_start + true_end > predicted_start + predicted_end:
            cost += cost_additional_command(predicted_commands[p])
            p +=1
            continue

        if true_start + true_end < predicted_start + predicted_end:
            cost += cost_missing_command(true_commands[t])
            t +=1
            continue

        raise Error("Cost matching failed")

    return cost

def cost_missing_command(true_command):
    return 0.5

def cost_additional_command(predicted_command):
    command_object, _, _, _ = predicted_command
    
    if command_object in set(["fernseher", "licht", "radio", "staubsauger"]):
        return 2

    if command_object in set(["heizung", "lüftung"]):
        return 3

    if command_object in set(["ofen", "alarm"]):
        return 4

    raise Error(f"Invalid command object: {command_object}")

def cost_match_command(true_object, true_action, predicted_object, predicted_action):
    
    if true_object == predicted_object:
        if true_action == predicted_action:
            return -1
        else:
            return 0.1
    else:
        return 1
