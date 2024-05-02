CLASSES = {
    "uninteresting": 0,
    "staubsauger": 1,
    "alarm": 2,
    "lüftung": 3,
    "ofen": 4,
    "heizung": 5,
    "fernseher": 6,
    "licht": 7,
    "aus": 8,
    "an": 9,
    "radio": 10,
}

REVERSE_CLASSES = {v: k for k, v in CLASSES.items()}

def label_to_class(label):
    if label in CLASSES:
        return CLASSES[label]
    else:
        return CLASSES["uninteresting"]

def class_to_label(number):
    if number in REVERSE_CLASSES:
        return REVERSE_CLASSES[number]
    else:
        return REVERSE_CLASSES[0]