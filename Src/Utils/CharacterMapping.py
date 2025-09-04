def CreateCharacterMapping():
    """Create mapping from class indices to characters (0-9, A-Z)"""
    # Digits 0-9
    Characters = [str(i) for i in range(10)]
    # Uppercase letters A-Z
    Characters.extend([chr(i) for i in range(ord('A'), ord('Z') + 1)])
    return {i: char for i, char in enumerate(Characters)}