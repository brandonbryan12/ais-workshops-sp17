import requests
import os

def performBotAction(i):
    res = "Testing"

    if i == 0:
        # trending
        res = "You have requested trending twitter tags"

    elif i == 1:
        # headlines
        res = "You have requested news headlines"

    elif i == 2:
        # gibberish
        res = "I don't know what you're saying"

    return res
